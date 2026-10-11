"""Cola de trabajos: un solo worker procesa de uno en uno (la GPU es una sola)."""
from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import config

log = logging.getLogger("splitter.jobs")

QUEUED, PROCESSING, DONE, ERROR = "queued", "processing", "done", "error"


@dataclass
class Job:
    id: str
    original_name: str
    preset: str
    wanted_stems: List[str]
    input_path: Path
    out_dir: Path
    status: str = QUEUED
    progress: Optional[float] = 0.0  # 0..1, None = indeterminado
    message: str = "En cola"
    error: Optional[str] = None
    stem_files: Dict[str, Path] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    version: int = 0  # sube con cada cambio; el stream de eventos lo usa para saber si hay novedades

    def touch(self) -> None:
        self.version += 1


class JobManager:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.jobs: Dict[str, Job] = {}
        self._order: List[str] = []  # ids en cola, para calcular la posición
        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=config.MAX_QUEUE)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="splitter-worker", daemon=True)
        self._cleaner = threading.Thread(target=self._clean_loop, name="splitter-cleaner", daemon=True)

    # -- ciclo de vida -------------------------------------------------------
    def start(self) -> None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._worker.start()
        self._cleaner.start()

    def stop(self) -> None:
        self._stop.set()

    # -- API para las rutas --------------------------------------------------
    def new_job_dir(self) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex
        d = config.DATA_DIR / job_id
        d.mkdir(parents=True, exist_ok=True)
        return job_id, d

    def submit(self, job: Job) -> None:
        """Encola el trabajo. Lanza queue.Full si la cola está llena."""
        with self._lock:
            self._queue.put_nowait(job.id)
            self.jobs[job.id] = job
            self._order.append(job.id)

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def queue_position(self, job_id: str) -> Optional[int]:
        with self._lock:
            if job_id in self._order:
                return self._order.index(job_id) + 1
        return None

    def delete(self, job_id: str) -> bool:
        job = self.jobs.pop(job_id, None)
        if job is None:
            return False
        with self._lock:
            if job_id in self._order:
                self._order.remove(job_id)
        shutil.rmtree(config.DATA_DIR / job_id, ignore_errors=True)
        return True

    # -- worker ---------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            job = self.jobs.get(job_id)
            with self._lock:
                if job_id in self._order:
                    self._order.remove(job_id)
            if job is None:  # fue borrado mientras esperaba
                continue
            self._process(job)

    def _process(self, job: Job) -> None:
        job.status, job.message, job.progress = PROCESSING, "Separando audio…", 0.0
        job.touch()

        def on_progress(fraction: float) -> None:
            # monótono y con tope en 95%: el 100% solo llega cuando termina de verdad
            new = min(0.95, max(job.progress or 0.0, fraction * 0.95))
            if new - (job.progress or 0.0) >= 0.01:
                job.progress = new
                job.touch()

        try:
            files = self.engine.separate(job.input_path, job.out_dir, job.preset, on_progress)
            # Solo se conservan las pistas que el usuario pidió (idea 2).
            keep = {s: p for s, p in files.items() if s in job.wanted_stems}
            for s, p in files.items():
                if s not in keep:
                    p.unlink(missing_ok=True)
            job.stem_files = keep
            job.status, job.progress, job.message = DONE, 1.0, "Listo"
        except Exception as exc:  # noqa: BLE001 - cualquier fallo del modelo debe llegar a la página
            log.exception("Falló el trabajo %s", job.id)
            job.status, job.progress, job.message = ERROR, None, "Error al procesar el audio"
            job.error = str(exc)[:300]
        finally:
            job.finished_at = time.time()
            job.input_path.unlink(missing_ok=True)  # no guardamos el original más de lo necesario
            job.touch()

    # -- limpieza -------------------------------------------------------------
    def _clean_loop(self) -> None:
        while not self._stop.wait(60):
            cutoff = time.time() - config.JOB_TTL_MINUTES * 60
            for job_id, job in list(self.jobs.items()):
                if job.status in (DONE, ERROR) and (job.finished_at or job.created_at) < cutoff:
                    log.info("Borrando trabajo expirado %s", job_id)
                    self.delete(job_id)
