"""Separación en las GPUs de Modal.

La GPU se elige al desplegar con SPLITTER_GPU, y cada GPU es una app distinta de Modal
para que puedan convivir y compararse sin redesplegar:
    SPLITTER_GPU=L4 .venv/bin/modal deploy modal_app.py    # -> app "splitter-gpu-l4"
    SPLITTER_GPU=T4 .venv/bin/modal deploy modal_app.py    # -> app "splitter-gpu-t4"
Hay que redesplegar cada vez que cambies este archivo o algo dentro de app/.

El backend apunta a una de ellas con SPLITTER_ENGINE=modal y SPLITTER_MODAL_APP=<nombre>.
Dentro del contenedor se reutiliza el mismo RealEngine que corre en local; los modelos se guardan
en un Volume (compartido por las dos apps) para descargarlos una sola vez.
"""
from __future__ import annotations

import os

import modal

SUPPORTED_GPUS = ("T4", "L4")
GPU = os.environ.get("SPLITTER_GPU", "L4").upper()
if GPU not in SUPPORTED_GPUS:
    raise ValueError(f"SPLITTER_GPU debe ser uno de {SUPPORTED_GPUS}, no {GPU!r}")

APP_NAME = f"splitter-gpu-{GPU.lower()}"
MODEL_DIR = "/models"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("audio-separator[gpu]==0.47.0", "audioread")  # librosa 1.0 ya no instala audioread
    # SPLITTER_GPU va en la imagen para que el contenedor, al reimportar este archivo, vea la misma app.
    # Los ajustes del modelo se leen dentro del contenedor, así que se toman de tu entorno al desplegar.
    .env({
        "SPLITTER_MODEL_DIR": MODEL_DIR,
        "SPLITTER_DATA_DIR": "/tmp/splitter",
        "SPLITTER_GPU": GPU,
        "SPLITTER_AUTOCAST_2STEM": os.environ.get("SPLITTER_AUTOCAST_2STEM", "1"),
        "SPLITTER_MDXC_OVERLAP": os.environ.get("SPLITTER_MDXC_OVERLAP", "4"),
    })
    .add_local_python_source("app")
)

models = modal.Volume.from_name("splitter-models", create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=GPU,
    volumes={MODEL_DIR: models},
    timeout=20 * 60,
    # El contenedor sigue vivo 2 min tras el último trabajo (modelo ya cargado). Esa espera se cobra:
    # en L4, 5 min costaban ~3 veces una canción; con 2 min un arranque en frío suma ~15-25 s.
    scaledown_window=2 * 60,
)
class Separator:
    @modal.enter()
    def setup(self) -> None:
        from app.engine import RealEngine

        self.engine = RealEngine()

    @modal.method()
    def device_info(self) -> dict:
        return self.engine.device_info()

    @modal.method()
    def separate(self, audio: bytes, filename: str, preset: str):
        """Generador. Emite, en orden:
          {"type": "progress", "value": 0..1}           mientras separa
          {"type": "stem", "name": str, "data": bytes}  una por pista, en FLAC (sin pérdida, ~la mitad que WAV)
          {"type": "done", "gpu": str, "gpu_seconds": float, "load_seconds": float, "separate_seconds": float}
        `gpu_seconds` es todo lo que corre esta llamada en el contenedor (incluye codificar y mandar
        las pistas); no incluye el arranque del contenedor."""
        import queue
        import subprocess
        import tempfile
        import threading
        import time
        from pathlib import Path

        started = time.perf_counter()
        work = Path(tempfile.mkdtemp())
        src = work / Path(filename).name
        src.write_bytes(audio)
        out_dir = work / "out"

        events: queue.Queue = queue.Queue()
        result: dict = {}

        def run() -> None:
            try:
                files_before = set(os.listdir(MODEL_DIR)) if os.path.isdir(MODEL_DIR) else set()
                result["files"] = self.engine.separate(src, out_dir, preset, lambda f: events.put(f))
                if set(os.listdir(MODEL_DIR)) != files_before:
                    models.commit()  # guardar el modelo recién descargado para próximos contenedores
            except BaseException as exc:  # noqa: BLE001 - se re-lanza en el hilo principal
                result["error"] = exc
            finally:
                events.put(None)

        threading.Thread(target=run, daemon=True).start()
        last = -1.0
        while (item := events.get()) is not None:
            if item - last >= 0.01:
                last = item
                yield {"type": "progress", "value": item}
        if "error" in result:
            raise result["error"]

        for stem, wav in result["files"].items():
            flac = wav.with_suffix(".flac")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "flac", str(flac)],
                check=True,
            )
            yield {"type": "stem", "name": stem, "data": flac.read_bytes()}

        timings = self.engine.last_timings
        yield {
            "type": "done",
            "gpu": GPU,
            "gpu_seconds": time.perf_counter() - started,
            "load_seconds": timings["load_seconds"],
            "separate_seconds": timings["separate_seconds"],
        }
