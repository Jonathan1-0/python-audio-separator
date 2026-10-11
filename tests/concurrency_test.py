"""Peticiones simultáneas a la misma pista/formato (vista previa, descargas y ZIP).

Comprueba que dos conversiones a la vez no se pisan el archivo temporal: antes daban 500
o, peor, un 200 con el archivo vacío. Usa el motor simulado (no necesita GPU ni modelos).

Uso:  python tests/concurrency_test.py
"""
import atexit
import io
import os
import shutil
import sys
import tempfile
import time
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["SPLITTER_ENGINE"] = "fake"
os.environ["SPLITTER_DATA_DIR"] = DATA_DIR = tempfile.mkdtemp(prefix="splitter-test-")
atexit.register(shutil.rmtree, DATA_DIR, ignore_errors=True)  # también si la prueba falla
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

ROUNDS = 5  # cada ronda es un trabajo nuevo, para que no haya nada en caché
COPIES = 4  # peticiones idénticas lanzadas a la vez
FORMATS = ["mp3", "flac", "ogg", "m4a"]


def make_wav(seconds=1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x01" * 8000 * seconds)
    return buf.getvalue()


def check(cond, msg):
    print(("OK   " if cond else "FALLA"), msg)
    if not cond:
        sys.exit(1)


def wait_done(c, job_id):
    for _ in range(100):
        st = c.get(f"/api/jobs/{job_id}").json()
        if st["status"] in ("done", "error"):
            return st
        time.sleep(0.2)
    return st


# Sin relanzar excepciones: un 500 debe contarse como fallo, no cortar la prueba.
with TestClient(app, raise_server_exceptions=False) as c:
    for rnd in range(1, ROUNDS + 1):
        r = c.post("/api/jobs", files={"file": ("rock.wav", make_wav(), "audio/wav")}, data={"preset": "4stem"})
        job_id = r.json()["id"]
        st = wait_done(c, job_id)
        check(st["status"] == "done", f"ronda {rnd}: trabajo listo")

        urls = [f"/api/jobs/{job_id}/stems/vocals/preview"] * COPIES * 2
        urls += [f"/api/jobs/{job_id}/stems/drums/download?format={fmt}" for fmt in FORMATS for _ in range(COPIES)]
        urls += [f"/api/jobs/{job_id}/download?format={fmt}" for fmt in ("mp3", "ogg") for _ in range(COPIES)]
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            responses = list(pool.map(c.get, urls))

        failed = [(u.split(job_id)[1], r.status_code, len(r.content)) for u, r in zip(urls, responses)
                  if r.status_code != 200 or len(r.content) < 500]
        check(not failed, f"ronda {rnd}: {len(urls)} peticiones simultáneas, todas 200 y con contenido"
              + (f" -> {len(failed)} fallaron, p. ej. {failed[:3]}" if failed else ""))

        zips = [r for u, r in zip(urls, responses) if u.split("?")[0].endswith(f"{job_id}/download")]
        check(all(zipfile.ZipFile(io.BytesIO(r.content)).testzip() is None for r in zips), f"ronda {rnd}: los ZIP están completos")

leftovers = list(Path(DATA_DIR).rglob("*.part"))
check(not leftovers, f"no quedan temporales .part ({len(leftovers)})")

print("\nTodo bien.")
