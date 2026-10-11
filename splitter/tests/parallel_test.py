"""Varios trabajos a la vez: con SPLITTER_PARALLEL_JOBS=3 corren 3 y el cuarto espera en cola.

Usa el motor simulado (cada trabajo tarda ~2 s; no necesita GPU ni modelos).

Uso:  python tests/parallel_test.py
"""
import atexit
import io
import os
import shutil
import sys
import tempfile
import time
import wave

os.environ["SPLITTER_ENGINE"] = "fake"
os.environ["SPLITTER_DATA_DIR"] = DATA_DIR = tempfile.mkdtemp(prefix="splitter-test-")
atexit.register(shutil.rmtree, DATA_DIR, ignore_errors=True)  # también si la prueba falla
os.environ["SPLITTER_PARALLEL_JOBS"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402


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


with TestClient(main.app) as c:
    start = time.time()
    ids = []
    for i in range(4):
        r = c.post("/api/jobs", files={"file": (f"c{i}.wav", make_wav(), "audio/wav")})
        check(r.status_code == 202, f"trabajo {i + 1} aceptado")
        ids.append(r.json()["id"])

    time.sleep(0.6)  # los workers ya tomaron lo suyo; ninguno terminó (tardan ~2 s)
    jobs = [c.get(f"/api/jobs/{i}").json() for i in ids]
    states = [j["status"] for j in jobs]
    check(states.count("processing") == 3, f"3 procesando a la vez: {states}")
    queued = [j for j in jobs if j["status"] == "queued"]
    check(len(queued) == 1 and queued[0]["queue_position"] == 1, "el cuarto espera en la posición 1")

    while not all(c.get(f"/api/jobs/{i}").json()["status"] == "done" for i in ids):
        time.sleep(0.1)
        if time.time() - start > 20:
            check(False, "los 4 trabajos terminan")
    took = time.time() - start
    # En fila serían ~8 s (4 × 2 s); con 3 en paralelo, ~4 s (dos tandas).
    check(took < 6.5, f"los 4 terminan en {took:.1f} s (en fila serían ~8 s)")

print("\nTodo bien.")
