"""Límites que protegen el crédito de la GPU: duración máxima del audio y trabajos por hora por IP.

Usa el motor simulado (no necesita GPU ni modelos).

Uso:  python tests/limits_test.py
"""
import atexit
import io
import os
import shutil
import sys
import tempfile
import wave

os.environ["SPLITTER_ENGINE"] = "fake"
os.environ["SPLITTER_DATA_DIR"] = DATA_DIR = tempfile.mkdtemp(prefix="splitter-test-")
atexit.register(shutil.rmtree, DATA_DIR, ignore_errors=True)  # también si la prueba falla
os.environ["SPLITTER_MAX_AUDIO_MINUTES"] = "1"
os.environ["SPLITTER_MAX_JOBS_PER_HOUR"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402
from app.ratelimit import RateLimiter  # noqa: E402


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


def post(c, data: bytes, name="a.wav", headers=None):
    return c.post("/api/jobs", files={"file": (name, data, "audio/wav")}, headers=headers or {})


with TestClient(main.app) as c:
    cfg = c.get("/api/config").json()
    check(cfg["max_audio_minutes"] == 1 and cfg["max_jobs_per_hour"] == 3, f"/api/config informa los límites: {cfg['max_audio_minutes']} min, {cfg['max_jobs_per_hour']}/h")

    # Duración
    r = post(c, make_wav(61))
    check(r.status_code == 413 and "dura 1:01" in r.json()["detail"], f"61 s con límite de 1 min -> {r.status_code}: {r.json()['detail']}")
    r = post(c, b"esto no es audio" * 100, name="x.mp3")
    check(r.status_code == 415 and "no se pudo leer" in r.json()["detail"].lower(), f"archivo dañado -> {r.status_code}: {r.json()['detail']}")
    r = post(c, make_wav(60))
    check(r.status_code == 202, f"60 s justos -> {r.status_code}")

    # Trabajos por hora (los rechazos de arriba no contaron: queda 1 de 3 usado)
    for i in (2, 3):
        r = post(c, make_wav())
        check(r.status_code == 202, f"trabajo {i} de 3 -> {r.status_code}")
    r = post(c, make_wav())
    check(r.status_code == 429, f"trabajo 4 -> {r.status_code}: {r.json()['detail']}")
    check(int(r.headers.get("retry-after", 0)) > 3000, f"Retry-After = {r.headers.get('retry-after')} s")

    # Sin SPLITTER_BEHIND_PROXY, X-Forwarded-For no sirve para saltarse el límite
    r = post(c, make_wav(), headers={"X-Forwarded-For": "1.2.3.4"})
    check(r.status_code == 429, f"X-Forwarded-For inventado sin proxy -> {r.status_code}")

    # Detrás de un proxy, cada usuario tiene su propio límite
    main.config.BEHIND_PROXY = True
    r = post(c, make_wav(), headers={"X-Forwarded-For": "1.2.3.4, 10.0.0.1"})
    check(r.status_code == 202, f"con proxy, otra IP -> {r.status_code}")
    main.config.BEHIND_PROXY = False

# RateLimiter por separado: ventana deslizante y 0 = sin límite
rl = RateLimiter(2, window_seconds=0.2)
check(rl.acquire("a") == 0 and rl.acquire("a") == 0 and rl.acquire("a") > 0, "ventana: el tercero espera")
check(rl.acquire("b") == 0, "otra IP no se ve afectada")
import time  # noqa: E402

time.sleep(0.25)
check(rl.acquire("a") == 0, "al pasar la ventana se libera")
check(all(RateLimiter(0, 1).acquire("a") == 0 for _ in range(50)), "0 = sin límite")

print("\nTodo bien.")
