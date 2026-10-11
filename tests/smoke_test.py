"""Prueba de punta a punta con el motor simulado (no necesita GPU ni modelos).

Uso:  python tests/smoke_test.py
"""
import atexit
import io
import json
import os
import shutil
import sys
import tempfile
import time
import wave
import zipfile

os.environ["SPLITTER_ENGINE"] = "fake"
os.environ["SPLITTER_DATA_DIR"] = DATA_DIR = tempfile.mkdtemp(prefix="splitter-test-")
atexit.register(shutil.rmtree, DATA_DIR, ignore_errors=True)  # también si la prueba falla
os.environ["SPLITTER_MAX_UPLOAD_MB"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


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


with TestClient(app) as c:
    r = c.get("/api/health")
    check(r.status_code == 200 and r.json()["ffmpeg"], f"health + ffmpeg -> {r.json()}")

    cfg = c.get("/api/config").json()
    check(cfg["max_upload_mb"] == 1 and "mp3" in cfg["output_formats"], "config expone límites y formatos")

    # --- validaciones ---
    r = c.post("/api/jobs", files={"file": ("a.txt", b"hola", "text/plain")})
    check(r.status_code == 415, "rechaza extensión no soportada (415)")

    r = c.post("/api/jobs", files={"file": ("a.wav", b"x" * (2 * 1024 * 1024), "audio/wav")})
    check(r.status_code == 413, "rechaza archivo que pasa el límite (413)")

    r = c.post("/api/jobs", files={"file": ("a.wav", make_wav(), "audio/wav")}, data={"stems": "guitarra"})
    check(r.status_code == 400, "rechaza pista inexistente (400)")

    # --- flujo normal: solo vocals, preset 2stem ---
    r = c.post("/api/jobs", files={"file": ("Mi canción.wav", make_wav(), "audio/wav")}, data={"preset": "2stem", "stems": "vocals"})
    check(r.status_code == 202, f"crea trabajo (202) -> {r.json()['id']}")
    job_id = r.json()["id"]

    # progreso en tiempo real por SSE
    progress_seen, final = [], None
    with c.stream("GET", f"/api/jobs/{job_id}/events") as s:
        event = None
        for line in s.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event == "update":
                final = json.loads(line.split(":", 1)[1])
                progress_seen.append(final["progress"])
    check(final and final["status"] == "done", f"SSE termina en 'done' (progresos vistos: {len(progress_seen)})")
    check(len(set(progress_seen)) > 2 and progress_seen == sorted(p for p in progress_seen if p is not None), "el progreso sube de forma monótona")
    check([s["name"] for s in final["stems"]] == ["vocals"], "solo conserva la pista pedida (vocals)")

    # vista previa (mp3) y descargas en varios formatos
    r = c.get(final["stems"][0]["preview_url"])
    check(r.status_code == 200 and r.headers["content-type"] == "audio/mpeg" and len(r.content) > 500, "vista previa MP3")

    for fmt, mime in [("wav", "audio/wav"), ("mp3", "audio/mpeg"), ("flac", "audio/flac"), ("ogg", "audio/ogg"), ("m4a", "audio/mp4")]:
        r = c.get(f"/api/jobs/{job_id}/stems/vocals/download", params={"format": fmt})
        check(r.status_code == 200 and r.headers["content-type"].startswith(mime) and len(r.content) > 500, f"descarga {fmt}")

    r = c.get(f"/api/jobs/{job_id}/stems/vocals/download", params={"format": "exe"})
    check(r.status_code == 400, "rechaza formato inválido")
    r = c.get(f"/api/jobs/{job_id}/stems/drums/download")
    check(r.status_code == 404, "pista no pedida -> 404")

    # --- 4 pistas + ZIP ---
    r = c.post("/api/jobs", files={"file": ("rock.mp3", make_wav(), "audio/mpeg")}, data={"preset": "4stem"})
    job2 = r.json()["id"]
    for _ in range(100):
        st = c.get(f"/api/jobs/{job2}").json()
        if st["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    check(st["status"] == "done" and len(st["stems"]) == 4, "4 pistas con preset 4stem")
    r = c.get(f"/api/jobs/{job2}/download", params={"format": "mp3"})
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    check(r.status_code == 200 and len(names) == 4 and all(n.endswith(".mp3") for n in names), f"ZIP con 4 mp3 -> {names}")

    # --- borrar ---
    r = c.delete(f"/api/jobs/{job_id}")
    check(r.status_code == 204 and c.get(f"/api/jobs/{job_id}").status_code == 404, "DELETE borra el trabajo")

print("\nTodo bien.")
