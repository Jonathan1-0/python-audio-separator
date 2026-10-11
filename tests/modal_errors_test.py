"""Los fallos de Modal llegan a la página como mensajes entendibles (no como el error técnico).

No usa la GPU ni gasta crédito: simula cada fallo reemplazando la llamada a Modal.
Necesita el paquete `modal` instalado (está en requirements.txt).

Uso:  python tests/modal_errors_test.py
"""
import atexit
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ["SPLITTER_ENGINE"] = "fake"  # la app arranca con el simulado; luego se cambia el motor
os.environ["SPLITTER_DATA_DIR"] = DATA_DIR = tempfile.mkdtemp(prefix="splitter-test-")
atexit.register(shutil.rmtree, DATA_DIR, ignore_errors=True)  # también si la prueba falla
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal.exception as me  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402
from app.engine import ModalEngine, SeparationError  # noqa: E402


def check(cond, msg):
    print(("OK   " if cond else "FALLA"), msg)
    if not cond:
        sys.exit(1)


def engine_that_raises(exc):
    """ModalEngine sin conexión real: la llamada remota lanza `exc`."""
    eng = ModalEngine.__new__(ModalEngine)
    eng.app_name, eng.last_stats = "splitter-gpu-test", {}

    class Gen:
        def remote_gen(self, *a, **k):
            raise exc

    class Obj:
        separate = Gen()

    eng._cls = lambda: Obj()
    return eng


def engine_without_stems():
    eng = engine_that_raises(None)

    class Gen:
        def remote_gen(self, *a, **k):
            yield {"type": "progress", "value": 1.0}

    eng._cls = lambda: type("Obj", (), {"separate": Gen()})()
    return eng


CASES = [
    ("tiempo agotado", me.FunctionTimeoutError("Function timed out"), "tardó demasiado"),
    ("crédito agotado", me.ResourceExhaustedError("quota"), "límite de uso"),
    ("límite de gasto", me.Error("Workspace billing cycle spend limit reached"), "límite de uso"),
    ("app no desplegada", me.NotFoundError("App 'splitter-gpu-x' not found"), "no está disponible"),
    ("token inválido", me.AuthError("Token missing"), "no está disponible"),
    ("sin conexión", me.ConnectionError("could not connect"), "No se pudo conectar"),
    ("archivo enorme", me.RequestSizeError("too large"), "demasiado grande"),
    ("falla del modelo", ValueError("cannot reshape array"), "No se pudo procesar el audio"),
]

for name, exc, expected in CASES:
    try:
        engine_that_raises(exc).separate(Path(__file__), Path(DATA_DIR) / "out", "2stem", lambda f: None)
        check(False, f"{name}: debía fallar")
    except SeparationError as err:
        check(expected in str(err) and str(exc) not in str(err), f"{name}: «{err}»")
        check(err.__cause__ is exc, f"{name}: el error original queda encadenado para el log")

try:
    engine_without_stems().separate(Path(__file__), Path(DATA_DIR) / "out", "2stem", lambda f: None)
    check(False, "sin pistas: debía fallar")
except SeparationError as err:
    check("No se pudo procesar el audio" in str(err), f"sin pistas: «{err}»")

# De punta a punta por la API: lo que ve la página en `error`.
with TestClient(main.app) as client:
    main.manager.engine = engine_that_raises(me.FunctionTimeoutError("Function timed out after 1200s"))
    r = client.post("/api/jobs", files={"file": ("a.wav", b"RIFF" + b"\0" * 2000, "audio/wav")})
    check(r.status_code == 202, f"POST /api/jobs -> {r.status_code}")
    job_id = r.json()["id"]
    for _ in range(50):
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    check(j["status"] == "error", f"estado final: {j['status']}")
    check(j["error"] == "El audio tardó demasiado en procesarse. Prueba con un archivo más corto.",
          f"la página recibe: «{j['error']}»")
    check("1200s" not in j["error"], "sin detalles técnicos")

print("\nTodo bien.")
