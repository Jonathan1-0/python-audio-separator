"""Configuración del backend de Splitter. Todo se puede cambiar con variables de entorno."""
import os
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


BASE_DIR = Path(__file__).resolve().parent.parent

# Dónde se guardan subidas y resultados (se borran solos pasado el TTL).
DATA_DIR = Path(os.getenv("SPLITTER_DATA_DIR", BASE_DIR / "data")).resolve()
# Dónde se descargan los modelos de IA (pesan cientos de MB, se bajan una sola vez).
MODEL_DIR = Path(os.getenv("SPLITTER_MODEL_DIR", BASE_DIR / "models")).resolve()

# Límite de tamaño de archivo (idea 9 del alcance).
MAX_UPLOAD_MB = _int("SPLITTER_MAX_UPLOAD_MB", 100)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Cuánto tiempo se conservan los resultados antes de borrarse.
JOB_TTL_MINUTES = _int("SPLITTER_JOB_TTL_MINUTES", 60)
# Cuántos trabajos pueden esperar en cola.
MAX_QUEUE = _int("SPLITTER_MAX_QUEUE", 20)

# "real" usa python-audio-separator en esta máquina; "modal" separa en una GPU L4 de Modal;
# "fake" simula el proceso (para probar la página sin modelos/GPU).
ENGINE = os.getenv("SPLITTER_ENGINE", "real").lower()
# App de Modal a la que se mandan los trabajos: "splitter-gpu-l4" o "splitter-gpu-t4"
# (se despliegan con `SPLITTER_GPU=L4|T4 modal deploy modal_app.py`).
MODAL_APP = os.getenv("SPLITTER_MODAL_APP", "splitter-gpu-l4")

# Precio de Modal por segundo de GPU (USD), para estimar costos en tests/benchmark_gpus.py.
GPU_PRICE_PER_SECOND = {
    "T4": float(os.getenv("SPLITTER_PRICE_T4", "0.000164")),
    "L4": float(os.getenv("SPLITTER_PRICE_L4", "0.000222")),
}

# Orígenes que pueden llamar a la API desde el navegador (la página). En producción pon el dominio real.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("SPLITTER_ALLOWED_ORIGINS", "*").split(",") if o.strip()]

ALLOWED_INPUT_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".aiff", ".aif"}

# Formatos de salida (idea 8). Los stems se generan una vez en WAV y se convierten al descargar.
OUTPUT_FORMATS = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "m4a": "audio/mp4",
}
# Calidad de la vista previa (MP3 ligero para que cargue rápido).
PREVIEW_BITRATE = "128k"

# Modelos disponibles. La clave es lo que manda la página en el campo "preset".
PRESETS = {
    "2stem": {
        "label": "Voz e instrumental",
        "model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "stems": ["vocals", "instrumental"],
        # Precisión mixta (PyTorch autocast) en la GPU: más rápido, diferencia de calidad mínima.
        "autocast": os.getenv("SPLITTER_AUTOCAST_2STEM", "1") == "1",
    },
    "4stem": {
        "label": "Voz, batería, bajo y otros",
        "model": "htdemucs_ft.yaml",
        "stems": ["vocals", "drums", "bass", "other"],
    },
}
DEFAULT_PRESET = "2stem"

# Cuántas ventanas solapadas procesa el modelo Roformer (preset 2stem) por cada trozo de audio.
# El modelo actual ya trae 4 en su YAML (num_overlap); aquí queda fijo para que no dependa del modelo.
# Bajarlo a 2 tarda ~la mitad, con algo de pérdida de calidad en los bordes de cada trozo.
MDXC_OVERLAP = _int("SPLITTER_MDXC_OVERLAP", 4)
