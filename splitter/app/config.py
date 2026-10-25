"""Configuración de Splitter. Cada valor se puede cambiar con su variable de entorno."""
import os
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


BASE_DIR = Path(__file__).resolve().parent.parent

# Subidas y resultados.
DATA_DIR = Path(os.getenv("SPLITTER_DATA_DIR", BASE_DIR / "data")).resolve()
# Modelos de IA; solo se usa dentro del contenedor de Modal.
MODEL_DIR = Path(os.getenv("SPLITTER_MODEL_DIR", BASE_DIR / "models")).resolve()

MAX_UPLOAD_MB = _int("SPLITTER_MAX_UPLOAD_MB", 100)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# En minutos. Hace falta además del tamaño: un MP3 de 100 MB puede durar más de una hora. 0 = sin límite.
MAX_AUDIO_MINUTES = _int("SPLITTER_MAX_AUDIO_MINUTES", 10)
# Por persona (IP). 0 = sin límite.
MAX_JOBS_PER_HOUR = _int("SPLITTER_MAX_JOBS_PER_HOUR", 10)
# Entre todas las personas, en 24 h: es el tope de gasto (~$0.03 por canción). 0 = sin límite.
MAX_JOBS_PER_DAY = _int("SPLITTER_MAX_JOBS_PER_DAY", 100)
# 1 = tomar la IP del usuario de X-Forwarded-For. Solo detrás de un proxy (nginx, Caddy, Modal):
# sin él, cualquiera podría inventar ese encabezado y saltarse el límite.
BEHIND_PROXY = os.getenv("SPLITTER_BEHIND_PROXY", "0") == "1"

JOB_TTL_MINUTES = _int("SPLITTER_JOB_TTL_MINUTES", 60)
# Trabajos a la vez, uno por GPU. Debe coincidir con SPLITTER_MAX_CONTAINERS de modal_app.py.
PARALLEL_JOBS = _int("SPLITTER_PARALLEL_JOBS", 3)

MAX_QUEUE = _int("SPLITTER_MAX_QUEUE", 20)

# "modal" separa en Modal; "fake" devuelve tonos de prueba sin gastar crédito.
ENGINE = os.getenv("SPLITTER_ENGINE", "modal").lower()
# "splitter-gpu-l4" o "splitter-gpu-t4" (ver modal_app.py).
MODAL_APP = os.getenv("SPLITTER_MODAL_APP", "splitter-gpu-l4")

# CORS. En producción, el dominio de la página.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("SPLITTER_ALLOWED_ORIGINS", "*").split(",") if o.strip()]

ALLOWED_INPUT_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".aiff", ".aif"}

# Las pistas se guardan en WAV y se convierten al descargar.
OUTPUT_FORMATS = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "m4a": "audio/mp4",
}
PREVIEW_BITRATE = "128k"

# Precisión mixta en el modelo de voz: ~2× más rápido, casi sin diferencia de calidad.
AUTOCAST_VOCALS = os.getenv("SPLITTER_AUTOCAST_2STEM", "1") == "1"

# La clave es el "preset" que manda la página.
PRESETS = {
    "2stem": {
        "label": "Voz e instrumental",
        "model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "stems": ["vocals", "instrumental"],
        "autocast": AUTOCAST_VOCALS,
    },
    "4stem": {
        "label": "Voz, batería, bajo y otros (lenta, mejor calidad)",
        "model": "htdemucs_ft.yaml",  # 4 modelos combinados
        "stems": ["vocals", "drums", "bass", "other"],
    },
    "4stem_fast": {
        "label": "Voz, batería, bajo y otros (rápida)",
        "model": "htdemucs.yaml",  # 1 modelo en vez de 4: más rápido, algo menos de calidad
        "stems": ["vocals", "drums", "bass", "other"],
    },
    "denoise": {
        "label": "Limpiar ruido: deja solo la voz (quita música y ruido de fondo)",
        "stems": ["voice"],
        # Primero se separa la voz y luego se le quita el ruido: el modelo de ruido es para voces
        # solas, y sobre la mezcla completa se comía la música.
        "steps": [
            {
                "model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                "autocast": AUTOCAST_VOCALS,  # igual que en 2stem: comparten el modelo ya cargado
                "stems": {"vocals": "voz_tmp"},
            },
            {
                "input": "voz_tmp",
                "model": "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
                "stems": {"dry": "voice"},
            },
        ],
    },
}
DEFAULT_PRESET = "2stem"

# Ventanas solapadas del modelo de voz. Con 2 tarda ~la mitad, con algo menos de calidad.
MDXC_OVERLAP = _int("SPLITTER_MDXC_OVERLAP", 4)
