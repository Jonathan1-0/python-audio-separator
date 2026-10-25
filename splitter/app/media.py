"""Conversión de formatos con ffmpeg y empaquetado en ZIP."""
from __future__ import annotations

import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path
from typing import Dict

from . import config

_CODEC_ARGS = {
    "wav": ["-c:a", "pcm_s16le"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
    "flac": ["-c:a", "flac"],
    "ogg": ["-c:a", "libvorbis", "-q:a", "7"],
    "m4a": ["-c:a", "aac", "-b:a", "256k"],
}


def _tmp_path(dst: Path) -> Path:
    """Un temporal por petición: si dos piden lo mismo a la vez, no se pisan."""
    return dst.with_name(f"{dst.name}.{uuid.uuid4().hex}.part")


def _publish(tmp: Path, dst: Path) -> None:
    """En Windows falla si otra petición está enviando `dst`; entonces ya hay una copia válida."""
    try:
        tmp.replace(dst)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        if not dst.exists():
            raise


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def audio_duration(path: Path) -> float:
    """Segundos. Lanza ValueError si no es audio legible."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    try:
        seconds = float(proc.stdout.strip())
    except ValueError:
        raise ValueError(proc.stderr.strip()[:200] or "sin pista de audio") from None
    if proc.returncode != 0 or seconds <= 0:
        raise ValueError(proc.stderr.strip()[:200] or "duración inválida")
    return seconds


def convert(src: Path, fmt: str, *, preview: bool = False) -> Path:
    """Convierte el WAV y deja el resultado al lado como caché."""
    fmt = fmt.lower()
    if fmt not in config.OUTPUT_FORMATS:
        raise ValueError(f"Formato no soportado: {fmt}")

    if preview:
        dst = src.with_name(f"{src.stem}.preview.mp3")
        args = ["-c:a", "libmp3lame", "-b:a", config.PREVIEW_BITRATE]
    else:
        if fmt == "wav":
            return src
        dst = src.with_suffix(f".{fmt}")
        args = _CODEC_ARGS[fmt]

    if dst.exists() and dst.stat().st_size > 0:
        return dst
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg no está instalado en el servidor.")

    tmp = _tmp_path(dst)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", *args, "-f", dst.suffix.lstrip("."), str(tmp)]
    if dst.suffix == ".m4a":
        cmd[cmd.index("-f") + 1] = "ipod"
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg falló: {proc.stderr.strip()[:200]}")
    _publish(tmp, dst)
    return dst


def build_zip(stems: Dict[str, Path], fmt: str, base_name: str, dest_dir: Path) -> Path:
    zip_path = dest_dir / f"all.{fmt}.zip"
    if zip_path.exists():
        return zip_path
    tmp = _tmp_path(zip_path)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:  # el audio ya viene comprimido
            for stem, wav in stems.items():
                out = convert(wav, fmt)
                zf.write(out, arcname=f"{base_name} - {stem}.{fmt}")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _publish(tmp, zip_path)
    return zip_path
