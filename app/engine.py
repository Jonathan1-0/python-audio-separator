"""Motor de separación: envuelve python-audio-separator y reporta progreso real.

Tres motores:
  - RealEngine: usa la librería (GPU NVIDIA si está disponible, si no CPU).
  - ModalEngine: manda el audio a una GPU L4 en Modal (ver modal_app.py) y recibe las pistas.
  - FakeEngine: simula el proceso con audio sintético, para probar la página sin modelos.
"""
from __future__ import annotations

import logging
import math
import os
import struct
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Dict, Optional

from . import config

log = logging.getLogger("splitter.engine")

ProgressCb = Callable[[float], None]  # recibe un valor de 0.0 a 1.0

# ---------------------------------------------------------------------------
# Progreso real: la librería usa tqdm en sus bucles de inferencia, así que lo
# interceptamos para saber en qué fracción va el trabajo.
# ---------------------------------------------------------------------------
_local = threading.local()


def _report(fraction: float) -> None:
    """`fraction` es el avance de la barra actual. Algunos modelos (demucs con varios
    submodelos/shifts) recorren varias barras seguidas; se combinan en un solo 0..1."""
    cb: Optional[ProgressCb] = getattr(_local, "cb", None)
    if cb is not None:
        if getattr(_local, "bars", None) is None:
            # Se calcula aquí y no antes: demucs carga su modelo dentro de separate().
            fn = getattr(_local, "bars_fn", None)
            _local.bars = fn() if fn else 1
        bars = _local.bars
        done = min(getattr(_local, "bars_done", 0), bars - 1)
        cb(max(0.0, min(1.0, (done + fraction) / bars)))


def _bar_finished() -> None:
    _local.bars_done = getattr(_local, "bars_done", 0) + 1


def _make_hooked_tqdm():
    from tqdm import tqdm

    class HookedTqdm(tqdm):
        def __iter__(self):
            total = self.total
            for i, item in enumerate(super().__iter__(), start=1):
                yield item
                if total:
                    _report(i / total)
            _bar_finished()

        def update(self, n=1):
            result = super().update(n)
            if self.total:
                _report(self.n / self.total)
            return result

    return HookedTqdm


_patched = False


def _patch_tqdm() -> None:
    """Sustituye tqdm por la versión con hook en los módulos que hacen la inferencia."""
    global _patched
    if _patched:
        return
    hooked = _make_hooked_tqdm()
    for modname in (
        "audio_separator.separator.architectures.mdxc_separator",
        "audio_separator.separator.architectures.mdx_separator",
        "audio_separator.separator.architectures.vr_separator",
        "audio_separator.separator.architectures.demucs_separator",
        "audio_separator.separator.uvr_lib_v5.demucs.apply",
        "demucs.apply",
    ):
        try:
            mod = __import__(modname, fromlist=["tqdm"])
        except Exception:  # módulo no disponible en esta versión
            continue
        current = getattr(mod, "tqdm", None)
        if current is None:
            continue
        # `from tqdm import tqdm` -> clase; `import tqdm` -> módulo
        if hasattr(current, "tqdm"):
            current.tqdm = hooked  # type: ignore[attr-defined]
        else:
            setattr(mod, "tqdm", hooked)
    _patched = True


def _stem_from_filename(path: str, known: list[str]) -> Optional[str]:
    """Deduce a qué stem corresponde un archivo de salida (p. ej. 'x_(Vocals)_modelo.wav')."""
    name = Path(path).stem.lower()
    for stem in known:
        if f"({stem})" in name or name == stem or name.endswith(f"_{stem}"):
            return stem
    return None


# ---------------------------------------------------------------------------
class RealEngine:
    """Mantiene un único Separator cargado en memoria (cargar el modelo es lo lento)."""

    def __init__(self) -> None:
        self._sep = None
        self._loaded_model: Optional[str] = None
        # Tiempos del último trabajo (los usa el worker de Modal para calcular costos).
        # Ojo: demucs carga sus pesos dentro de separate(), así que en 4stem eso cae en separate_seconds.
        self.last_timings: Dict[str, float] = {}

    def device_info(self) -> dict:
        info = {"engine": "real", "cuda": False, "device": "cpu"}
        try:
            import torch

            if torch.cuda.is_available():
                info.update(cuda=True, device=torch.cuda.get_device_name(0))
        except Exception:
            pass
        return info

    def _ensure_model(self, model: str, out_dir: Path, autocast: bool = False) -> None:
        if self._sep is None:
            from audio_separator.separator import Separator

            config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self._sep = Separator(
                model_file_dir=str(config.MODEL_DIR),
                output_dir=str(out_dir),
                output_format="WAV",  # siempre WAV sin pérdida; se convierte al descargar
                log_level=logging.INFO,
                mdxc_params={
                    "segment_size": 256,
                    "override_model_segment_size": False,
                    "batch_size": None,
                    "overlap": config.MDXC_OVERLAP,
                    "pitch_shift": 0,
                },
            )
        if self._loaded_model != model:
            log.info("Cargando modelo %s (autocast=%s)", model, autocast)
            # La precisión se decide al cargar el modelo, así que puede ser distinta por preset.
            self._sep.use_autocast = autocast
            self._sep.load_model(model_filename=model)
            self._loaded_model = model

    def _expected_bars(self) -> int:
        """Demucs hace una pasada (una barra de tqdm) por cada submodelo del bag y cada shift."""
        inst = getattr(self._sep, "model_instance", None)
        demucs = getattr(inst, "demucs_model_instance", None)
        if demucs is None:
            return 1
        return max(1, len(getattr(demucs, "models", ()))) * max(1, getattr(inst, "shifts", 1))

    def separate(self, input_path: Path, out_dir: Path, preset: str, on_progress: ProgressCb) -> Dict[str, Path]:
        spec = config.PRESETS[preset]
        out_dir.mkdir(parents=True, exist_ok=True)
        _patch_tqdm()
        t0 = time.perf_counter()
        self._ensure_model(spec["model"], out_dir, spec.get("autocast", False))
        t1 = time.perf_counter()

        # Redirigir la salida de este trabajo a su carpeta (mismo patrón que usa la librería).
        self._sep.output_dir = str(out_dir)
        if getattr(self._sep, "model_instance", None) is not None:
            self._sep.model_instance.output_dir = str(out_dir)

        custom_names = {s.capitalize(): s for s in spec["stems"]}
        _local.cb, _local.bars_fn, _local.bars, _local.bars_done = on_progress, self._expected_bars, None, 0
        try:
            produced = self._sep.separate(str(input_path), custom_output_names=custom_names)
        finally:
            _local.cb = None
        self.last_timings = {"load_seconds": t1 - t0, "separate_seconds": time.perf_counter() - t1}

        result: Dict[str, Path] = {}
        for p in produced:
            path = Path(p)
            if not path.is_absolute():
                path = out_dir / path.name if not path.exists() else path
            stem = _stem_from_filename(str(path), spec["stems"])
            if stem and path.exists():
                result[stem] = path
        if not result:
            raise RuntimeError("El modelo no produjo ningún archivo de salida reconocible.")
        return result


# ---------------------------------------------------------------------------
class ModalEngine:
    """Separa en la nube con Modal. Requiere `modal deploy modal_app.py` y el token de `modal setup`."""

    SEPARATION_SHARE = 0.7  # el resto de la barra es la descarga de las pistas

    def __init__(self, app_name: Optional[str] = None) -> None:
        import modal

        self.app_name = app_name or config.MODAL_APP
        self._cls = modal.Cls.from_name(self.app_name, "Separator")
        # Estadísticas del último trabajo: tiempos que reporta el worker + lo medido desde aquí.
        self.last_stats: Dict[str, float] = {}

    def device_info(self) -> dict:
        gpu = self.app_name.rsplit("-", 1)[-1].upper() if self.app_name.startswith("splitter-gpu-") else "GPU"
        return {"engine": "modal", "cuda": True, "device": f"NVIDIA {gpu} (Modal: {self.app_name})"}

    def separate(self, input_path: Path, out_dir: Path, preset: str, on_progress: ProgressCb) -> Dict[str, Path]:
        import subprocess

        out_dir.mkdir(parents=True, exist_ok=True)
        result: Dict[str, Path] = {}
        stats: Dict[str, float] = {}
        # La bajada de las pistas tarda bastante (decenas de MB), así que se lleva su parte de la barra.
        n_stems = len(config.PRESETS[preset]["stems"])
        audio = input_path.read_bytes()
        bytes_received = 0
        started = time.perf_counter()
        remote = self._cls().separate.remote_gen(audio, input_path.name, preset)
        for event in remote:
            kind = event["type"]
            if kind == "progress":
                on_progress(event["value"] * self.SEPARATION_SHARE)
                continue
            if kind == "done":
                stats = {k: v for k, v in event.items() if k != "type"}
                continue
            stem, data = event["name"], event["data"]
            bytes_received += len(data)
            flac = out_dir / f"{stem}.flac"
            flac.write_bytes(data)
            # El resto del backend trabaja con WAV (se convierte al descargar), así que se decodifica aquí.
            wav = out_dir / f"{stem}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(flac), "-c:a", "pcm_s16le", str(wav)],
                check=True,
            )
            flac.unlink()
            result[stem] = wav
            on_progress(self.SEPARATION_SHARE + (1 - self.SEPARATION_SHARE) * len(result) / n_stems)
        if not result:
            raise RuntimeError("Modal no devolvió ninguna pista.")
        stats.update(
            wall_seconds=time.perf_counter() - started,
            bytes_sent=len(audio),
            bytes_received=bytes_received,
        )
        self.last_stats = stats
        log.info("Modal %s (%s): %s", self.app_name, preset, stats)
        return result


# ---------------------------------------------------------------------------
class FakeEngine:
    """Simula la separación: genera tonos distintos por stem y avanza el progreso poco a poco."""

    def device_info(self) -> dict:
        return {"engine": "fake", "cuda": False, "device": "simulado"}

    def separate(self, input_path: Path, out_dir: Path, preset: str, on_progress: ProgressCb) -> Dict[str, Path]:
        spec = config.PRESETS[preset]
        out_dir.mkdir(parents=True, exist_ok=True)
        steps = 20
        for i in range(1, steps + 1):
            time.sleep(0.1)
            on_progress(i / steps)
        result: Dict[str, Path] = {}
        for idx, stem in enumerate(spec["stems"]):
            path = out_dir / f"{stem}.wav"
            freq = 220 * (idx + 1)
            rate, seconds = 22050, 2
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                frames = b"".join(
                    struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * n / rate)))
                    for n in range(rate * seconds)
                )
                w.writeframes(frames)
            result[stem] = path
        return result


def build_engine():
    if config.ENGINE == "fake":
        return FakeEngine()
    if config.ENGINE == "modal":
        return ModalEngine()
    return RealEngine()
