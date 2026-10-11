"""Compara las GPUs de Modal (T4 vs L4) en tiempo y costo con canciones reales.

Requiere las dos apps desplegadas:
    SPLITTER_GPU=T4 .venv/bin/modal deploy modal_app.py
    SPLITTER_GPU=L4 .venv/bin/modal deploy modal_app.py

Uso:
    python tests/benchmark_gpus.py CARPETA_CON_CANCIONES [--max-cost 3] [--estimate-only]

Por cada GPU × preset hace 1 corrida "fría" (contenedor recién arrancado; se reporta aparte)
y N corridas "calientes" por canción (las que cuentan). Guarda cada corrida en
benchmarks_gpus.csv e imprime un resumen.

El costo es una ESTIMACIÓN: segundos de GPU × precio por segundo (app/config.py). Modal cobra
además CPU y memoria del contenedor, y el tiempo que el contenedor sigue encendido sin trabajo.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config  # noqa: E402

GPUS = ("T4", "L4")
PRESETS = ("2stem", "4stem")

# Segundos de GPU por segundo de audio, para la estimación ANTES de correr. Son conservadores:
# en L4 se midió ~0.85 (2stem, BS-Roformer en fp32; con autocast bajó a ~0.44) y ~0.17 (4stem, htdemucs_ft);
# se suben y para la T4 se supone 3 veces más lenta que la L4.
EST_GPU_SECONDS_PER_AUDIO_SECOND = {
    ("L4", "2stem"): 1.2,
    ("L4", "4stem"): 0.3,
    ("T4", "2stem"): 3.6,
    ("T4", "4stem"): 0.9,
}
EST_RUN_OVERHEAD_S = 20     # recibir el audio, codificar y mandar las pistas
EST_COLD_OVERHEAD_S = 150   # arrancar el contenedor y cargar el modelo
EST_IDLE_TAIL_S = 30        # hasta que se detiene el contenedor al terminar cada GPU

# Si la diferencia de costo por minuto es menor que esto, se consideran "parecidas" y gana la más rápida.
SIMILAR_COST = 0.10

AUDIO_EXT = config.ALLOWED_INPUT_EXT
FIELDS = [
    "gpu", "preset", "song", "run_type", "run", "audio_seconds", "wall_seconds", "gpu_seconds",
    "load_seconds", "separate_seconds", "bytes_sent", "bytes_received", "est_cost_usd", "error",
]


class _BudgetExceeded(Exception):
    pass


def app_name(gpu: str) -> str:
    return f"splitter-gpu-{gpu.lower()}"


def audio_seconds(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def stop_containers(gpu: str) -> None:
    """Apaga los contenedores de esa app: para que la corrida fría lo sea de verdad
    y para no pagar la espera (scaledown_window) al terminar."""
    modal_cli = [sys.executable, "-m", "modal"]
    listing = subprocess.run([*modal_cli, "container", "list", "--json"], capture_output=True, text=True, check=True)
    for c in json.loads(listing.stdout or "[]"):
        if c.get("app_name") == app_name(gpu):
            subprocess.run([*modal_cli, "container", "stop", "-y", c["container_id"]], capture_output=True, text=True)
            print(f"  (contenedor {c['container_id']} de {app_name(gpu)} detenido)")


def estimate(songs: dict[Path, float], gpus, presets, warm_runs: int) -> float:
    total = 0.0
    shortest = min(songs.values())
    for gpu in gpus:
        price = config.GPU_PRICE_PER_SECOND[gpu]
        secs = EST_IDLE_TAIL_S
        for preset in presets:
            rate = EST_GPU_SECONDS_PER_AUDIO_SECOND[(gpu, preset)]
            secs += EST_COLD_OVERHEAD_S + shortest * rate + EST_RUN_OVERHEAD_S
            secs += sum(warm_runs * (dur * rate + EST_RUN_OVERHEAD_S) for dur in songs.values())
        cost = secs * price
        print(f"  {gpu}: ~{secs / 60:.0f} min de GPU  → ~${cost:.2f}")
        total += cost
    return total


def run_once(engine, song: Path, preset: str) -> dict:
    out_dir = Path(tempfile.mkdtemp(prefix="splitter-bench-"))
    try:
        engine.separate(song, out_dir, preset, lambda _f: None)
        return dict(engine.last_stats)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def summarize(rows: list[dict], gpus, presets) -> None:
    warm = [r for r in rows if r["run_type"] == "warm" and not r["error"]]
    cold = [r for r in rows if r["run_type"] == "cold" and not r["error"]]

    if cold:
        print("\nCorridas frías (incluyen arranque del contenedor; no se usan para comparar):")
        for r in cold:
            print(f"  {r['gpu']} {r['preset']:5}  {r['song']}: {r['wall_seconds']:.1f} s vistos desde aquí, "
                  f"{r['gpu_seconds']:.1f} s en el worker (carga del modelo {r['load_seconds']:.1f} s)")

    print("\nResumen (solo corridas calientes):")
    header = f"{'GPU':4} {'preset':6} {'corridas':>8} {'s GPU prom.':>11} {'× tiempo real':>13} {'$ / canción':>11} {'$ / min audio':>13}"
    print(header)
    print("-" * len(header))
    results = {}
    for preset in presets:
        for gpu in gpus:
            rs = [r for r in warm if r["gpu"] == gpu and r["preset"] == preset]
            if not rs:
                print(f"{gpu:4} {preset:6} {'0':>8}  (sin corridas válidas)")
                continue
            price = config.GPU_PRICE_PER_SECOND[gpu]
            gpu_s = sum(r["gpu_seconds"] for r in rs)
            audio_s = sum(r["audio_seconds"] for r in rs)
            res = {
                "avg_gpu_s": statistics.mean(r["gpu_seconds"] for r in rs),
                "x_realtime": audio_s / gpu_s,
                "cost_song": statistics.mean(r["gpu_seconds"] * price for r in rs),
                "cost_min": gpu_s * price / (audio_s / 60),
            }
            results[(gpu, preset)] = res
            print(f"{gpu:4} {preset:6} {len(rs):>8} {res['avg_gpu_s']:>11.1f} {res['x_realtime']:>12.2f}× "
                  f"{res['cost_song']:>11.4f} {res['cost_min']:>13.5f}")

    print("\nConclusión:")
    for preset in presets:
        cands = {g: results[(g, preset)] for g in gpus if (g, preset) in results}
        if len(cands) < 2:
            print(f"  {preset}: no hay datos de todas las GPUs, no se puede comparar.")
            continue
        cheapest = min(cands, key=lambda g: cands[g]["cost_min"])
        fastest = max(cands, key=lambda g: cands[g]["x_realtime"])
        print(f"  {preset}: más barata por minuto de audio → {cheapest} "
              f"(${cands[cheapest]['cost_min']:.5f}/min); más rápida → {fastest} "
              f"({cands[fastest]['x_realtime']:.2f}× tiempo real).")
        if cheapest == fastest:
            print(f"    Recomendación: {cheapest}, gana en las dos cosas.")
            continue
        diff = cands[fastest]["cost_min"] / cands[cheapest]["cost_min"] - 1
        speedup = cands[fastest]["x_realtime"] / cands[cheapest]["x_realtime"]
        if diff <= SIMILAR_COST:
            print(f"    Recomendación: {fastest}. Cuesta solo {diff:.0%} más por minuto y es {speedup:.1f}× más rápida.")
        else:
            print(f"    Recomendación: {cheapest}. {fastest} es {speedup:.1f}× más rápida pero cuesta {diff:.0%} más "
                  f"por minuto; tu criterio es el costo por canción.")

    print("\nOjo: el costo es una ESTIMACIÓN (segundos de GPU × precio por segundo). Modal también cobra CPU y")
    print("memoria del contenedor, el arranque y el tiempo que sigue encendido sin trabajo antes de apagarse.")
    print("El gasto real está en modal.com → Settings → Usage (filtra por las apps splitter-gpu-t4 / splitter-gpu-l4).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="carpeta con las canciones de prueba")
    ap.add_argument("--gpus", default=",".join(GPUS))
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--warm-runs", type=int, default=2, help="corridas calientes por canción (default 2)")
    ap.add_argument("--max-cost", type=float, default=3.0,
                    help="USD. No arranca si la estimación lo supera, y aborta si el costo acumulado lo alcanza")
    ap.add_argument("--csv", type=Path, default=Path("benchmarks_gpus.csv"))
    ap.add_argument("--estimate-only", action="store_true", help="solo calcula la estimación, no corre nada")
    args = ap.parse_args()

    gpus = [g.strip().upper() for g in args.gpus.split(",") if g.strip()]
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    for g in gpus:
        if g not in GPUS:
            sys.exit(f"GPU desconocida: {g}")
    for p in presets:
        if p not in config.PRESETS:
            sys.exit(f"Preset desconocido: {p}")

    songs = {p: audio_seconds(p) for p in sorted(args.folder.iterdir()) if p.suffix.lower() in AUDIO_EXT}
    if not songs:
        sys.exit(f"No hay audio en {args.folder}")
    print("Canciones:")
    for p, d in songs.items():
        print(f"  {p.name}: {d:.0f} s")
    shortest = min(songs, key=songs.get)

    print(f"\nEstimación conservadora ({len(gpus)} GPU × {len(presets)} presets, 1 fría + "
          f"{args.warm_runs} calientes por canción):")
    est = estimate(songs, gpus, presets, args.warm_runs)
    print(f"  TOTAL estimado: ~${est:.2f}  (límite --max-cost ${args.max_cost:.2f})")
    if args.estimate_only:
        return
    if est > args.max_cost:
        sys.exit("La estimación supera --max-cost; no se corre nada. Súbelo si estás de acuerdo con el gasto.")

    from app.engine import ModalEngine

    rows: list[dict] = []
    aborted = ""
    spent = 0.0  # conservador: usa los segundos vistos desde aquí (≥ segundos de GPU, incluye arranque)

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        def record(gpu, preset, song, run_type, run, stats, error="", wall=0.0):
            nonlocal spent
            price = config.GPU_PRICE_PER_SECOND[gpu]
            wall = stats.get("wall_seconds", wall)
            spent += wall * price
            row = {
                "gpu": gpu, "preset": preset, "song": song.name, "run_type": run_type, "run": run,
                "audio_seconds": round(songs[song], 2),
                "wall_seconds": round(wall, 2),
                "gpu_seconds": round(stats.get("gpu_seconds", 0.0), 2),
                "load_seconds": round(stats.get("load_seconds", 0.0), 2),
                "separate_seconds": round(stats.get("separate_seconds", 0.0), 2),
                "bytes_sent": stats.get("bytes_sent", 0),
                "bytes_received": stats.get("bytes_received", 0),
                "est_cost_usd": round(stats.get("gpu_seconds", 0.0) * price, 5),
                "error": error,
            }
            writer.writerow(row)
            f.flush()
            rows.append(row)
            status = f"ERROR: {error}" if error else (
                f"{row['wall_seconds']:.1f} s aquí, {row['gpu_seconds']:.1f} s GPU, ~${row['est_cost_usd']:.4f}")
            print(f"  [{run_type} {run}] {gpu} {preset} {song.name}: {status}  (acumulado ~${spent:.2f})")

        try:
            for gpu in gpus:
                engine = ModalEngine(app_name(gpu))
                for preset in presets:
                    print(f"\n== {gpu} · {preset}")
                    stop_containers(gpu)
                    plan = [("cold", 1, shortest)] + [
                        ("warm", r, s) for s in songs for r in range(1, args.warm_runs + 1)
                    ]
                    for run_type, run, song in plan:
                        if spent >= args.max_cost:
                            raise _BudgetExceeded(f"Costo acumulado estimado ~${spent:.2f} alcanzó --max-cost; aborto.")
                        t0 = time.perf_counter()
                        try:
                            stats = run_once(engine, song, preset)
                            record(gpu, preset, song, run_type, run, stats)
                        except Exception as exc:  # noqa: BLE001 - se registra y se sigue con lo demás
                            traceback.print_exc()
                            record(gpu, preset, song, run_type, run, {}, error=f"{type(exc).__name__}: {exc}",
                                   wall=time.perf_counter() - t0)
                stop_containers(gpu)
        except _BudgetExceeded as exc:
            aborted = str(exc)
            for gpu in gpus:
                stop_containers(gpu)
        except BaseException:
            for gpu in gpus:  # se cortó a la mitad (Ctrl+C, error): no dejar contenedores encendidos
                try:
                    stop_containers(gpu)
                except Exception:  # noqa: BLE001
                    pass
            raise

    if aborted:
        print(f"\n*** {aborted} El resumen solo incluye lo que alcanzó a correr.")
    print(f"\nDetalle por corrida en {args.csv}")
    print(f"Costo acumulado estimado (conservador, con tiempo visto desde aquí): ~${spent:.2f}")
    summarize(rows, gpus, presets)


if __name__ == "__main__":
    main()
