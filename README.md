# Splitter — Backend

API (FastAPI) que conecta la página web de Splitter con el separador de audio
([python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)).

Cubre el alcance acordado: subir archivo (drag & drop), elegir pistas a extraer,
vista previa antes de descargar, barra de progreso en tiempo real, varios formatos
de salida y límite de tamaño de archivo.

## Cómo funciona

```
Página  ──POST /api/jobs──▶  guarda el audio y lo mete a la cola
   ▲                              │ (con Modal, hasta 3 trabajos a la vez, cada uno en su GPU)
   │◀── SSE /events (progreso) ───┘
   │──GET preview / download──▶  convierte con ffmpeg y entrega
```

- Los stems se generan **una sola vez en WAV** y se convierten al formato pedido al descargar (se cachean).
- Los resultados se borran solos a los 60 min (`SPLITTER_JOB_TTL_MINUTES`). El audio original se borra al terminar.
- Con Modal corren hasta `SPLITTER_PARALLEL_JOBS` trabajos a la vez (3), cada uno en su propio contenedor; los demás esperan en cola y la API informa su posición. Con la GPU local, uno a la vez.

## Instalación (Linux)

Requisitos: Python 3.10+ y `ffmpeg` (`sudo apt install ffmpeg`).

```bash
cd splitter-backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "audio-separator[cpu]"     # para probar en tu laptop sin GPU
uvicorn app.main:app --reload --port 8000
```

Docs interactivas de la API: <http://localhost:8000/docs>

En **Windows** (Jonathan): igual, pero `.venv\Scripts\activate` y ffmpeg desde <https://ffmpeg.org> (o `winget install ffmpeg`).

### Probar la página sin modelos ni GPU

```bash
SPLITTER_ENGINE=fake uvicorn app.main:app --reload --port 8000
python tests/smoke_test.py         # prueba de punta a punta
python tests/concurrency_test.py   # descargas simultáneas de la misma pista
python tests/modal_errors_test.py  # los fallos de Modal llegan a la página con mensajes entendibles
python tests/limits_test.py        # límite de minutos por audio y de trabajos por hora
python tests/parallel_test.py      # varios trabajos a la vez
```

El modo `fake` genera tonos de prueba; sirve para desarrollar el frontend sin esperar a los modelos.

## Cuando tengan la tarjeta NVIDIA

1. Instala el driver de NVIDIA y CUDA (en Linux: `nvidia-smi` debe mostrar la tarjeta).
2. En el entorno virtual: `pip install "audio-separator[gpu]"`.
3. Arranca igual. `GET /api/health` debe mostrar `"cuda": true` y el nombre de la tarjeta.
   Si dice `false`, está corriendo en CPU (funciona, pero tarda mucho más).

La primera vez que se usa cada modelo se descarga solo a `./models` (cientos de MB) y queda guardado.
Conviene hacer una separación de prueba antes de la entrega para que ya estén descargados.

## Separar en la nube con Modal (GPU T4 o L4)

En vez de la tarjeta local, el backend puede mandar cada trabajo a una GPU de [Modal](https://modal.com).
La app de Modal está en `modal_app.py` y reutiliza el mismo código de `app/`. La GPU se elige al
desplegar con `SPLITTER_GPU`, y cada una es una app distinta (`splitter-gpu-t4`, `splitter-gpu-l4`).

```bash
pip install modal && modal setup                     # una vez: guarda el token en ~/.modal.toml
SPLITTER_GPU=L4 modal deploy modal_app.py            # y otra vez cada que cambies modal_app.py o app/
SPLITTER_GPU=T4 modal deploy modal_app.py
SPLITTER_ENGINE=modal SPLITTER_MODAL_APP=splitter-gpu-l4 uvicorn app.main:app --port 8000
```

### Probar con Modal desde otra computadora (p. ej. Jonathan en Windows)

Las apps de Modal ya están desplegadas en el workspace de Diego; para separar desde otra
computadora no hay que desplegar nada, solo conectarse a ese workspace.

1. Clonar el repo, crear el entorno e instalar: `pip install -r requirements.txt` (ya incluye `modal`).
   No hace falta `audio-separator` ni GPU: la separación corre en Modal.
2. Instalar `ffmpeg` (`winget install ffmpeg` en Windows).
3. Credenciales de Modal: Diego crea un token en modal.com → Settings → API Tokens y se lo pasa;
   en la otra computadora: `modal token set --token-id <id> --token-secret <secret>`.
   (Un token propio por persona se puede revocar sin afectar a los demás.)
4. Arrancar y abrir <http://localhost:8000>:

```powershell
# Windows (PowerShell)
$env:SPLITTER_ENGINE="modal"; $env:SPLITTER_FRONTEND_DIR="frontend"
uvicorn app.main:app --port 8000
```
```bash
# Linux / macOS
SPLITTER_ENGINE=modal SPLITTER_FRONTEND_DIR=frontend uvicorn app.main:app --port 8000
```

Cada separación gasta el crédito de Modal del workspace (~$0.01–0.03 por canción en L4).
Para trabajar solo en la página sin gastar, usar `SPLITTER_ENGINE=fake`.

### Comparar T4 contra L4

```bash
python tests/benchmark_gpus.py carpeta_con_canciones --estimate-only   # solo la estimación de costo
python tests/benchmark_gpus.py carpeta_con_canciones --max-cost 3       # corre todo
```

Por cada GPU × preset hace 1 corrida fría (se reporta aparte) y 2 calientes por canción, guarda todo en
`benchmarks_gpus.csv` e imprime un resumen con costo por canción y por minuto de audio.
No arranca si la estimación supera `--max-cost` y aborta si el costo acumulado lo alcanza.
El costo es una estimación (segundos de GPU × precio de `app/config.py`); el gasto real está en
modal.com → Settings → Usage.

- Los modelos se guardan en el Volume `splitter-models`, así que se descargan una sola vez.
- El contenedor sigue vivo 2 min después del último trabajo (`scaledown_window`); esa espera también se cobra. Si llega un trabajo con el contenedor apagado, tarda ~15-25 s extra en arrancar.
- Las pistas viajan en FLAC y el backend las guarda en WAV, como en local. Aquí sí hace falta `ffmpeg`.
- Si cambias la imagen y el contenedor viejo sigue vivo, detenlo con `modal container list` y `modal container stop -y <id>`.

Presets disponibles (se cambian en `app/config.py`):

| preset | modelo | pistas | tiempo en L4 (canción de 3:41) |
|---|---|---|---|
| `2stem` | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (autocast) | vocals, instrumental | ~97 s |
| `4stem` | `htdemucs_ft.yaml` (lenta, mejor calidad) | vocals, drums, bass, other | ~48 s |
| `4stem_fast` | `htdemucs.yaml` (rápida) | vocals, drums, bass, other | ~17 s |
| `denoise` | "Limpiar ruido": `model_bs_roformer…` y luego `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt` sobre la voz | voice (voz limpia; lo que se quita no se guarda) | ~70 s por cada 90 s de audio |

`denoise` encadena dos modelos (`steps` en `app/config.py`): el modelo de ruido está hecho para voces
solas, y sobre una mezcla completa también quitaba reverberación y parte de la música.

## Contrato de la API (para el frontend)

Todas las rutas empiezan con `/api`. Errores: JSON `{"detail": "mensaje en español"}`.

| Método | Ruta | Para qué |
|---|---|---|
| GET | `/api/config` | Límites, formatos, presets y pistas para armar la interfaz |
| POST | `/api/jobs` | Subir audio. Form-data: `file`, `preset` (opcional), `stems` (opcional, ej. `vocals,drums`) |
| GET | `/api/jobs/{id}` | Estado actual (alternativa a SSE) |
| GET | `/api/jobs/{id}/events` | Progreso en vivo (Server-Sent Events) |
| GET | `/api/jobs/{id}/stems/{stem}/preview` | MP3 ligero para el reproductor |
| GET | `/api/jobs/{id}/stems/{stem}/download?format=mp3` | Descargar una pista (`wav`, `mp3`, `flac`, `ogg`, `m4a`) |
| GET | `/api/jobs/{id}/download?format=mp3` | Todas las pistas elegidas en un ZIP |
| DELETE | `/api/jobs/{id}` | Borrar los archivos |

Códigos útiles: `413` archivo muy grande o audio más largo que el límite de minutos · `415` tipo no soportado o audio ilegible · `429` límite de trabajos por hora (con encabezado `Retry-After` en segundos) · `400` pista/formato inválido · `503` cola llena · `404` trabajo expirado.
`GET /api/config` también informa `max_audio_minutes` y `max_jobs_per_hour` (`null` = sin límite) para mostrarlos en la página.

Objeto de trabajo (lo que devuelven `POST`, `GET` y cada evento):

```json
{
  "id": "833fa66c...",
  "status": "queued | processing | done | error",
  "progress": 0.42,
  "message": "Separando audio…",
  "queue_position": null,
  "filename": "Mi canción",
  "error": null,
  "stems": [{"name": "vocals", "preview_url": "...", "download_url": "..."}],
  "download_all_url": "/api/jobs/833fa66c.../download",
  "expires_in_minutes": 58
}
```

`stems`, `download_all_url` y `expires_in_minutes` solo vienen cuando `status` es `done`.
`progress` es de 0 a 1 (puede ser `null` si el modelo no permite medirlo).

### Ejemplo de uso desde la página (JavaScript)

```js
const API = "http://localhost:8000";   // si la página la sirve el mismo backend, usa ""

// 1) subir
const form = new FormData();
form.append("file", fileInput.files[0]);
form.append("preset", "2stem");
form.append("stems", "vocals,instrumental");
const res = await fetch(`${API}/api/jobs`, { method: "POST", body: form });
if (!res.ok) throw new Error((await res.json()).detail);   // 413, 415, ...
const job = await res.json();

// 2) barra de progreso en vivo
const es = new EventSource(`${API}/api/jobs/${job.id}/events`);
es.addEventListener("update", (e) => {
  const j = JSON.parse(e.data);
  bar.style.width = `${Math.round((j.progress ?? 0) * 100)}%`;
  if (j.status === "done") {
    es.close();
    player.src = API + j.stems[0].preview_url;               // vista previa
    link.href  = `${API}${j.stems[0].download_url}?format=flac`;
  }
  if (j.status === "error") { es.close(); alert(j.error); }
});
```

## Variables de entorno

| Variable | Por defecto | Qué hace |
|---|---|---|
| `SPLITTER_MAX_UPLOAD_MB` | `100` | Límite de tamaño de archivo |
| `SPLITTER_JOB_TTL_MINUTES` | `60` | Cuánto se conservan los resultados |
| `SPLITTER_MAX_QUEUE` | `20` | Trabajos que pueden esperar |
| `SPLITTER_PARALLEL_JOBS` | `3` | Trabajos a la vez con Modal (con la GPU local siempre 1) |
| `SPLITTER_MAX_CONTAINERS` | `3` | Solo al desplegar `modal_app.py`: tope de contenedores con GPU a la vez. Debe coincidir con el anterior |
| `SPLITTER_MAX_AUDIO_MINUTES` | `10` | Duración máxima del audio (`0` = sin límite) |
| `SPLITTER_MAX_JOBS_PER_HOUR` | `10` | Trabajos por persona (IP) por hora (`0` = sin límite). Se cuentan en memoria |
| `SPLITTER_BEHIND_PROXY` | `0` | `1` solo si el backend está detrás de nginx/Caddy: usa la IP de `X-Forwarded-For` |
| `SPLITTER_ALLOWED_ORIGINS` | `*` | Dominios de la página (CORS). En producción pon el real |
| `SPLITTER_ENGINE` | `real` | `modal` para separar en Modal, `fake` para simular |
| `SPLITTER_MODAL_APP` | `splitter-gpu-l4` | App de Modal a usar (`splitter-gpu-l4` o `splitter-gpu-t4`) |
| `SPLITTER_AUTOCAST_2STEM` | `1` | `0` para separar el 2stem en fp32 (más lento). Se toma al desplegar `modal_app.py` |
| `SPLITTER_MDXC_OVERLAP` | `4` | Ventanas solapadas del modelo del 2stem. Se toma al desplegar `modal_app.py` |
| `SPLITTER_GPU` | `L4` | Solo al desplegar `modal_app.py`: `T4` o `L4` |
| `SPLITTER_PRICE_T4` / `SPLITTER_PRICE_L4` | `0.000164` / `0.000222` | USD por segundo de GPU, para el benchmark |
| `SPLITTER_DATA_DIR` / `SPLITTER_MODEL_DIR` | `./data` / `./models` | Carpetas de trabajo y modelos |
| `SPLITTER_FRONTEND_DIR` | — | Si se define, el backend también sirve la página desde esa carpeta |

## Pendiente para antes de publicar en internet

- Poner el backend detrás de HTTPS (nginx/Caddy) y restringir `SPLITTER_ALLOWED_ORIGINS`.
- Límites por duración y por IP: hechos. Detrás de un proxy, poner `SPLITTER_BEHIND_PROXY=1` o todos los usuarios compartirían el mismo límite. Si quieren, cuentas de usuario.
- Los trabajos viven en memoria: si se reinicia el servidor, se pierden los que estaban en curso.
