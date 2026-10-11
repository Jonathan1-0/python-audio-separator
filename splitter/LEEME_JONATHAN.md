# Jonathan: qué hacer antes de usar Splitter

Splitter es la página + el backend que separan audio (voz/música, instrumentos y limpiar ruido).
La separación no corre en tu computadora: corre en una GPU NVIDIA L4 en la nube (Modal), en la
cuenta de Diego. Tú solo necesitas correr el backend y abrir la página.

Todo esto está en la carpeta `splitter/` del repositorio. Lo que cambió y cuándo está en
`BITACORA.md` (en la raíz del repo) y en el historial de commits.

## 1. Lo que necesitas instalar (una sola vez)

1. **Python 3.10 o más nuevo** — <https://www.python.org/downloads/>. Al instalar, marca
   "Add python.exe to PATH".
2. **ffmpeg** — en PowerShell: `winget install ffmpeg`. Cierra y abre PowerShell, y comprueba
   con `ffmpeg -version` y `ffprobe -version`.
3. **Git** — si aún no lo tienes: `winget install Git.Git`.

No necesitas tarjeta de video ni instalar `audio-separator`: eso vive en Modal.

## 2. Bajar el proyecto e instalar dependencias

En PowerShell:

```powershell
git clone https://github.com/Jonathan1-0/python-audio-separator.git
cd python-audio-separator\splitter
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Si ya tenías el repo clonado: `git checkout main` y `git pull`, y luego los pasos desde `cd splitter`.

Si PowerShell no deja activar el entorno (error de "ejecución de scripts"), corre una vez:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a intentar.

## 3. Conectarte a Modal (una sola vez)

Diego te pasa un **token de Modal** (lo crea en modal.com → Settings → API Tokens; son dos
valores: un id que empieza con `ak-` y un secreto que empieza con `as-`). Con el entorno activado:

```powershell
modal token set --token-id ak-XXXX --token-secret as-XXXX
```

El token es como una contraseña: no lo pegues en el código, en el repo ni en chats públicos.

## 4. Arrancar la aplicación

Siempre desde `python-audio-separator\splitter` con el entorno activado (`.venv\Scripts\activate`):

```powershell
$env:SPLITTER_ENGINE="modal"; $env:SPLITTER_FRONTEND_DIR="frontend"
uvicorn app.main:app --port 8000
```

Abre <http://localhost:8000> en el navegador. Para apagarla: `Ctrl+C` en PowerShell.

**Para trabajar en la página sin gastar crédito**, usa el modo simulado (devuelve tonos de prueba
en vez de separar de verdad):

```powershell
$env:SPLITTER_ENGINE="fake"; $env:SPLITTER_FRONTEND_DIR="frontend"
uvicorn app.main:app --port 8000
```

## 5. Lo que tienes que saber antes de usarla

- **Cada separación real gasta crédito de Modal** de la cuenta de Diego: ~$0.01–0.03 por canción.
  El crédito es de $30 y el límite de gasto está en $0: si se acaba, la separación deja de
  funcionar hasta que se recargue. Para pruebas de la página usa el modo `fake`.
- **Límites**: audios de hasta 10 minutos y 10 trabajos por hora por persona. Si al probar te topas
  con el de 10 por hora, arranca con `$env:SPLITTER_MAX_JOBS_PER_HOUR="0"` (sin límite, solo para
  desarrollar).
- **Tiempos aproximados** (canción de ~4 min): instrumentos rápida ~35 s, instrumentos mejor
  calidad ~1 min, voz y música ~2 min, limpiar ruido ~3 min. Si nadie ha usado la app en los
  últimos 2 minutos, súmale ~20 s mientras se enciende la GPU.
- **No redespliegues en Modal** (`modal deploy modal_app.py`) sin hablarlo con Diego: las apps
  `splitter-gpu-l4` y `splitter-gpu-t4` las usan los dos, y un despliegue cambia lo que corre para
  ambos. Para cambios en la página (`frontend/index.html`) no hace falta desplegar nada.

## 6. Comprobar que todo funciona

Con el entorno activado, desde `splitter`:

```powershell
python tests\smoke_test.py
python tests\concurrency_test.py
python tests\modal_errors_test.py
python tests\limits_test.py
python tests\parallel_test.py
```

Todas deben terminar con "Todo bien." y no gastan crédito (usan el modo simulado).

## 7. Dónde está cada cosa

| Archivo | Qué es |
|---|---|
| `frontend/index.html` | La página (todo en un archivo: HTML, estilos y JavaScript) |
| `app/main.py` | La API que usa la página (`/api/jobs`, progreso, descargas) |
| `app/config.py` | Opciones de separación, modelos y límites |
| `app/engine.py` | Cómo se manda el audio a Modal y se reciben las pistas |
| `modal_app.py` | Lo que corre en la GPU de Modal |
| `README.md` | Documentación técnica completa (API, variables, benchmark) |
| `tests/` | Pruebas |

Si algo falla, el error técnico completo aparece en la ventana de PowerShell donde corre
`uvicorn`; la página solo muestra un mensaje simple.
