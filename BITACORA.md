# Bitácora del proyecto

- 20/09/2026 - Diego: configuré Git, SSH y el entorno en Linux.
- 10/10/2026 - Diego: activé la GPU local (RTX 2050) tras actualizar el driver; luego pasamos la separación a la nube con Modal (GPU L4), con barra de progreso en vivo.
- 10/10/2026 - Diego: comparé GPUs T4 y L4 con una canción real (benchmark en splitter/benchmarks_gpus.csv); nos quedamos con L4 (~$0.01–0.03 por canción).
- 10/10/2026 - Diego: activé precisión mixta (autocast) en "voz y música": separa el doble de rápido y la voz suena igual; la GPU se apaga tras 2 min sin trabajo para ahorrar crédito.
- 10/10/2026 - Diego: agregué "instrumentos rápida" (htdemucs, ~3× más rápido) y "limpiar ruido" (separa la voz y le quita el ruido; entrega solo la voz limpia).
- 10/10/2026 - Diego: el backend ahora da mensajes claros cuando falla Modal, limita audios a 10 min y 10 trabajos por hora por persona, y procesa hasta 3 trabajos a la vez.
- 10/10/2026 - Diego: rehice la página para usuarios: elegir qué hacer (voz y música / instrumentos rápida o mejor calidad / limpiar ruido), subir y descargar, todo en español.
- 10/10/2026 - Diego: subí Splitter al repo en la carpeta splitter/; guía para empezar en splitter/LEEME_JONATHAN.md.
