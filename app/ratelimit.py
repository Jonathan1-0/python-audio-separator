"""Límite de trabajos por persona (por IP) en una ventana de tiempo, en memoria."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict


class RateLimiter:
    def __init__(self, max_events: int, window_seconds: float) -> None:
        self.max_events = max_events  # 0 = sin límite
        self.window = window_seconds
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def acquire(self, key: str) -> float:
        """Reserva un lugar para `key`. Devuelve 0 si se pudo, o los segundos que faltan
        para que se libere uno. Se reserva antes de procesar para que dos subidas
        simultáneas no pasen las dos el límite; si la subida falla, se llama a `release`."""
        if self.max_events <= 0:
            return 0.0
        now = time.monotonic()
        with self._lock:
            events = self._events[key]
            while events and now - events[0] >= self.window:
                events.popleft()
            if len(events) >= self.max_events:
                return self.window - (now - events[0])
            events.append(now)
            return 0.0

    def release(self, key: str) -> None:
        """Devuelve la última reserva (la subida fue rechazada y no debe contar)."""
        if self.max_events <= 0:
            return
        with self._lock:
            events = self._events.get(key)
            if events:
                events.pop()
            if not events:
                self._events.pop(key, None)
