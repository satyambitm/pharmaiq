"""In-process request limits for the paid LLM endpoint.

These bound spend and abuse on a single instance; they reset on restart and are not shared between instances,
which is acceptable for the one-instance deployment. They are not a substitute for authentication.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class AgentLimiter:
    def __init__(self, per_hour: int, per_day: int) -> None:
        self.per_hour, self.per_day = per_hour, per_day
        self._clients: dict[str, deque[float]] = {}
        self._day: deque[float] = deque()
        self._lock = threading.Lock()

    def check(self, client: str, now: float | None = None) -> str | None:
        """Record a request and return None, or return why it is refused (without recording it)."""
        now = time.time() if now is None else now
        with self._lock:
            while self._day and self._day[0] <= now - 86_400:
                self._day.popleft()
            hits = self._clients.setdefault(client, deque())
            while hits and hits[0] <= now - 3_600:
                hits.popleft()
            if len(self._day) >= self.per_day:
                return "The AI analyst has reached its daily limit for this deployment. Try again tomorrow."
            if len(hits) >= self.per_hour:
                return f"At most {self.per_hour} questions per hour. Try again later."
            hits.append(now)
            self._day.append(now)
            if len(self._clients) > 10_000:  # drop idle clients so memory stays bounded
                for k in [k for k, v in self._clients.items() if not v]:
                    del self._clients[k]
            return None
