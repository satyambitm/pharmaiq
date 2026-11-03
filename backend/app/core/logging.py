"""Structured single-line logging (key=value) suitable for Render's log stream."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=\"%(message)s\"", datefmt="%Y-%m-%dT%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("uvicorn.access", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
