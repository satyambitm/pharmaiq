import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["DATA_BACKEND"] = "csv"
os.environ["ENVIRONMENT"] = "test"

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.main import create_app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
