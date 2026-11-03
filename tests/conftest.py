import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generate import generate_clean, make_dirty  # noqa: E402
from src.data.validate import to_panel  # noqa: E402
from src.features.build import build_features  # noqa: E402


@pytest.fixture(scope="session")
def clean_raw() -> pd.DataFrame:
    """Clean CSV-shaped frame generated in memory (no files needed)."""
    return generate_clean(42)


@pytest.fixture(scope="session")
def dirty_raw(clean_raw):
    return make_dirty(clean_raw, 42)


@pytest.fixture(scope="session")
def panel(clean_raw) -> pd.DataFrame:
    return to_panel(clean_raw)


@pytest.fixture(scope="session")
def features(panel) -> pd.DataFrame:
    return build_features(panel)
