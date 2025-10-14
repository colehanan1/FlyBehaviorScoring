from __future__ import annotations

from typing import List, Tuple

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from flypca.io import TrialTimeseries
from flypca.synthetic import generate_synthetic_trials


@pytest.fixture(scope="session")
def synthetic_dataset() -> Tuple[List[TrialTimeseries], pd.DataFrame]:
    return generate_synthetic_trials()
