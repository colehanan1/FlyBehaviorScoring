from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import pytest

from flypca.io import TrialTimeseries
from flypca.synthetic import generate_synthetic_trials


@pytest.fixture(scope="session")
def synthetic_dataset() -> Tuple[List[TrialTimeseries], pd.DataFrame]:
    return generate_synthetic_trials()
