"""Tests for BH-FDR helper."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from stats import utils


def test_bh_fdr_monotonic() -> None:
    pvals = np.array([0.001, 0.01, 0.2, np.nan, 0.05])
    qvals = utils.bh_fdr(pvals)
    finite = np.isfinite(qvals)
    assert np.all((qvals[finite] >= 0) & (qvals[finite] <= 1))
    sorted_indices = np.argsort(pvals[np.isfinite(pvals)])
    sorted_q = qvals[np.isfinite(pvals)][sorted_indices]
    assert np.all(np.diff(sorted_q) >= -1e-8)
