"""Shape and dimension sanity checks for shared utilities."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from stats import utils


def make_groups() -> list[utils.FlyGroups]:
    timepoints = 5
    fly_ids = ["fly1", "fly2", "fly3"]
    groups = []
    for idx, fly in enumerate(fly_ids):
        baseline = np.linspace(0, 1, timepoints) + idx
        a_trials = baseline + 0.1
        b_trials = baseline
        groups.append(
            utils.FlyGroups(
                fly_id=fly,
                group_a=utils.FlyGroup(fly, np.vstack([a_trials, a_trials])),
                group_b=utils.FlyGroup(fly, np.vstack([b_trials, b_trials])),
            )
        )
    return groups


def test_compute_time_axis_length() -> None:
    axis = utils.compute_time_axis(100, 40.0)
    assert axis.shape == (100,)
    assert np.isclose(axis[-1], (99) / 40.0)


def test_diff_matrix_shape() -> None:
    groups = make_groups()
    diff = utils.diff_matrix(groups)
    assert diff.shape == (len(groups), groups[0].group_a.trials.shape[1])
    assert np.allclose(diff.mean(axis=1), 0.1)


def test_latency_to_threshold_window() -> None:
    traces = np.array([[0, 0.1, 0.2, 0.4, 0.5], [0, 0.05, 0.07, 0.09, 0.5]])
    latencies, events = utils.latency_to_threshold(traces, 0.3, (0, 5))
    assert latencies.shape == (2,)
    assert events.tolist() == [True, False]
