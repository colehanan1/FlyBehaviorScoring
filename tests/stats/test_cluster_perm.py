"""Synthetic tests for the cluster permutation module."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from stats import cluster_perm, utils


def make_groups(num_flies: int = 4, timepoints: int = 30) -> list[utils.FlyGroups]:
    groups: list[utils.FlyGroups] = []
    time = np.linspace(0, 1, timepoints)
    for idx in range(num_flies):
        base = np.sin(time * np.pi) * 0.1
        effect = np.zeros_like(time)
        effect[:5] = 0.5
        a_trace = base + effect
        b_trace = base
        groups.append(
            utils.FlyGroups(
                fly_id=f"fly{idx}",
                group_a=utils.FlyGroup(f"fly{idx}", np.vstack([a_trace, a_trace])),
                group_b=utils.FlyGroup(f"fly{idx}", np.vstack([b_trace, b_trace])),
            )
        )
    return groups


def test_cluster_permutation_detects_signal(tmp_path) -> None:
    groups = make_groups()
    time_s = utils.compute_time_axis(groups[0].group_a.trials.shape[1], 40.0)
    rng = np.random.default_rng(0)
    result = cluster_perm.cluster_permutation_test(groups, time_s, n_perm=50, alpha=0.05, method="t", rng=rng)
    assert result.stats.shape[0] == time_s.size
    assert len(result.clusters) >= 1
    clusters_csv, time_csv, plot_path = cluster_perm.save_outputs(result, time_s, tmp_path, 0.05, "t")
    assert tmp_path.joinpath("cluster_perm_clusters.csv").exists()
    assert tmp_path.joinpath("cluster_perm_timewise.csv").exists()
    assert tmp_path.joinpath("cluster_perm_plot.png").exists()
