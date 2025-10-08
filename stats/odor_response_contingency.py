"""Utilities for summarising trained vs. untrained odor responses.

This module provides helpers to read a behavioural CSV file that contains the
columns ``dataset``, ``fly``, ``trial_num``, ``odor_sent``, ``during_hit`` and
``after_hit``.  It focuses on the ``during_hit`` values only and aggregates the
response pattern for each fly into the four canonical cells of a 2×2
contingency table:

* ``a`` – responds to both trained and untrained odours.
* ``b`` – responds to the trained odour but not the untrained odour.
* ``c`` – responds to the untrained odour but not the trained odour.
* ``d`` – does not respond to either odour.

The module also exposes a small command line interface that prints those
counts and can optionally export a figure mirroring the 2×2 table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures


@dataclass(frozen=True)
class OdorResponseSummary:
    """Container for the four contingency-table cells."""

    both_positive: int  # a
    trained_only: int  # b
    untrained_only: int  # c
    both_negative: int  # d

    @property
    def row_totals(self) -> List[int]:
        """Return row totals ``[a + b, c + d]``."""

        return [self.both_positive + self.trained_only, self.untrained_only + self.both_negative]

    @property
    def column_totals(self) -> List[int]:
        """Return column totals ``[a + c, b + d]``."""

        return [self.both_positive + self.untrained_only, self.trained_only + self.both_negative]

    @property
    def grand_total(self) -> int:
        """Return ``a + b + c + d``."""

        return sum(self.as_tuple())

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return a tuple ``(a, b, c, d)`` for convenience."""

        return (self.both_positive, self.trained_only, self.untrained_only, self.both_negative)


# ---------------------------------------------------------------------------
# Core logic


_TRUE_VALUES = {"true", "1", "t", "yes", "y"}
_FALSE_VALUES = {"false", "0", "f", "no", "n"}


def _coerce_to_bool(series: pd.Series) -> pd.Series:
    """Convert a Series with heterogeneous truthy values into booleans.

    The CSVs in the project often encode responses as ``0``/``1`` integers, but
    other encodings (``True``/``False`` or ``yes``/``no``) are also accounted
    for here.  Missing values are treated as ``False``.
    """

    def _convert(value) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in _TRUE_VALUES:
            return True
        if text in _FALSE_VALUES:
            return False
        try:
            return float(text) != 0.0
        except ValueError as exc:  # pragma: no cover - defensive path
            raise ValueError(f"Cannot coerce value '{value}' to boolean") from exc

    return series.apply(_convert).astype(bool)


def _normalise_trials(trials: Iterable[int]) -> List[int]:
    """Return a sorted list of unique integer trial identifiers."""

    return sorted({int(t) for t in trials})


def summarize_odor_responses(
    data: pd.DataFrame,
    *,
    dataset: str,
    trained_trials: Sequence[int],
    untrained_trials: Sequence[int],
    response_column: str = "during_hit",
) -> OdorResponseSummary:
    """Compute contingency-table counts for a dataset.

    Parameters
    ----------
    data:
        DataFrame containing the behavioural observations.
    dataset:
        Name of the dataset to analyse (matched against the ``dataset`` column).
    trained_trials / untrained_trials:
        Iterable of trial numbers that should be associated with the trained
        and untrained odours respectively.
    response_column:
        Column that indicates the binary response (defaults to ``during_hit``).
    """

    if "dataset" not in data.columns:
        raise KeyError("Input data must include a 'dataset' column")
    if "fly" not in data.columns:
        raise KeyError("Input data must include a 'fly' column")
    if "trial_num" not in data.columns:
        raise KeyError("Input data must include a 'trial_num' column")
    if response_column not in data.columns:
        raise KeyError(f"Input data must include a '{response_column}' column")

    trained_trials = _normalise_trials(trained_trials)
    untrained_trials = _normalise_trials(untrained_trials)

    if not trained_trials:
        raise ValueError("At least one trained trial must be provided")
    if not untrained_trials:
        raise ValueError("At least one untrained trial must be provided")

    dataset_df = data.loc[data["dataset"] == dataset].copy()
    if dataset_df.empty:
        raise ValueError(f"No rows found for dataset '{dataset}'")

    dataset_df[response_column] = _coerce_to_bool(dataset_df[response_column])

    dataset_df["trial_num"] = pd.to_numeric(dataset_df["trial_num"], errors="coerce")
    dataset_df = dataset_df.dropna(subset=["trial_num"])
    dataset_df["trial_num"] = dataset_df["trial_num"].astype(int)

    # Restrict to relevant trials only.
    relevant_trials = set(trained_trials) | set(untrained_trials)
    dataset_df = dataset_df.loc[dataset_df["trial_num"].isin(relevant_trials)]
    if dataset_df.empty:
        raise ValueError(
            "No rows remaining after filtering by trained/untrained trials. "
            "Please check the trial numbers supplied."
        )

    per_fly = dataset_df.groupby("fly")

    def _has_response(sub_df: pd.DataFrame, trials: Sequence[int]) -> bool:
        trial_mask = sub_df["trial_num"].isin(trials)
        if not trial_mask.any():
            return False
        return bool(sub_df.loc[trial_mask, response_column].any())

    both_positive = trained_only = untrained_only = both_negative = 0

    for _, fly_df in per_fly:
        trained_response = _has_response(fly_df, trained_trials)
        untrained_response = _has_response(fly_df, untrained_trials)

        if trained_response and untrained_response:
            both_positive += 1
        elif trained_response and not untrained_response:
            trained_only += 1
        elif not trained_response and untrained_response:
            untrained_only += 1
        else:
            both_negative += 1

    return OdorResponseSummary(
        both_positive=both_positive,
        trained_only=trained_only,
        untrained_only=untrained_only,
        both_negative=both_negative,
    )


# ---------------------------------------------------------------------------
# Plotting helpers


def build_contingency_table(summary: OdorResponseSummary) -> pd.DataFrame:
    """Create a DataFrame representing the full contingency table."""

    a, b, c, d = summary.as_tuple()

    table = pd.DataFrame(
        data=[
            [a, b, a + b],
            [c, d, c + d],
            [a + c, b + d, summary.grand_total],
        ],
        index=["Trained +", "Trained -", "Column total"],
        columns=["Untrained +", "Untrained -", "Row total"],
    )
    return table


def plot_contingency_table(
    table: pd.DataFrame,
    *,
    dataset: str,
    trained_trials: Sequence[int],
    untrained_trials: Sequence[int],
    output: Path,
) -> None:
    """Render the contingency table to ``output`` as an image."""

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axis("off")

    table_artist = ax.table(
        cellText=table.astype(int).values,
        rowLabels=table.index,
        colLabels=table.columns,
        loc="center",
        cellLoc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(12)
    table_artist.scale(1.2, 1.4)

    title = (
        f"Dataset: {dataset}\n"
        f"Trained trials: {', '.join(map(str, trained_trials))} | "
        f"Untrained trials: {', '.join(map(str, untrained_trials))}"
    )
    ax.set_title(title, fontsize=12, pad=16)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Command line interface


def _parse_trial_list(values: Sequence[str]) -> List[int]:
    if not values:
        return []
    try:
        return [int(v) for v in values]
    except ValueError as exc:
        raise ValueError("Trial numbers must be integers") from exc


def _create_argument_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Summarise trained vs. untrained odor responses for a dataset. "
            "Counts are computed using the during-hit responses only."
        )
    )
    parser.add_argument("csv", type=Path, help="Path to the input CSV file")
    parser.add_argument("dataset", help="Dataset name to analyse")
    parser.add_argument(
        "--trained-trials",
        nargs="+",
        default=[2, 4, 5],
        help="Trial numbers that correspond to the trained odor (default: 2 4 5)",
    )
    parser.add_argument(
        "--untrained-trials",
        nargs="+",
        required=True,
        help="Trial numbers that correspond to the untrained odor",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for saving a contingency-table figure",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _create_argument_parser()
    args = parser.parse_args(argv)

    trained_trials = _parse_trial_list(args.trained_trials)
    untrained_trials = _parse_trial_list(args.untrained_trials)

    df = pd.read_csv(args.csv)

    summary = summarize_odor_responses(
        df,
        dataset=args.dataset,
        trained_trials=trained_trials,
        untrained_trials=untrained_trials,
    )

    a, b, c, d = summary.as_tuple()
    print("Trained + / Untrained + (a):", a)
    print("Trained + / Untrained - (b):", b)
    print("Trained - / Untrained + (c):", c)
    print("Trained - / Untrained - (d):", d)
    print("Total flies:", summary.grand_total)

    if args.output is not None:
        table = build_contingency_table(summary)
        plot_contingency_table(
            table,
            dataset=args.dataset,
            trained_trials=trained_trials,
            untrained_trials=untrained_trials,
            output=args.output,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

