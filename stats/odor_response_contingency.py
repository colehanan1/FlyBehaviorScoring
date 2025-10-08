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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Sequence

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


def create_contingency_figure(
    table: pd.DataFrame,
    *,
    dataset: str,
    trained_trials: Sequence[int],
    untrained_trials: Sequence[int],
    style: ContingencyTableStyle | None = None,
):
    """Create a Matplotlib figure representing the contingency table."""

    style = style or ContingencyTableStyle()

    fig_kwargs = {}
    if style.figure_size is not None:
        fig_kwargs["figsize"] = style.figure_size
    else:
        fig_kwargs["figsize"] = (6, 3.5)

    fig, ax = plt.subplots(**fig_kwargs)
    ax.axis("off")

    row_labels = style.resolve_row_labels(table.index.tolist())
    column_labels = style.resolve_column_labels(table.columns.tolist())

    table_kwargs = dict(style.table_kwargs)
    table_kwargs.setdefault("cellLoc", "center")

    table_artist = ax.table(
        cellText=table.astype(int).values,
        rowLabels=row_labels,
        colLabels=column_labels,
        loc="center",
        **table_kwargs,
    )

    style.apply_to_table(table_artist, table)

    title, title_kwargs = style.render_title(
        dataset=dataset, trained_trials=trained_trials, untrained_trials=untrained_trials
    )
    if "fontsize" not in title_kwargs:
        title_kwargs["fontsize"] = 12
    if "pad" not in title_kwargs:
        title_kwargs["pad"] = 16
    ax.set_title(title, **title_kwargs)

    fig.tight_layout()
    return fig, ax, table_artist


def plot_contingency_table(
    table: pd.DataFrame,
    *,
    dataset: str,
    trained_trials: Sequence[int],
    untrained_trials: Sequence[int],
    output: Path,
    style: ContingencyTableStyle | None = None,
) -> None:
    """Render the contingency table to ``output`` as an EPS figure."""

    if output.suffix.lower() != ".eps":
        raise ValueError("Output path must end with '.eps' for EPS export")

    fig, _, _ = create_contingency_figure(
        table,
        dataset=dataset,
        trained_trials=trained_trials,
        untrained_trials=untrained_trials,
        style=style,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="eps", dpi=300)
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
        help="Optional path for saving a contingency-table EPS figure (*.eps)",
    )
    parser.add_argument(
        "--style",
        type=Path,
        help=(
            "Optional JSON file defining styling overrides for the contingency "
            "figure"
        ),
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

    style = None
    if args.style is not None:
        style = ContingencyTableStyle.from_json(args.style)

    if args.output is not None:
        table = build_contingency_table(summary)
        plot_contingency_table(
            table,
            dataset=args.dataset,
            trained_trials=trained_trials,
            untrained_trials=untrained_trials,
            output=args.output,
            style=style,
        )
    elif style is not None:
        # Allow previewing the styled figure interactively when running via CLI.
        table = build_contingency_table(summary)
        fig, _, _ = create_contingency_figure(
            table,
            dataset=args.dataset,
            trained_trials=trained_trials,
            untrained_trials=untrained_trials,
            style=style,
        )
        plt.show()
        plt.close(fig)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# Styling helpers


@dataclass
class ContingencyTableStyle:
    """Styling controls for :func:`plot_contingency_table`."""

    figure_size: tuple[float, float] | None = None
    row_labels: Sequence[str] | None = None
    column_labels: Sequence[str] | None = None
    title_template: str | None = None
    title_kwargs: Mapping[str, object] = field(default_factory=dict)
    table_kwargs: Mapping[str, object] = field(default_factory=dict)
    scale: tuple[float, float] | None = None
    font_size: float | None = 12.0
    auto_font_size: bool = False
    cell_text_color: str | None = None
    header_text_color: str | None = None
    row_label_text_color: str | None = None
    cell_facecolors: Sequence[Sequence[str | None]] | None = None
    row_label_facecolor: str | None = None
    column_label_facecolor: str | None = None
    totals_facecolor: str | None = None
    edge_color: str | None = None
    background_color: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ContingencyTableStyle":
        """Create a style object from a mapping (e.g. JSON data)."""

        def _maybe_tuple(value):
            if value is None:
                return None
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return tuple(value)
            raise TypeError("Expected a sequence for scale/figure_size fields")

        payload = dict(payload)  # shallow copy we can mutate

        figure_size = payload.pop("figure_size", None)
        if figure_size is not None:
            payload["figure_size"] = _maybe_tuple(figure_size)

        scale = payload.pop("scale", None)
        if scale is not None:
            payload["scale"] = _maybe_tuple(scale)

        title_kwargs = payload.pop("title_kwargs", None)
        if title_kwargs is not None and not isinstance(title_kwargs, Mapping):
            raise TypeError("title_kwargs must be a mapping")
        if title_kwargs is not None:
            payload["title_kwargs"] = dict(title_kwargs)

        table_kwargs = payload.pop("table_kwargs", None)
        if table_kwargs is not None and not isinstance(table_kwargs, Mapping):
            raise TypeError("table_kwargs must be a mapping")
        if table_kwargs is not None:
            payload["table_kwargs"] = dict(table_kwargs)

        return cls(**payload)

    @classmethod
    def from_json(cls, path: Path) -> "ContingencyTableStyle":
        """Load a style from a JSON file."""

        import json

        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, Mapping):
            raise TypeError("Style JSON must define an object at the top level")
        return cls.from_mapping(payload)

    def resolve_row_labels(self, default: Sequence[str]) -> Sequence[str]:
        if self.row_labels is None:
            return default
        if len(self.row_labels) != len(default):
            raise ValueError(
                "row_labels must contain exactly "
                f"{len(default)} entries (received {len(self.row_labels)})"
            )
        return self.row_labels

    def resolve_column_labels(self, default: Sequence[str]) -> Sequence[str]:
        if self.column_labels is None:
            return default
        if len(self.column_labels) != len(default):
            raise ValueError(
                "column_labels must contain exactly "
                f"{len(default)} entries (received {len(self.column_labels)})"
            )
        return self.column_labels

    def render_title(
        self,
        *,
        dataset: str,
        trained_trials: Sequence[int],
        untrained_trials: Sequence[int],
    ) -> tuple[str, MutableMapping[str, object]]:
        if self.title_template is None:
            title = (
                f"Dataset: {dataset}\n"
                f"Trained trials: {', '.join(map(str, trained_trials))} | "
                f"Untrained trials: {', '.join(map(str, untrained_trials))}"
            )
        else:
            title = self.title_template.format(
                dataset=dataset,
                trained_trials=trained_trials,
                untrained_trials=untrained_trials,
            )
        return title, dict(self.title_kwargs)

    def apply_to_table(self, table_artist, table: pd.DataFrame) -> None:
        """Apply the style customisations to a Matplotlib table artist."""

        if self.auto_font_size:
            table_artist.auto_set_font_size(True)
        else:
            table_artist.auto_set_font_size(False)
            if self.font_size is not None:
                table_artist.set_fontsize(self.font_size)

        if self.scale is not None:
            table_artist.scale(*self.scale)
        else:
            table_artist.scale(1.2, 1.4)

        cells = table_artist.get_celld()

        def _set_text_color(indices, color):
            if color is None:
                return
            for idx in indices:
                if idx in cells:
                    cells[idx].get_text().set_color(color)

        n_rows, n_cols = table.shape

        # Header row (column labels)
        header_indices = [(0, col) for col in range(n_cols)]
        _set_text_color(header_indices, self.header_text_color)
        if self.column_label_facecolor is not None:
            for idx in header_indices:
                if idx in cells:
                    cells[idx].set_facecolor(self.column_label_facecolor)

        # Row labels (stored in column -1)
        row_label_indices = [(row, -1) for row in range(1, n_rows + 1)]
        _set_text_color(row_label_indices, self.row_label_text_color)
        if self.row_label_facecolor is not None:
            for idx in row_label_indices:
                if idx in cells:
                    cells[idx].set_facecolor(self.row_label_facecolor)

        # Data cells (including totals)
        body_indices = [(row, col) for row in range(1, n_rows + 1) for col in range(n_cols)]
        _set_text_color(body_indices, self.cell_text_color)

        if self.edge_color is not None:
            for cell in cells.values():
                cell.set_edgecolor(self.edge_color)

        if self.background_color is not None:
            table_artist.patch.set_facecolor(self.background_color)

        if self.totals_facecolor is not None:
            total_rows = [n_rows]
            total_cols = [n_cols - 1]
            for row in total_rows:
                for col in range(n_cols):
                    idx = (row, col)
                    if idx in cells:
                        cells[idx].set_facecolor(self.totals_facecolor)
                idx = (row, -1)
                if idx in cells:
                    cells[idx].set_facecolor(self.totals_facecolor)
            for row in range(1, n_rows + 1):
                idx = (row, total_cols[0])
                if idx in cells:
                    cells[idx].set_facecolor(self.totals_facecolor)

        if self.cell_facecolors is not None:
            if len(self.cell_facecolors) != n_rows:
                raise ValueError(
                    "cell_facecolors must provide a colour for each row, "
                    f"expected {n_rows} rows"
                )
            for row_idx, row_colors in enumerate(self.cell_facecolors, start=1):
                if len(row_colors) != n_cols:
                    raise ValueError(
                        "Each row in cell_facecolors must have "
                        f"{n_cols} entries (row {row_idx})"
                    )
                for col_idx, color in enumerate(row_colors):
                    if color is None:
                        continue
                    idx = (row_idx, col_idx)
                    if idx in cells:
                        cells[idx].set_facecolor(color)


