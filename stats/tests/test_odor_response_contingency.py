import pandas as pd
import pytest

from ..odor_response_contingency import (
    OdorResponseSummary,
    build_contingency_table,
    plot_contingency_table,
    summarize_odor_responses,
)


def _make_dataframe():
    # Dataset with three flies and mixed responses. The "during_hit" values are
    # encoded in a variety of ways to ensure coercion works as expected.
    return pd.DataFrame(
        {
            "dataset": [
                "sessionA",
                "sessionA",
                "sessionA",
                "sessionA",
                "sessionA",
                "sessionA",
                "sessionA",
                "sessionA",
                "sessionA",
            ],
            "fly": [
                "fly1",
                "fly1",
                "fly1",
                "fly2",
                "fly2",
                "fly2",
                "fly3",
                "fly3",
                "fly3",
            ],
            "trial_num": [2, 4, 6, 2, 4, 1, 2, 4, 1],
            "during_hit": [1, 0, 1, "true", "False", 0, 0, 0, 0],
        }
    )


def test_summarize_odor_responses_counts():
    df = _make_dataframe()

    summary = summarize_odor_responses(
        df,
        dataset="sessionA",
        trained_trials=[2, 4, 5],
        untrained_trials=[1, 6],
    )

    assert summary == OdorResponseSummary(
        both_positive=1,  # fly1 responds to trained (trial 2) and untrained (trial 6)
        trained_only=1,  # fly2 responds to trained only (trial 2)
        untrained_only=0,
        both_negative=1,  # fly3 never responds
    )


def test_build_contingency_table_structure():
    summary = OdorResponseSummary(3, 2, 1, 4)
    table = build_contingency_table(summary)

    assert list(table.index) == ["Trained +", "Trained -", "Column total"]
    assert list(table.columns) == ["Untrained +", "Untrained -", "Row total"]
    assert table.loc["Trained +", "Untrained +"] == 3
    assert table.loc["Trained +", "Row total"] == 5
    assert table.loc["Column total", "Untrained -"] == 6
    assert table.loc["Column total", "Row total"] == 10


def test_plot_contingency_table_writes_eps(tmp_path):
    summary = OdorResponseSummary(1, 2, 3, 4)
    table = build_contingency_table(summary)

    output = tmp_path / "contingency.eps"

    plot_contingency_table(
        table,
        dataset="sessionA",
        trained_trials=[2, 4, 5],
        untrained_trials=[1, 3, 6],
        output=output,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_contingency_table_rejects_non_eps(tmp_path):
    summary = OdorResponseSummary(1, 0, 0, 1)
    table = build_contingency_table(summary)

    with pytest.raises(ValueError, match=r"\.eps"):
        plot_contingency_table(
            table,
            dataset="sessionA",
            trained_trials=[2],
            untrained_trials=[1],
            output=tmp_path / "contingency.png",
        )
