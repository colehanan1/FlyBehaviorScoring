import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ..odor_response_contingency import (
    ContingencyTableStyle,
    OdorResponseSummary,
    build_contingency_table,
    create_contingency_figure,
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


def test_create_contingency_figure_allows_custom_styling(tmp_path):
    summary = OdorResponseSummary(1, 2, 3, 4)
    table = build_contingency_table(summary)

    style = ContingencyTableStyle(
        figure_size=(4, 4),
        row_labels=["Learnt", "Not learnt", "Totals"],
        column_labels=["Novel +", "Novel -", "Totals"],
        title_template="Summary for {dataset}",
        font_size=14,
        cell_text_color="blue",
        header_text_color="white",
        row_label_text_color="red",
        column_label_facecolor="#222222",
        row_label_facecolor="#dddddd",
        totals_facecolor="#ffeeaa",
        cell_facecolors=[
            ["#ffeeee", "#ffeeee", "#ffeeaa"],
            ["#eefeee", "#eefeee", "#ffeeaa"],
            ["#ffeeaa", "#ffeeaa", "#ffeeaa"],
        ],
        edge_color="black",
        table_kwargs={"cellLoc": "center"},
    )

    fig, ax, table_artist = create_contingency_figure(
        table,
        dataset="sessionA",
        trained_trials=[2, 4, 5],
        untrained_trials=[1, 3, 6],
        style=style,
    )

    cells = table_artist.get_celld()
    assert ax.get_title() == "Summary for sessionA"
    assert cells[(0, 0)].get_text().get_text() == "Novel +"
    assert cells[(1, -1)].get_text().get_text() == "Learnt"
    assert cells[(1, 0)].get_facecolor()[:3] == pytest.approx((1.0, 0.933333, 0.933333), rel=1e-3)
    assert cells[(3, 2)].get_facecolor()[:3] == pytest.approx((1.0, 0.933333, 0.666667), rel=1e-3)

    output = tmp_path / "styled.eps"
    plot_contingency_table(
        table,
        dataset="sessionA",
        trained_trials=[2, 4, 5],
        untrained_trials=[1, 3, 6],
        output=output,
        style=style,
    )
    assert output.exists()
    plt.close(fig)


def test_style_factory_validates_lengths():
    style = ContingencyTableStyle(row_labels=["only one label"])
    table = build_contingency_table(OdorResponseSummary(1, 1, 1, 1))

    with pytest.raises(ValueError):
        style.resolve_row_labels(table.index.tolist())

    with pytest.raises(ValueError):
        ContingencyTableStyle(column_labels=["A"]).resolve_column_labels(table.columns.tolist())


def test_style_from_json(tmp_path):
    import json

    style_path = tmp_path / "style.json"
    style_path.write_text(
        json.dumps(
            {
                "figure_size": [8, 5],
                "scale": [1.5, 1.6],
                "title_template": "Report {dataset}",
                "title_kwargs": {"fontsize": 18},
                "table_kwargs": {"cellLoc": "left"},
            }
        ),
        encoding="utf-8",
    )

    style = ContingencyTableStyle.from_json(style_path)

    assert style.figure_size == (8, 5)
    assert style.scale == (1.5, 1.6)
    assert style.title_template == "Report {dataset}"
    assert style.title_kwargs == {"fontsize": 18}
    assert style.table_kwargs == {"cellLoc": "left"}
