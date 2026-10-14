from decidekit.report import build_comparison, render_markdown


def metrics(accuracy, brier, ece, mae=None, cpc=None, latency=None):
    return {
        "accuracy": accuracy,
        "brier": brier,
        "log_loss": 0.5,
        "ece": ece,
        "score_mae": mae,
        "total_cost_usd": None,
        "cost_per_correct_usd": cpc,
        "mean_latency_ms": latency,
        "n_scored": 10,
        "n_correct": 8,
        "n_categorical": 10,
    }


def test_ranking_by_accuracy_descending():
    comparison = build_comparison(
        {
            "weak": metrics(0.60, 0.30, 0.20),
            "strong": metrics(0.95, 0.05, 0.02),
            "mid": metrics(0.80, 0.15, 0.10),
        }
    )
    assert comparison["ranking"] == ["strong", "mid", "weak"]


def test_ranking_tie_breaks_on_brier():
    comparison = build_comparison(
        {
            "sloppy": metrics(0.80, 0.30, 0.05),
            "honest": metrics(0.80, 0.12, 0.05),
        }
    )
    assert comparison["ranking"] == ["honest", "sloppy"]


def test_ranking_is_input_order_independent():
    a = build_comparison({"x": metrics(0.7, 0.2, 0.1), "y": metrics(0.9, 0.1, 0.1)})
    b = build_comparison({"y": metrics(0.9, 0.1, 0.1), "x": metrics(0.7, 0.2, 0.1)})
    assert a["ranking"] == b["ranking"] == ["y", "x"]


def test_render_markdown_table_shape():
    comparison = build_comparison(
        {
            "alpha": metrics(0.9, 0.1, 0.03, mae=1.5, cpc=0.0052, latency=813),
            "beta": metrics(0.7, 0.25, 0.12, mae=2.0, cpc=0.0015, latency=310),
        }
    )
    md = render_markdown(comparison)
    lines = [line for line in md.strip().splitlines() if line]
    assert lines[0].startswith("| Model |")
    assert len(lines) == 4  # header, divider, two rows
    assert lines[2].startswith("| alpha |")  # ranked first
    assert "0.900" in lines[2]
    assert "$0.0052" in lines[2]
    assert "| beta |" in lines[3]


def test_render_markdown_missing_values_dash():
    comparison = build_comparison({"bare": metrics(0.5, 0.4, 0.3)})
    md = render_markdown(comparison)
    assert "—" in md  # no MAE / cost / latency data
