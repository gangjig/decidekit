"""Comparison reports: rank N models over the same dataset.

Ranking is by accuracy (descending); ties break on Brier score (ascending),
then ECE (ascending), then model name -- deterministic, so a report is a
stable artifact you can diff in review.
"""

from __future__ import annotations

from typing import Any

_COLUMNS = [
    ("model", "Model"),
    ("accuracy", "Accuracy"),
    ("brier", "Brier"),
    ("log_loss", "Log loss"),
    ("ece", "ECE"),
    ("score_mae", "Score MAE"),
    ("cost_per_correct_usd", "$ / correct"),
    ("mean_latency_ms", "Latency (ms)"),
]


def _sort_key(row: dict[str, Any]) -> tuple:
    accuracy = row.get("accuracy")
    brier = row.get("brier")
    ece = row.get("ece")
    return (
        -(accuracy if accuracy is not None else -1.0),
        brier if brier is not None else float("inf"),
        ece if ece is not None else float("inf"),
        row["model"],
    )


def build_comparison(named_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build a comparison dict from ``{model_name: evaluate(...) output}``."""
    rows = []
    for model, metrics in named_metrics.items():
        row = {"model": model}
        for key in (
            "accuracy",
            "brier",
            "log_loss",
            "ece",
            "score_mae",
            "total_cost_usd",
            "cost_per_correct_usd",
            "mean_latency_ms",
            "n_scored",
            "n_correct",
            "n_categorical",
        ):
            row[key] = metrics.get(key)
        rows.append(row)
    rows.sort(key=_sort_key)
    return {"rows": rows, "ranking": [r["model"] for r in rows]}


def _fmt(key: str, value: Any) -> str:
    if value is None:
        return "—"
    if key in ("accuracy", "brier", "log_loss", "ece", "score_mae"):
        return f"{value:.3f}"
    if key in ("cost_per_correct_usd",):
        return f"${value:.4f}"
    if key == "mean_latency_ms":
        return f"{value:.0f}"
    return str(value)


def render_markdown(comparison: dict[str, Any]) -> str:
    """Render a comparison dict as a Markdown table."""
    rows = comparison["rows"]
    lines = [
        "| " + " | ".join(title for _, title in _COLUMNS) + " |",
        "|" + "|".join(" --- " for _ in _COLUMNS) + "|",
    ]
    for row in rows:
        cells = []
        for key, _ in _COLUMNS:
            cells.append(row["model"] if key == "model" else _fmt(key, row.get(key)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
