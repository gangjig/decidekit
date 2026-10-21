"""Threshold tuning: confusion metrics, selection, CV, costs, CLI."""

import json
from pathlib import Path

import pytest

from decidekit import (
    Case,
    DecisionValidationError,
    ResultRecord,
    confusion_metrics,
    render_threshold_report,
    tune_threshold,
)
from decidekit.cli import main

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATASET = str(DATA / "sample_dataset.jsonl")
CAREFUL = str(DATA / "sample_results_careful-1.jsonl")


def _dataset(pairs):
    """pairs: list of (probability, label) -> (cases, records) for yes_no."""
    cases, records = [], []
    for i, (prob, label) in enumerate(pairs):
        cases.append(Case(id=f"c{i}", kind="yes_no", prompt="?", correct=label))
        records.append(
            ResultRecord(
                case_id=f"c{i}",
                model="m",
                decision={"probability": prob},
                cost_usd=0.01,
            )
        )
    return cases, records


SEPARABLE = [(0.9, True), (0.8, True), (0.7, True), (0.6, True),
             (0.4, False), (0.3, False), (0.2, False), (0.1, False)]

# The Hamel pattern: two real positives sit below the 0.50 default, so the
# default threshold throws away half the recall for no reason.
HAMEL = [(0.95, True), (0.90, True), (0.45, True), (0.42, True),
         (0.30, False), (0.20, False), (0.10, False), (0.05, False)]


def test_confusion_metrics_known_values():
    m = confusion_metrics([0.9, 0.4, 0.6, 0.2], [True, True, False, False], 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5
    assert m["accuracy"] == 0.5


def test_confusion_metrics_no_predicted_positives_is_zero_not_nan():
    m = confusion_metrics([0.1, 0.2], [True, False], 0.5)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_separable_data_tunes_cleanly():
    cases, records = _dataset(SEPARABLE)
    report = tune_threshold(cases, records)
    assert report["tuned"]["constraint_met"] is True
    assert report["tuned"]["f1"] == 1.0
    assert report["default"]["f1"] == 1.0
    assert report["n"] == 8
    assert report["n_positive"] == 4


def test_default_threshold_leaves_recall_on_the_table():
    cases, records = _dataset(HAMEL)
    report = tune_threshold(cases, records)
    assert report["default"]["recall"] == 0.5
    assert report["default"]["threshold"] == 0.5
    tuned = report["tuned"]
    assert tuned["constraint_met"] is True
    assert tuned["threshold"] == 0.42
    assert tuned["recall"] == 1.0
    assert tuned["precision"] == 1.0
    assert tuned["f1"] > report["default"]["f1"]


def test_infeasible_floors_are_reported_not_hidden():
    # Overlapping scores: no threshold reaches precision 0.8 AND recall 0.8.
    cases, records = _dataset([(0.6, True), (0.4, True), (0.55, False), (0.45, False)])
    report = tune_threshold(cases, records)
    assert report["tuned"]["constraint_met"] is False
    # Best-F1 fallback: threshold 0.6 has F1 2/3 with precision 1.0.
    assert report["tuned"]["threshold"] == 0.6


def test_cross_validation_shape_and_determinism():
    cases, records = _dataset(HAMEL)
    first = tune_threshold(cases, records)
    second = tune_threshold(cases, records)
    assert first == second  # same seed -> identical report
    cv = first["cross_validation"]
    assert first["n_folds"] == 4  # min(5 requested, 4 pos, 4 neg)
    assert len(cv["folds"]) == 4
    assert len(cv["thresholds"]) == 4
    assert 0.0 <= cv["mean_f1"] <= 1.0
    total_validation = sum(f["n_validation"] for f in cv["folds"])
    assert total_validation == 8


def test_costs_per_100_identical_at_both_thresholds():
    cases, records = _dataset(HAMEL)
    report = tune_threshold(cases, records)
    # 8 records at $0.01 -> $1.00 per 100 eval runs, threshold-independent.
    assert report["default"]["cost_per_100_usd"] == pytest.approx(1.0)
    assert report["tuned"]["cost_per_100_usd"] == pytest.approx(1.0)
    # Cost per correct DOES move: default gets 6/8 right, tuned gets 8/8.
    assert report["default"]["cost_per_correct_usd"] == pytest.approx(0.08 / 6)
    assert report["tuned"]["cost_per_correct_usd"] == pytest.approx(0.08 / 8)


def test_non_yes_no_results_are_skipped_and_counted():
    cases, records = _dataset(SEPARABLE)
    cases.append(
        Case(id="ch", kind="choice", prompt="?", correct="a", options=("a", "b"))
    )
    records.append(
        ResultRecord(
            case_id="ch",
            model="m",
            decision={"picked": "a", "probabilities": {"a": 0.7, "b": 0.3}},
        )
    )
    report = tune_threshold(cases, records)
    assert report["n_skipped_non_yes_no"] == 1
    assert report["n"] == 8


def test_single_class_labels_fail_loudly():
    cases, records = _dataset([(0.9, True), (0.8, True), (0.7, True)])
    with pytest.raises(DecisionValidationError, match="both positive and negative"):
        tune_threshold(cases, records)


def test_too_few_of_a_class_fails_loudly():
    cases, records = _dataset([(0.9, True), (0.8, True), (0.1, False)])
    with pytest.raises(DecisionValidationError, match="at least 2"):
        tune_threshold(cases, records)


def test_no_yes_no_results_fail_loudly():
    cases = [Case(id="ch", kind="choice", prompt="?", correct="a", options=("a", "b"))]
    records = [
        ResultRecord(
            case_id="ch",
            model="m",
            decision={"picked": "a", "probabilities": {"a": 0.7, "b": 0.3}},
        )
    ]
    with pytest.raises(DecisionValidationError, match="yes_no"):
        tune_threshold(cases, records)


def test_unknown_case_fails_loudly():
    cases, records = _dataset(SEPARABLE)
    records.append(
        ResultRecord(case_id="nope", model="m", decision={"probability": 0.5})
    )
    with pytest.raises(DecisionValidationError, match="unknown case_id"):
        tune_threshold(cases, records)


def test_invalid_parameters_fail_loudly():
    cases, records = _dataset(SEPARABLE)
    with pytest.raises(DecisionValidationError, match="min_precision"):
        tune_threshold(cases, records, min_precision=1.5)
    with pytest.raises(DecisionValidationError, match="n_folds"):
        tune_threshold(cases, records, n_folds=1)


def test_render_mentions_both_rows_and_recommendation():
    cases, records = _dataset(HAMEL)
    text = render_threshold_report(tune_threshold(cases, records), model="m")
    assert "default" in text
    assert "tuned" in text
    assert "Recommended threshold: 0.420" in text
    assert "no new model calls" in text


def test_cli_tune_on_sample_data(capsys):
    rc = main(["tune", "--dataset", DATASET, "--results", CAREFUL])
    out = capsys.readouterr().out
    assert rc == 0
    assert "careful-1" in out
    assert "Recommended threshold" in out


def test_cli_tune_json(capsys):
    rc = main(["tune", "--dataset", DATASET, "--results", CAREFUL, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["model"] == "careful-1"
    assert payload["n"] == 6  # 6 yes_no cases in the sample dataset
    assert payload["n_folds"] == 3  # min(5, 3 positive, 3 negative)
    assert payload["tuned"]["threshold"] is not None
