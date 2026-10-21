"""Threshold tuning for ``yes_no`` decision models.

A decision model's raw probability is not a decision until someone picks
the threshold that turns it into one -- and the near-universal default of
0.50 is an arbitrary choice that quietly leaves performance on the table.
Hamel Husain's decision-model evals (see the README) found exactly this:
models compared at their default 0.50 threshold ranked differently, and
looked worse, than the same saved scores at a tuned threshold.

This module tunes the threshold from *already saved* scores and human
labels.  It never calls a model: the scores in the results file are reused
as-is, so tuning is free -- the cost per 100 eval runs is identical at the
default and tuned thresholds, only the quality of the decisions changes.

Method: stratified K-fold cross-validation.  On each fold's training part,
the candidate threshold (drawn from the observed scores) with the best F1
among those meeting the caller's minimum precision and recall is chosen;
it is then scored on the held-out fold, so the reported cross-validated
metrics estimate how the *tuning procedure* generalises rather than how
well one threshold overfits the full dataset.  A full-data tuned threshold
is reported as the recommendation to deploy.  If no candidate meets both
minimums, the best-F1 candidate is used and ``constraint_met`` is False --
the report says so instead of pretending the floors were reached.

Only ``yes_no`` decisions participate (their probability *is* a binary
classifier score).  ``choice`` and ``score`` results are counted as
skipped: they have no single threshold to tune.

Cost accounting: ``cost_per_100_usd`` is the mean recorded cost per
``yes_no`` call, times 100 -- the unit Hamel's comparison used.
``cost_per_correct_usd`` divides the total recorded ``yes_no`` cost by
the number of correct decisions *at that threshold*, so it does move
with the threshold.  Both are None when no costs were recorded.
"""

from __future__ import annotations

import random
from typing import Any, Iterable

from .data import ResultRecord
from .schemas import Case, DecisionValidationError, YesNoDecision, parse_decision

#: The default everyone ships and nobody tunes.
DEFAULT_THRESHOLD = 0.5


def confusion_metrics(
    scores: list[float], labels: list[bool], threshold: float
) -> dict[str, Any]:
    """Precision / recall / F1 / accuracy of ``score >= threshold``.

    ``labels`` are the human ground truth (True = the positive, "yes",
    class).  Following the usual ``zero_division=0`` convention, a metric
    whose denominator is empty (e.g. precision when nothing is predicted
    positive) is reported as 0.0.
    """
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        predicted = score >= threshold
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and label:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    total = tp + fp + tn + fn
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": ((tp + tn) / total) if total else 0.0,
    }


def _select_threshold(
    scores: list[float],
    labels: list[bool],
    min_precision: float,
    min_recall: float,
) -> tuple[float, dict[str, Any], bool]:
    """Best candidate threshold for one (training) set of scores.

    Candidates are the observed score values: with a ``>=`` rule, every
    threshold between two consecutive observed scores classifies exactly
    like the higher one, so no distinct outcome is skipped.  Feasible
    candidates (both floors met) outrank infeasible ones; within a class,
    order by F1, then precision, then recall, then the higher threshold
    (fewer positive calls wins an exact tie).
    """
    best: tuple[tuple, float, dict[str, Any], bool] | None = None
    for threshold in sorted(set(scores)):
        metrics = confusion_metrics(scores, labels, threshold)
        feasible = (
            metrics["precision"] >= min_precision and metrics["recall"] >= min_recall
        )
        key = (
            1 if feasible else 0,
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            threshold,
        )
        if best is None or key > best[0]:
            best = (key, threshold, metrics, feasible)
    assert best is not None  # scores is non-empty by construction
    return best[1], best[2], best[3]


def _stratified_folds(labels: list[bool], n_folds: int, seed: int) -> list[list[int]]:
    """Split indices into ``n_folds`` folds, each keeping the class ratio."""
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for wanted in (True, False):
        members = [i for i, label in enumerate(labels) if label == wanted]
        rng.shuffle(members)
        for j, index in enumerate(members):
            folds[j % n_folds].append(index)
    return folds


def tune_threshold(
    cases: Iterable[Case],
    results: Iterable[ResultRecord],
    min_precision: float = 0.8,
    min_recall: float = 0.8,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Tune the ``yes_no`` decision threshold from saved scores + labels.

    Reuses the probabilities in ``results``; makes no model calls.  Fails
    loudly on an unknown ``case_id`` or a malformed decision (same policy
    as :func:`decidekit.metrics.evaluate`), on data with no ``yes_no``
    results, on a single-class label set, and when either class has fewer
    than two examples (cross-validation would be meaningless).
    """
    if not 0.0 <= min_precision <= 1.0:
        raise DecisionValidationError(
            f"min_precision must be in [0, 1], got {min_precision}"
        )
    if not 0.0 <= min_recall <= 1.0:
        raise DecisionValidationError(f"min_recall must be in [0, 1], got {min_recall}")
    if n_folds < 2:
        raise DecisionValidationError(f"n_folds must be >= 2, got {n_folds}")

    case_by_id = {c.id: c for c in cases}
    scores: list[float] = []
    labels: list[bool] = []
    costs: list[float] = []
    n_skipped = 0
    for record in results:
        case = case_by_id.get(record.case_id)
        if case is None:
            raise DecisionValidationError(
                f"result references unknown case_id {record.case_id!r} (not in the dataset)"
            )
        decision = parse_decision(record.decision, case)
        if not isinstance(decision, YesNoDecision):
            n_skipped += 1
            continue
        scores.append(decision.probability)
        labels.append(bool(case.correct))
        if record.cost_usd is not None:
            costs.append(record.cost_usd)

    n = len(scores)
    if n == 0:
        raise DecisionValidationError(
            "threshold tuning needs yes_no results; the results file has none"
        )
    n_positive = sum(1 for label in labels if label)
    n_negative = n - n_positive
    if n_positive == 0 or n_negative == 0:
        raise DecisionValidationError(
            "threshold tuning needs both positive and negative labels; "
            f"got {n_positive} positive / {n_negative} negative"
        )
    effective_folds = min(n_folds, n_positive, n_negative)
    if effective_folds < 2:
        raise DecisionValidationError(
            "threshold tuning needs at least 2 examples of each class for "
            f"cross-validation; got {n_positive} positive / {n_negative} negative"
        )

    any_cost = bool(costs)
    total_cost = sum(costs)
    cost_per_100 = (total_cost / len(costs) * 100) if any_cost else None

    def with_cost(metrics: dict[str, Any]) -> dict[str, Any]:
        correct = metrics["tp"] + metrics["tn"]
        return {
            **metrics,
            "cost_per_100_usd": cost_per_100,
            "cost_per_correct_usd": (total_cost / correct)
            if (any_cost and correct > 0)
            else None,
        }

    default = with_cost(confusion_metrics(scores, labels, DEFAULT_THRESHOLD))

    tuned_threshold_value, tuned_full, tuned_feasible = _select_threshold(
        scores, labels, min_precision, min_recall
    )
    tuned = {**with_cost(tuned_full), "constraint_met": tuned_feasible}

    folds = _stratified_folds(labels, effective_folds, seed)
    all_indices = set(range(n))
    fold_reports: list[dict[str, Any]] = []
    for fold_no, validation_idx in enumerate(folds):
        train_idx = sorted(all_indices - set(validation_idx))
        fold_threshold, _, fold_feasible = _select_threshold(
            [scores[i] for i in train_idx],
            [labels[i] for i in train_idx],
            min_precision,
            min_recall,
        )
        validation_metrics = confusion_metrics(
            [scores[i] for i in validation_idx],
            [labels[i] for i in validation_idx],
            fold_threshold,
        )
        fold_reports.append(
            {
                "fold": fold_no,
                "n_validation": len(validation_idx),
                "threshold": fold_threshold,
                "constraint_met_on_train": fold_feasible,
                "precision": validation_metrics["precision"],
                "recall": validation_metrics["recall"],
                "f1": validation_metrics["f1"],
                "accuracy": validation_metrics["accuracy"],
            }
        )

    def _mean(key: str) -> float:
        return sum(f[key] for f in fold_reports) / len(fold_reports)

    return {
        "n": n,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_skipped_non_yes_no": n_skipped,
        "min_precision": min_precision,
        "min_recall": min_recall,
        "n_folds": effective_folds,
        "n_folds_requested": n_folds,
        "seed": seed,
        "default": default,
        "tuned": tuned,
        "cross_validation": {
            "folds": fold_reports,
            "mean_precision": _mean("precision"),
            "mean_recall": _mean("recall"),
            "mean_f1": _mean("f1"),
            "mean_accuracy": _mean("accuracy"),
            "thresholds": [f["threshold"] for f in fold_reports],
        },
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _fmt_usd(value: float | None) -> str:
    return "—" if value is None else f"${value:.4f}"


def render_threshold_report(report: dict[str, Any], model: str | None = None) -> str:
    """Render a tuning report as a plain-text side-by-side comparison."""
    default = report["default"]
    tuned = report["tuned"]
    cv = report["cross_validation"]
    lines: list[str] = []
    if model:
        lines.append(f"Model: {model}")
    lines.append(
        f"yes_no cases: {report['n']} "
        f"({report['n_positive']} positive / {report['n_negative']} negative)"
    )
    if report["n_skipped_non_yes_no"]:
        lines.append(
            f"Skipped {report['n_skipped_non_yes_no']} non-yes_no result(s) "
            "(threshold tuning applies to yes_no only)"
        )
    lines.append(
        f"Floors: precision >= {report['min_precision']:.2f}, "
        f"recall >= {report['min_recall']:.2f}   "
        f"Validation: {report['n_folds']}-fold stratified cross-validation "
        f"(seed {report['seed']})"
    )
    lines.append("")
    header = (
        f"{'':<10}{'threshold':>10}{'precision':>11}{'recall':>9}"
        f"{'F1':>8}{'accuracy':>10}{'$/100':>10}{'$/correct':>11}"
    )
    lines.append(header)

    def row(label: str, m: dict[str, Any]) -> str:
        return (
            f"{label:<10}{m['threshold']:>10.3f}{m['precision']:>11.3f}"
            f"{m['recall']:>9.3f}{m['f1']:>8.3f}{m['accuracy']:>10.3f}"
            f"{_fmt_usd(m['cost_per_100_usd']):>10}"
            f"{_fmt_usd(m['cost_per_correct_usd']):>11}"
        )

    lines.append(row("default", default))
    lines.append(row("tuned", tuned))
    lines.append("")
    if tuned["constraint_met"]:
        lines.append(
            f"Recommended threshold: {tuned['threshold']:.3f} "
            "(meets both floors on the full dataset)"
        )
    else:
        lines.append(
            f"Recommended threshold: {tuned['threshold']:.3f} "
            "(no threshold meets both floors on this data; best F1 shown)"
        )
    lines.append(
        f"Out-of-fold (cross-validated): precision {_fmt(cv['mean_precision'])}, "
        f"recall {_fmt(cv['mean_recall'])}, F1 {_fmt(cv['mean_f1'])}, "
        f"accuracy {_fmt(cv['mean_accuracy'])}; "
        f"per-fold thresholds: "
        + ", ".join(f"{t:.3f}" for t in cv["thresholds"])
    )
    lines.append(
        "Tuning reuses saved scores -- no new model calls were made, so "
        "$/100 is the same at both thresholds; only decision quality moves."
    )
    return "\n".join(lines) + "\n"
