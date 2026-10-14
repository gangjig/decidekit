"""Scoring: accuracy, MAE, Brier score, log loss, and calibration (ECE).

Probabilistic metrics (Brier, log loss, calibration) are computed over the
categorical decisions -- ``yes_no`` and ``choice`` -- where the model emits
a probability distribution.  Calibration uses the standard
confidence-vs-correctness formulation: a prediction's *confidence* is the
probability assigned to the predicted answer, and we check whether, among
predictions made at ~90% confidence, roughly 90% are actually right.

``score`` decisions contribute MAE and are excluded from the probabilistic
metrics (their self-reported confidence is not a probability of being
correct, so folding it into ECE would be a category error).
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from .data import ResultRecord
from .schemas import (
    Case,
    ChoiceDecision,
    DecisionValidationError,
    ScoreDecision,
    YesNoDecision,
    parse_decision,
)

_EPS = 1e-15


def _brier(decision: YesNoDecision | ChoiceDecision, case: Case) -> float:
    if isinstance(decision, YesNoDecision):
        target = 1.0 if case.correct else 0.0
        return (decision.probability - target) ** 2
    return sum(
        (p - (1.0 if option == case.correct else 0.0)) ** 2
        for option, p in zip(decision.options, decision.probabilities)
    )


def _log_loss(decision: YesNoDecision | ChoiceDecision, case: Case) -> float:
    if isinstance(decision, YesNoDecision):
        p_true = decision.probability if case.correct else 1.0 - decision.probability
    else:
        p_true = decision.probability_of(case.correct)
    return -math.log(min(max(p_true, _EPS), 1.0))


def _is_correct(decision: YesNoDecision | ChoiceDecision, case: Case) -> bool:
    if isinstance(decision, YesNoDecision):
        return decision.answer == case.correct
    return decision.picked == case.correct


def reliability_bins(
    confidences: list[float], correct: list[bool], n_bins: int = 10
) -> list[dict[str, Any]]:
    """Group predictions into confidence bins with per-bin accuracy."""
    bins: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        members = [
            (c, ok)
            for c, ok in zip(confidences, correct)
            if (lo <= c < hi) or (i == n_bins - 1 and c == 1.0)
        ]
        bins.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "count": len(members),
                "mean_confidence": (
                    sum(c for c, _ in members) / len(members) if members else None
                ),
                "accuracy": (sum(1 for _, ok in members if ok) / len(members))
                if members
                else None,
            }
        )
    return bins


def expected_calibration_error(bins: list[dict[str, Any]], total: int) -> float:
    """ECE: count-weighted mean gap between bin confidence and bin accuracy."""
    if total == 0:
        return 0.0
    ece = 0.0
    for b in bins:
        if b["count"]:
            ece += (b["count"] / total) * abs(b["accuracy"] - b["mean_confidence"])
    return ece


def evaluate(
    cases: Iterable[Case], results: Iterable[ResultRecord], n_bins: int = 10
) -> dict[str, Any]:
    """Score one model's results against a dataset.

    Fails loudly on a result whose ``case_id`` is not in the dataset, and on
    any decision that fails schema validation.  Cases with no result are
    counted in ``n_missing`` and excluded from every denominator.
    """
    case_by_id = {c.id: c for c in cases}
    results = list(results)

    n_correct = 0
    n_categorical = 0
    brier_terms: list[float] = []
    logloss_terms: list[float] = []
    confidences: list[float] = []
    correctness: list[bool] = []
    abs_errors: list[float] = []
    total_cost = 0.0
    any_cost = False
    latencies: list[float] = []
    n_scored = 0

    for record in results:
        case = case_by_id.get(record.case_id)
        if case is None:
            raise DecisionValidationError(
                f"result references unknown case_id {record.case_id!r} (not in the dataset)"
            )
        decision = parse_decision(record.decision, case)
        n_scored += 1
        if record.cost_usd is not None:
            any_cost = True
            total_cost += record.cost_usd
        if record.latency_ms is not None:
            latencies.append(record.latency_ms)

        if isinstance(decision, ScoreDecision):
            abs_errors.append(abs(decision.value - float(case.correct)))
        else:
            n_categorical += 1
            ok = _is_correct(decision, case)
            if ok:
                n_correct += 1
            correctness.append(ok)
            confidences.append(decision.confidence)
            brier_terms.append(_brier(decision, case))
            logloss_terms.append(_log_loss(decision, case))

    bins = reliability_bins(confidences, correctness, n_bins=n_bins)
    total_cases = len(case_by_id)

    return {
        "n_cases": total_cases,
        "n_scored": n_scored,
        "n_missing": total_cases - n_scored,
        "n_categorical": n_categorical,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_categorical) if n_categorical else None,
        "score_mae": (sum(abs_errors) / len(abs_errors)) if abs_errors else None,
        "n_score": len(abs_errors),
        "brier": (sum(brier_terms) / len(brier_terms)) if brier_terms else None,
        "log_loss": (sum(logloss_terms) / len(logloss_terms)) if logloss_terms else None,
        "ece": expected_calibration_error(bins, len(confidences)) if confidences else None,
        "reliability_bins": bins,
        # Cost-per-correct-decision: the cost of ALL scored calls divided by
        # the number of correct yes_no/choice decisions -- the headline
        # "what does one good decision cost me?" number.
        "total_cost_usd": total_cost if any_cost else None,
        "cost_per_correct_usd": (total_cost / n_correct)
        if (any_cost and n_correct > 0)
        else None,
        "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
    }
