import math

import pytest

from decidekit.data import ResultRecord
from decidekit.metrics import evaluate
from decidekit.schemas import Case, DecisionValidationError


def rec(case_id, decision, cost=None, latency=None):
    return ResultRecord(
        case_id=case_id, model="m", decision=decision, cost_usd=cost, latency_ms=latency
    )


def yn(case_id, correct):
    return Case(id=case_id, kind="yes_no", prompt="?", correct=correct)


# -- hand-calculated Brier / log loss ----------------------------------------


def test_brier_yes_no_hand_calculated():
    # p = 0.8, correct is yes -> (0.8 - 1)^2 = 0.04
    m = evaluate([yn("a", True)], [rec("a", {"probability": 0.8})])
    assert m["brier"] == pytest.approx(0.04)
    assert m["accuracy"] == 1.0


def test_brier_yes_no_wrong_side():
    # p = 0.8 but correct is no -> (0.8 - 0)^2 = 0.64, and the call is wrong
    m = evaluate([yn("a", False)], [rec("a", {"probability": 0.8})])
    assert m["brier"] == pytest.approx(0.64)
    assert m["accuracy"] == 0.0


def test_log_loss_hand_calculated():
    m = evaluate([yn("a", True)], [rec("a", {"probability": 0.8})])
    assert m["log_loss"] == pytest.approx(-math.log(0.8))


def test_brier_choice_hand_calculated():
    # probs [0.7, 0.2, 0.1], correct is first option:
    # (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.09 + 0.04 + 0.01 = 0.14
    case = Case(id="c", kind="choice", prompt="?", correct="x", options=("x", "y", "z"))
    m = evaluate([case], [rec("c", {"picked": "x", "probabilities": [0.7, 0.2, 0.1]})])
    assert m["brier"] == pytest.approx(0.14)
    assert m["log_loss"] == pytest.approx(-math.log(0.7))


# -- hand-calculated ECE -------------------------------------------------------
#
# Four yes_no predictions, 10 bins:
#   p=0.6 correct=yes -> conf 0.6, right     (bin 0.6-0.7)
#   p=0.4 correct=yes -> conf 0.6, wrong     (bin 0.6-0.7: acc 1/2, conf 0.6)
#   p=0.9 correct=yes -> conf 0.9, right     (bin 0.9-1.0)
#   p=0.1 correct=no  -> conf 0.9, right     (bin 0.9-1.0: acc 1, conf 0.9)
# ECE = (2/4)*|0.5-0.6| + (2/4)*|1.0-0.9| = 0.05 + 0.05 = 0.10


def test_ece_hand_calculated():
    cases = [yn("a", True), yn("b", True), yn("c", True), yn("d", False)]
    results = [
        rec("a", {"probability": 0.6}),
        rec("b", {"probability": 0.4}),
        rec("c", {"probability": 0.9}),
        rec("d", {"probability": 0.1}),
    ]
    m = evaluate(cases, results)
    assert m["ece"] == pytest.approx(0.10)
    bins = {round(b["bin_lo"], 1): b for b in m["reliability_bins"]}
    assert bins[0.6]["count"] == 2
    assert bins[0.6]["accuracy"] == pytest.approx(0.5)
    assert bins[0.9]["count"] == 2
    assert bins[0.9]["accuracy"] == pytest.approx(1.0)


def test_ece_zero_for_perfect_calibration_at_extremes():
    cases = [yn("a", True), yn("b", False)]
    results = [rec("a", {"probability": 1.0}), rec("b", {"probability": 0.0})]
    m = evaluate(cases, results)
    assert m["ece"] == pytest.approx(0.0)
    assert m["brier"] == pytest.approx(0.0)


# -- accuracy / MAE ------------------------------------------------------------


def test_accuracy_threshold_boundary():
    # 0.5 counts as "yes"
    m = evaluate([yn("a", True)], [rec("a", {"probability": 0.5})])
    assert m["accuracy"] == 1.0


def test_score_mae():
    case = Case(id="s", kind="score", prompt="?", correct=5, score_min=0, score_max=10)
    case2 = Case(id="t", kind="score", prompt="?", correct=8, score_min=0, score_max=10)
    m = evaluate(
        [case, case2],
        [rec("s", {"value": 7, "confidence": 0.9}), rec("t", {"value": 6, "confidence": 0.5})],
    )
    assert m["score_mae"] == pytest.approx(2.0)  # (|7-5| + |6-8|) / 2
    assert m["accuracy"] is None  # no categorical decisions


# -- cost / latency --------------------------------------------------------------


def test_cost_per_correct_decision():
    cases = [yn("a", True), yn("b", True), yn("c", False), yn("d", True)]
    results = [
        rec("a", {"probability": 0.9}, cost=0.10),
        rec("b", {"probability": 0.9}, cost=0.10),
        rec("c", {"probability": 0.1}, cost=0.10),
        rec("d", {"probability": 0.1}, cost=0.10),  # wrong
    ]
    m = evaluate(cases, results)
    assert m["total_cost_usd"] == pytest.approx(0.40)
    assert m["n_correct"] == 3
    assert m["cost_per_correct_usd"] == pytest.approx(0.40 / 3)


def test_cost_per_correct_none_when_no_correct():
    m = evaluate([yn("a", True)], [rec("a", {"probability": 0.1}, cost=0.5)])
    assert m["cost_per_correct_usd"] is None


def test_mean_latency():
    cases = [yn("a", True), yn("b", False)]
    results = [
        rec("a", {"probability": 0.9}, latency=100),
        rec("b", {"probability": 0.2}, latency=300),
    ]
    m = evaluate(cases, results)
    assert m["mean_latency_ms"] == pytest.approx(200.0)


# -- dataset bookkeeping ---------------------------------------------------------


def test_missing_cases_counted_and_excluded():
    cases = [yn("a", True), yn("b", False)]
    m = evaluate(cases, [rec("a", {"probability": 0.9})])
    assert m["n_missing"] == 1
    assert m["n_scored"] == 1
    assert m["accuracy"] == 1.0


def test_unknown_case_id_fails_loudly():
    with pytest.raises(DecisionValidationError):
        evaluate([yn("a", True)], [rec("zzz", {"probability": 0.9})])


def test_invalid_decision_fails_loudly():
    with pytest.raises(DecisionValidationError):
        evaluate([yn("a", True)], [rec("a", {"probability": 7})])
