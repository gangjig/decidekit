import pytest

from decidekit.schemas import (
    Case,
    ChoiceDecision,
    DecisionValidationError,
    ScoreDecision,
    YesNoDecision,
    parse_decision,
)


def yn_case(correct=True):
    return Case(id="c", kind="yes_no", prompt="?", correct=correct)


def choice_case():
    return Case(
        id="c",
        kind="choice",
        prompt="?",
        correct="flag",
        options=("allow", "flag", "remove"),
    )


def score_case():
    return Case(id="c", kind="score", prompt="?", correct=5, score_min=0, score_max=10)


# -- yes_no -----------------------------------------------------------------


def test_parse_yes_no_ok():
    d = parse_decision({"probability": 0.8}, yn_case())
    assert isinstance(d, YesNoDecision)
    assert d.probability == 0.8
    assert d.answer is True
    assert d.confidence == pytest.approx(0.8)


def test_parse_yes_no_from_json_string():
    d = parse_decision('{"probability": 0.3}', yn_case(correct=False))
    assert d.answer is False


def test_parse_yes_no_probability_above_one_fails():
    with pytest.raises(DecisionValidationError):
        parse_decision({"probability": 1.5}, yn_case())


def test_parse_yes_no_probability_below_zero_fails():
    with pytest.raises(DecisionValidationError):
        parse_decision({"probability": -0.1}, yn_case())


def test_parse_yes_no_probability_wrong_type_fails():
    with pytest.raises(DecisionValidationError):
        parse_decision({"probability": "high"}, yn_case())


def test_parse_yes_no_missing_probability_fails():
    with pytest.raises(DecisionValidationError):
        parse_decision({"answer": True}, yn_case())


def test_parse_invalid_json_string_fails():
    with pytest.raises(DecisionValidationError):
        parse_decision("{not json", yn_case())


def test_parse_non_object_fails():
    with pytest.raises(DecisionValidationError):
        parse_decision([0.8], yn_case())


# -- choice -----------------------------------------------------------------


def test_parse_choice_ok_dict_probabilities():
    d = parse_decision(
        {"picked": "flag", "probabilities": {"allow": 0.1, "flag": 0.7, "remove": 0.2}},
        choice_case(),
    )
    assert isinstance(d, ChoiceDecision)
    assert d.picked == "flag"
    assert d.probability_of("flag") == pytest.approx(0.7)
    assert d.confidence == pytest.approx(0.7)


def test_parse_choice_ok_list_probabilities():
    d = parse_decision({"picked": "allow", "probabilities": [0.34, 0.33, 0.33]}, choice_case())
    assert d.probabilities == pytest.approx((0.34, 0.33, 0.33))


def test_parse_choice_picked_not_in_options_fails():
    with pytest.raises(DecisionValidationError):
        parse_decision(
            {"picked": "nuke", "probabilities": {"allow": 0.1, "flag": 0.1, "remove": 0.8}},
            choice_case(),
        )


def test_parse_choice_probabilities_not_summing_to_one_fails():
    with pytest.raises(DecisionValidationError):
        parse_decision(
            {"picked": "flag", "probabilities": {"allow": 0.1, "flag": 0.2, "remove": 0.2}},
            choice_case(),
        )


def test_parse_choice_probability_out_of_range_fails():
    with pytest.raises(DecisionValidationError):
        parse_decision(
            {"picked": "flag", "probabilities": {"allow": -0.5, "flag": 1.0, "remove": 0.5}},
            choice_case(),
        )


def test_parse_choice_missing_option_key_fails():
    with pytest.raises(DecisionValidationError):
        parse_decision(
            {"picked": "flag", "probabilities": {"allow": 0.5, "flag": 0.5}},
            choice_case(),
        )


def test_parse_choice_wrong_length_list_fails():
    with pytest.raises(DecisionValidationError):
        parse_decision({"picked": "flag", "probabilities": [0.5, 0.5]}, choice_case())


# -- score ------------------------------------------------------------------


def test_parse_score_ok():
    d = parse_decision({"value": 7, "confidence": 0.9}, score_case())
    assert isinstance(d, ScoreDecision)
    assert d.value == 7.0


def test_parse_score_value_out_of_range_fails():
    with pytest.raises(DecisionValidationError):
        parse_decision({"value": 11, "confidence": 0.9}, score_case())


def test_parse_score_confidence_out_of_range_fails():
    with pytest.raises(DecisionValidationError):
        parse_decision({"value": 5, "confidence": 1.2}, score_case())


# -- case validation ----------------------------------------------------------


def test_case_choice_correct_must_be_an_option():
    with pytest.raises(DecisionValidationError):
        Case.from_dict(
            {"id": "x", "kind": "choice", "prompt": "?", "options": ["a", "b"], "correct": "c"}
        )


def test_case_yes_no_correct_must_be_bool():
    with pytest.raises(DecisionValidationError):
        Case.from_dict({"id": "x", "kind": "yes_no", "prompt": "?", "correct": "yes"})


def test_case_score_correct_out_of_range_fails():
    with pytest.raises(DecisionValidationError):
        Case.from_dict(
            {"id": "x", "kind": "score", "prompt": "?", "score_min": 0, "score_max": 5, "correct": 9}
        )


def test_case_unknown_kind_fails():
    with pytest.raises(DecisionValidationError):
        Case.from_dict({"id": "x", "kind": "essay", "prompt": "?", "correct": "?"})
