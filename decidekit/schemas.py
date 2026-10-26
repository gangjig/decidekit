"""Typed decision schemas and strict parsing of raw model output.

A *decision* is what a model returns when asked a typed question:

- ``yes_no``  -- a probability that the answer is "yes"
- ``choice``  -- one picked option plus a probability distribution over options
- ``score``   -- a numeric value inside a known range, plus a confidence

Raw model output is untrusted.  Every parser in this module fails loudly
(:class:`DecisionValidationError`) on malformed output instead of guessing:
a choice that is not among the options, a probability outside [0, 1], or a
probability vector that does not sum to ~1 is a bug in the model pipeline,
and silently repairing it would make the evaluation numbers meaningless.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Union

#: How far a choice probability vector may drift from summing to exactly 1.0.
#: Models that emit rounded numbers (0.34 / 0.33 / 0.33) stay well inside this.
PROB_SUM_TOLERANCE = 0.01

KINDS = ("yes_no", "choice", "score")


class DecisionValidationError(ValueError):
    """Raised when a case or a raw model decision fails validation."""


def _fail(message: str) -> None:
    raise DecisionValidationError(message)


def _check_probability(value: Any, what: str = "probability") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{what} must be a number, got {type(value).__name__}: {value!r}")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        _fail(f"{what} must be finite, got {number}")
    if not 0.0 <= number <= 1.0:
        _fail(f"{what} must be in [0, 1], got {number}")
    return number


def _check_number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{what} must be a number, got {type(value).__name__}: {value!r}")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        _fail(f"{what} must be finite, got {number}")
    return number


# ---------------------------------------------------------------------------
# Cases: the questions, with known-correct answers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One typed decision question with a known-correct answer.

    ``correct`` is a ``bool`` for ``yes_no``, one of ``options`` for
    ``choice``, and a number inside ``[score_min, score_max]`` for ``score``.

    ``group`` / ``field`` are optional and only used by composed-action
    evaluation (see :mod:`decidekit.actions`): several cases that are
    fields of one real-world item (one email's ``is_noise`` /
    ``needs_reply`` / ``urgent`` ...) share a ``group`` id, and ``field``
    names the case's role inside the group.  When ``field`` is omitted,
    the part of ``id`` after the last ``":"`` is used as the field name.
    """

    id: str
    kind: str
    prompt: str
    correct: Any
    options: tuple[str, ...] | None = None
    score_min: float | None = None
    score_max: float | None = None
    context: str | None = None
    group: str | None = None
    field: str | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> "Case":
        if not isinstance(raw, dict):
            _fail(f"case must be a JSON object, got {type(raw).__name__}")
        case_id = raw.get("id")
        if not isinstance(case_id, str) or not case_id:
            _fail(f"case is missing a non-empty string 'id': {raw!r}")
        kind = raw.get("kind")
        if kind not in KINDS:
            _fail(f"case {case_id!r}: kind must be one of {KINDS}, got {kind!r}")
        prompt = raw.get("prompt")
        if not isinstance(prompt, str):
            _fail(f"case {case_id!r}: 'prompt' must be a string")
        if "correct" not in raw:
            _fail(f"case {case_id!r}: missing 'correct'")
        correct = raw["correct"]
        options: tuple[str, ...] | None = None
        score_min = score_max = None

        if kind == "yes_no":
            if not isinstance(correct, bool):
                _fail(f"case {case_id!r}: yes_no 'correct' must be true/false, got {correct!r}")
        elif kind == "choice":
            raw_options = raw.get("options")
            if (
                not isinstance(raw_options, list)
                or len(raw_options) < 2
                or not all(isinstance(o, str) for o in raw_options)
            ):
                _fail(f"case {case_id!r}: choice cases need 'options', a list of >= 2 strings")
            if len(set(raw_options)) != len(raw_options):
                _fail(f"case {case_id!r}: duplicate entries in 'options'")
            options = tuple(raw_options)
            if correct not in options:
                _fail(f"case {case_id!r}: 'correct' ({correct!r}) is not one of the options")
        else:  # score
            score_min = _check_number(raw.get("score_min"), f"case {case_id!r} 'score_min'")
            score_max = _check_number(raw.get("score_max"), f"case {case_id!r} 'score_max'")
            if not score_min < score_max:
                _fail(f"case {case_id!r}: score_min must be < score_max")
            correct = _check_number(correct, f"case {case_id!r} 'correct'")
            if not score_min <= correct <= score_max:
                _fail(f"case {case_id!r}: 'correct' {correct} outside [{score_min}, {score_max}]")

        context = raw.get("context")
        if context is not None and not isinstance(context, str):
            _fail(f"case {case_id!r}: 'context' must be a string when present")
        group = raw.get("group")
        if group is not None and (not isinstance(group, str) or not group):
            _fail(f"case {case_id!r}: 'group' must be a non-empty string when present")
        field = raw.get("field")
        if field is not None and (not isinstance(field, str) or not field):
            _fail(f"case {case_id!r}: 'field' must be a non-empty string when present")
        return cls(
            id=case_id,
            kind=kind,
            prompt=prompt,
            correct=correct,
            options=options,
            score_min=score_min,
            score_max=score_max,
            context=context,
            group=group,
            field=field,
        )

    @property
    def field_name(self) -> str | None:
        """The case's field name inside its group, or None when ungrouped."""
        if self.group is None:
            return None
        if self.field:
            return self.field
        return self.id.rsplit(":", 1)[-1]


# ---------------------------------------------------------------------------
# Decisions: validated model answers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YesNoDecision:
    """A probability that the correct answer is "yes"."""

    probability: float

    @property
    def answer(self) -> bool:
        """The hard yes/no call, thresholded at 0.5."""
        return self.probability >= 0.5

    @property
    def confidence(self) -> float:
        """Probability assigned to the predicted answer."""
        return max(self.probability, 1.0 - self.probability)


@dataclass(frozen=True)
class ChoiceDecision:
    """A picked option and a probability distribution over all options."""

    options: tuple[str, ...]
    picked: str
    probabilities: tuple[float, ...]

    def probability_of(self, option: str) -> float:
        return self.probabilities[self.options.index(option)]

    @property
    def confidence(self) -> float:
        return self.probability_of(self.picked)


@dataclass(frozen=True)
class ScoreDecision:
    """A numeric value in ``[minimum, maximum]`` plus self-reported confidence."""

    value: float
    minimum: float
    maximum: float
    confidence: float


Decision = Union[YesNoDecision, ChoiceDecision, ScoreDecision]


def parse_decision(raw: Any, case: Case) -> Decision:
    """Parse and validate raw model output for ``case`` into a typed decision.

    ``raw`` may be a dict or a JSON string.  Raises
    :class:`DecisionValidationError` on anything malformed.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            _fail(f"case {case.id!r}: decision is not valid JSON: {exc}")
    if not isinstance(raw, dict):
        _fail(f"case {case.id!r}: decision must be a JSON object, got {type(raw).__name__}")

    if case.kind == "yes_no":
        if "probability" not in raw:
            _fail(f"case {case.id!r}: yes_no decision needs 'probability'")
        return YesNoDecision(probability=_check_probability(raw["probability"]))

    if case.kind == "choice":
        assert case.options is not None
        picked = raw.get("picked")
        if picked not in case.options:
            _fail(
                f"case {case.id!r}: picked choice {picked!r} is not one of "
                f"the options {list(case.options)}"
            )
        if "probabilities" not in raw:
            _fail(f"case {case.id!r}: choice decision needs 'probabilities'")
        probabilities = _parse_probability_vector(raw["probabilities"], case)
        return ChoiceDecision(options=case.options, picked=picked, probabilities=probabilities)

    # score
    assert case.score_min is not None and case.score_max is not None
    if "value" not in raw:
        _fail(f"case {case.id!r}: score decision needs 'value'")
    if "confidence" not in raw:
        _fail(f"case {case.id!r}: score decision needs 'confidence'")
    value = _check_number(raw["value"], f"case {case.id!r} score 'value'")
    if not case.score_min <= value <= case.score_max:
        _fail(
            f"case {case.id!r}: score value {value} outside "
            f"[{case.score_min}, {case.score_max}]"
        )
    return ScoreDecision(
        value=value,
        minimum=case.score_min,
        maximum=case.score_max,
        confidence=_check_probability(raw["confidence"], f"case {case.id!r} 'confidence'"),
    )


def _parse_probability_vector(raw_probs: Any, case: Case) -> tuple[float, ...]:
    assert case.options is not None
    options = case.options
    if isinstance(raw_probs, dict):
        missing = [o for o in options if o not in raw_probs]
        extra = [k for k in raw_probs if k not in options]
        if missing or extra:
            _fail(
                f"case {case.id!r}: probability keys must match the options; "
                f"missing={missing}, unexpected={extra}"
            )
        values = [raw_probs[o] for o in options]
    elif isinstance(raw_probs, list):
        if len(raw_probs) != len(options):
            _fail(
                f"case {case.id!r}: probability list has {len(raw_probs)} entries "
                f"for {len(options)} options"
            )
        values = list(raw_probs)
    else:
        _fail(
            f"case {case.id!r}: 'probabilities' must be an object keyed by option "
            f"or a list aligned with the options, got {type(raw_probs).__name__}"
        )
    probabilities = tuple(
        _check_probability(v, f"case {case.id!r} probability for option") for v in values
    )
    total = sum(probabilities)
    if abs(total - 1.0) > PROB_SUM_TOLERANCE:
        _fail(
            f"case {case.id!r}: probabilities sum to {total:.4f}, "
            f"not ~1.0 (tolerance {PROB_SUM_TOLERANCE})"
        )
    return probabilities
