"""Composed-action evaluation: score the action the user actually feels.

Per-field metrics (accuracy, Brier, ECE, threshold tuning) score each
typed decision on its own.  Production systems rarely ship a field --
they ship an *action* composed from several fields: an inbox agent
drafts a reply, files a task, or ignores the mail, based on its
``is_noise`` / ``needs_reply`` / ``urgent`` calls.  Lucca's Foyer study
(arXiv:2610.32955, "Measure Before You Train") measured why the two
views diverge: the system with the best per-field accuracy, composed
into actions, most often committed the most expensive error -- ignoring
mail that needed action (18% of actionable mail, vs 4% for the best
LLM) -- because field rankings say nothing about which mistakes the
composition turns into costly actions.

This module evaluates the composed layer.  The caller supplies an
*action spec* (JSON): the set of actions, an ordered rule list mapping
field values to an action (first match wins, last rule is the default),
optional per-field thresholds for ``yes_no`` fields, and a cost table
for action errors.  The gold action for an item is the spec applied to
the known-correct field values; the predicted action is the spec
applied to the model's decisions.  No model is ever called: like
threshold tuning, this reuses saved results.

Grouping: cases that are fields of one item share a ``group`` id in the
dataset, and each carries a ``field`` name (or an ``id`` ending in
``:<field>``).  Every group must have the same field set -- a group
missing a field an otherwise-identical group has is a dataset bug, and
silently skipping it would bend the numbers.  A group whose *results*
are incomplete (the model never answered one field) is counted in
``n_incomplete`` and excluded: that is a property of the run, not of
the dataset.

Costs: ``costs`` maps ``"gold->pred"`` to a non-negative number; either
side may be ``"*"`` (exact pairs win over wildcards).  A correct action
always costs 0.  Any other error costs ``default_error`` (default 1.0)
when no entry matches.  ``inaction`` names the do-nothing action (e.g.
``"ignore"``); the report singles out *false inaction* -- gold asked
for action, the system did nothing -- because in the Foyer study (and
in most triage systems) that is the expensive failure mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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

DEFAULT_YES_NO_THRESHOLD = 0.5


def _fail(message: str) -> None:
    raise DecisionValidationError(message)


# ---------------------------------------------------------------------------
# Action spec: actions, ordered rules, thresholds, costs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    when: tuple[tuple[str, Any], ...]  # (field, condition); empty = default rule
    action: str


@dataclass(frozen=True)
class ActionSpec:
    actions: tuple[str, ...]
    rules: tuple[Rule, ...]
    inaction: str | None = None
    thresholds: tuple[tuple[str, float], ...] = ()
    costs: tuple[tuple[str, str, float], ...] = ()  # (gold, pred, cost); "*" wildcards
    default_error: float = 1.0

    def threshold_for(self, field: str) -> float:
        for name, value in self.thresholds:
            if name == field:
                return value
        return DEFAULT_YES_NO_THRESHOLD

    def cost_of(self, gold: str, pred: str) -> float:
        """Cost of taking action ``pred`` when the right action was ``gold``."""
        if gold == pred:
            return 0.0
        best: tuple[int, float] | None = None
        for g, p, cost in self.costs:
            specificity = (1 if g != "*" else 0) + (1 if p != "*" else 0)
            if (g == "*" or g == gold) and (p == "*" or p == pred):
                if best is None or specificity > best[0]:
                    best = (specificity, cost)
        return best[1] if best is not None else self.default_error


def parse_action_spec(raw: Any) -> ActionSpec:
    """Parse and validate an action spec dict.  Fails loudly on anything off."""
    if not isinstance(raw, dict):
        _fail(f"action spec must be a JSON object, got {type(raw).__name__}")

    actions_raw = raw.get("actions")
    if (
        not isinstance(actions_raw, list)
        or not actions_raw
        or not all(isinstance(a, str) and a for a in actions_raw)
    ):
        _fail("action spec: 'actions' must be a non-empty list of strings")
    if len(set(actions_raw)) != len(actions_raw):
        _fail(f"action spec: duplicate entries in 'actions': {actions_raw}")
    actions = tuple(actions_raw)

    inaction = raw.get("inaction")
    if inaction is not None and inaction not in actions:
        _fail(f"action spec: 'inaction' {inaction!r} is not one of the actions {list(actions)}")

    rules_raw = raw.get("rules")
    if not isinstance(rules_raw, list) or not rules_raw:
        _fail("action spec: 'rules' must be a non-empty list")
    rules: list[Rule] = []
    for i, raw_rule in enumerate(rules_raw):
        where = f"action spec rule {i}"
        if not isinstance(raw_rule, dict):
            _fail(f"{where}: rule must be a JSON object")
        action = raw_rule.get("action")
        if action not in actions:
            _fail(f"{where}: 'action' {action!r} is not one of the actions {list(actions)}")
        when_raw = raw_rule.get("when")
        if when_raw is None:
            if i != len(rules_raw) - 1:
                _fail(
                    f"{where}: a rule without 'when' is the default rule and "
                    "must be the last rule"
                )
            when: tuple[tuple[str, Any], ...] = ()
        else:
            if not isinstance(when_raw, dict) or not when_raw:
                _fail(f"{where}: 'when' must be a non-empty object of field -> condition")
            for field_name, condition in when_raw.items():
                if not isinstance(field_name, str) or not field_name:
                    _fail(f"{where}: 'when' keys must be non-empty field names")
                _check_condition_shape(where, field_name, condition)
            when = tuple(sorted(when_raw.items()))
        rules.append(Rule(when=when, action=action))
    if rules[-1].when:
        _fail(
            "action spec: the last rule must be the default rule (no 'when'), "
            "so every field combination maps to an action"
        )

    thresholds: list[tuple[str, float]] = []
    thresholds_raw = raw.get("thresholds", {})
    if not isinstance(thresholds_raw, dict):
        _fail("action spec: 'thresholds' must be an object of field -> number")
    for field_name, value in thresholds_raw.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) < 1.0
        ):
            _fail(
                f"action spec: threshold for field {field_name!r} must be a "
                f"number in (0, 1), got {value!r}"
            )
        thresholds.append((field_name, float(value)))

    default_error = raw.get("default_error", 1.0)
    if (
        isinstance(default_error, bool)
        or not isinstance(default_error, (int, float))
        or float(default_error) < 0
    ):
        _fail(f"action spec: 'default_error' must be a non-negative number, got {default_error!r}")

    costs: list[tuple[str, str, float]] = []
    costs_raw = raw.get("costs", {})
    if not isinstance(costs_raw, dict):
        _fail("action spec: 'costs' must be an object of \"gold->pred\" -> number")
    for key, value in costs_raw.items():
        parts = key.split("->")
        if len(parts) != 2:
            _fail(f"action spec: cost key {key!r} must look like \"gold->pred\"")
        gold, pred = parts
        for side in (gold, pred):
            if side != "*" and side not in actions:
                _fail(
                    f"action spec: cost key {key!r} names {side!r}, which is "
                    f"not one of the actions {list(actions)} (or \"*\")"
                )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) < 0
        ):
            _fail(f"action spec: cost {key!r} must be a non-negative number, got {value!r}")
        costs.append((gold, pred, float(value)))

    referenced = {field for rule in rules for field, _ in rule.when}
    referenced |= {field for field, _ in thresholds}
    return ActionSpec(
        actions=actions,
        rules=tuple(rules),
        inaction=inaction,
        thresholds=tuple(sorted(thresholds)),
        costs=tuple(costs),
        default_error=float(default_error),
    )


def _check_condition_shape(where: str, field: str, condition: Any) -> None:
    """A condition is a bool (yes_no), a string or list of strings
    (choice), or an object with gte/lte numbers (score).  Kind-specific
    checks happen when the spec meets the dataset's fields."""
    if isinstance(condition, bool):
        return
    if isinstance(condition, str) and condition:
        return
    if isinstance(condition, list) and condition and all(
        isinstance(c, str) and c for c in condition
    ):
        return
    if isinstance(condition, dict) and condition and set(condition) <= {"gte", "lte"}:
        for bound, value in condition.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                _fail(f"{where}: score condition {bound!r} for field {field!r} must be a number")
        return
    _fail(
        f"{where}: condition for field {field!r} must be true/false (yes_no), "
        "an option name or list of names (choice), or {\"gte\"/\"lte\": number} "
        f"(score); got {condition!r}"
    )


def load_action_spec(path: str | Path) -> ActionSpec:
    """Load an action spec JSON file."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DecisionValidationError(f"{path}: invalid JSON: {exc}") from exc
    try:
        return parse_action_spec(raw)
    except DecisionValidationError as exc:
        raise DecisionValidationError(f"{path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Composition.
# ---------------------------------------------------------------------------


def _condition_matches(condition: Any, value: Any) -> bool:
    if isinstance(condition, bool):
        return value is condition
    if isinstance(condition, str):
        return value == condition
    if isinstance(condition, list):
        return value in condition
    # score range condition
    if "gte" in condition and not value >= condition["gte"]:
        return False
    if "lte" in condition and not value <= condition["lte"]:
        return False
    return True


def _apply_rules(spec: ActionSpec, values: dict[str, Any]) -> str:
    for rule in spec.rules:
        if all(_condition_matches(cond, values[field]) for field, cond in rule.when):
            return rule.action
    raise AssertionError("unreachable: the default rule matches everything")


def _gold_value(case: Case) -> Any:
    if case.kind == "yes_no":
        return bool(case.correct)
    if case.kind == "choice":
        return case.correct
    return float(case.correct)


def _predicted_value(case: Case, decision: Any, spec: ActionSpec) -> Any:
    if isinstance(decision, YesNoDecision):
        return decision.probability >= spec.threshold_for(case.field_name or "")
    if isinstance(decision, ChoiceDecision):
        return decision.picked
    assert isinstance(decision, ScoreDecision)
    return decision.value


def _field_is_correct(case: Case, decision: Any, spec: ActionSpec) -> bool:
    if isinstance(decision, YesNoDecision):
        return _predicted_value(case, decision, spec) is bool(case.correct)
    if isinstance(decision, ChoiceDecision):
        return decision.picked == case.correct
    assert isinstance(decision, ScoreDecision)
    return decision.value == float(case.correct)


def compose_actions(
    cases: Iterable[Case],
    results: Iterable[ResultRecord],
    spec: ActionSpec,
) -> dict[str, Any]:
    """Compose one model's field decisions into actions and score them.

    Fails loudly on an unknown ``case_id`` or a malformed decision (same
    policy as :func:`decidekit.metrics.evaluate`), on groups with
    inconsistent field sets, and on spec rules that reference fields or
    condition shapes the dataset does not have.
    """
    cases = list(cases)
    case_by_id = {c.id: c for c in cases}

    groups: dict[str, dict[str, Case]] = {}
    n_ungrouped = 0
    for case in cases:
        if case.group is None:
            n_ungrouped += 1
            continue
        field = case.field_name
        assert field is not None
        group = groups.setdefault(case.group, {})
        if field in group:
            _fail(
                f"group {case.group!r}: two cases claim field {field!r} "
                f"({group[field].id!r} and {case.id!r})"
            )
        group[field] = case
    if not groups:
        _fail(
            "composed-action evaluation needs grouped cases; no case in the "
            "dataset has a 'group' (see decidekit.actions)"
        )
    field_sets = {tuple(sorted(g)) for g in groups.values()}
    if len(field_sets) != 1:
        _fail(
            "groups disagree on their field sets: "
            + "; ".join(
                f"{sorted(fs)}" for fs in sorted(field_sets)
            )
            + " -- every group must carry the same fields"
        )
    fields = field_sets.pop()

    _validate_spec_against_fields(spec, groups, fields)

    decisions: dict[str, Any] = {}
    for record in results:
        case = case_by_id.get(record.case_id)
        if case is None:
            _fail(
                f"result references unknown case_id {record.case_id!r} "
                "(not in the dataset)"
            )
        decisions[record.case_id] = parse_decision(record.decision, case)

    confusion: dict[tuple[str, str], int] = {}
    per_field: dict[str, list[int]] = {f: [0, 0] for f in fields}  # correct, total
    total_cost = 0.0
    n_items = 0
    n_action_correct = 0
    n_incomplete = 0
    n_actionable = 0
    n_false_inaction = 0

    for group_id in sorted(groups):
        group = groups[group_id]
        if any(case.id not in decisions for case in group.values()):
            n_incomplete += 1
            continue
        gold_values = {f: _gold_value(group[f]) for f in fields}
        pred_values = {
            f: _predicted_value(group[f], decisions[group[f].id], spec) for f in fields
        }
        gold_action = _apply_rules(spec, gold_values)
        pred_action = _apply_rules(spec, pred_values)
        n_items += 1
        if gold_action == pred_action:
            n_action_correct += 1
        confusion[(gold_action, pred_action)] = (
            confusion.get((gold_action, pred_action), 0) + 1
        )
        total_cost += spec.cost_of(gold_action, pred_action)
        if spec.inaction is not None and gold_action != spec.inaction:
            n_actionable += 1
            if pred_action == spec.inaction:
                n_false_inaction += 1
        for f in fields:
            per_field[f][1] += 1
            if _field_is_correct(group[f], decisions[group[f].id], spec):
                per_field[f][0] += 1

    if n_items == 0:
        _fail("no complete groups to score: the results file covers no full group")

    field_correct = sum(c for c, _ in per_field.values())
    field_total = sum(t for _, t in per_field.values())
    costliest = sorted(
        (
            {
                "gold": gold,
                "pred": pred,
                "count": count,
                "cost_each": spec.cost_of(gold, pred),
                "total_cost": spec.cost_of(gold, pred) * count,
            }
            for (gold, pred), count in confusion.items()
            if gold != pred
        ),
        key=lambda e: (-e["total_cost"], e["gold"], e["pred"]),
    )
    return {
        "n_items": n_items,
        "n_groups": len(groups),
        "n_incomplete": n_incomplete,
        "n_ungrouped_cases": n_ungrouped,
        "fields": list(fields),
        "field_accuracy": (field_correct / field_total) if field_total else None,
        "per_field": {
            f: {"n": t, "accuracy": (c / t) if t else None}
            for f, (c, t) in sorted(per_field.items())
        },
        "action_accuracy": n_action_correct / n_items,
        "total_cost": total_cost,
        "mean_cost_per_item": total_cost / n_items,
        "confusion": {
            f"{gold}->{pred}": count
            for (gold, pred), count in sorted(confusion.items())
            if gold != pred
        },
        "costliest_errors": costliest,
        "inaction": spec.inaction,
        "n_actionable": n_actionable if spec.inaction is not None else None,
        "false_inaction_count": n_false_inaction if spec.inaction is not None else None,
        "false_inaction_rate": (n_false_inaction / n_actionable)
        if (spec.inaction is not None and n_actionable)
        else None,
    }


def _validate_spec_against_fields(
    spec: ActionSpec, groups: dict[str, dict[str, Case]], fields: tuple[str, ...]
) -> None:
    sample_group = next(iter(groups.values()))
    kind_of = {f: sample_group[f].kind for f in fields}
    for rule in spec.rules:
        for field, condition in rule.when:
            if field not in kind_of:
                _fail(
                    f"action spec rule references field {field!r}, which no "
                    f"group has (fields are {list(fields)})"
                )
            kind = kind_of[field]
            if kind == "yes_no" and not isinstance(condition, bool):
                _fail(
                    f"action spec: field {field!r} is yes_no, so its condition "
                    f"must be true/false, got {condition!r}"
                )
            if kind == "choice":
                if not (isinstance(condition, str) or isinstance(condition, list)):
                    _fail(
                        f"action spec: field {field!r} is choice, so its "
                        f"condition must be an option name or list, got {condition!r}"
                    )
                options = sample_group[field].options or ()
                wanted = [condition] if isinstance(condition, str) else condition
                for option in wanted:
                    if option not in options:
                        _fail(
                            f"action spec: condition names option {option!r} for "
                            f"field {field!r}, whose options are {list(options)}"
                        )
            if kind == "score" and not isinstance(condition, dict):
                _fail(
                    f"action spec: field {field!r} is score, so its condition "
                    f"must be {{\"gte\"/\"lte\": number}}, got {condition!r}"
                )
    for field, _ in spec.thresholds:
        if field not in kind_of:
            _fail(f"action spec: threshold names unknown field {field!r}")
        if kind_of[field] != "yes_no":
            _fail(
                f"action spec: thresholds apply to yes_no fields only; "
                f"field {field!r} is {kind_of[field]}"
            )


# ---------------------------------------------------------------------------
# Comparison across models: field ranking vs action ranking, side by side.
# ---------------------------------------------------------------------------


def build_action_comparison(
    named_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Rank models by composed-action cost, keeping the field ranking visible.

    Rows sort by mean cost per item (ascending -- the action-level winner
    first), then total cost, then name.  Each row also carries
    ``field_rank`` and ``action_rank`` (1-based) so a divergence between
    "best fields" and "cheapest actions" is a number, not a vibe.
    """
    rows = []
    for model, report in named_reports.items():
        rows.append(
            {
                "model": model,
                "field_accuracy": report["field_accuracy"],
                "action_accuracy": report["action_accuracy"],
                "mean_cost_per_item": report["mean_cost_per_item"],
                "total_cost": report["total_cost"],
                "false_inaction_rate": report["false_inaction_rate"],
                "n_items": report["n_items"],
            }
        )
    by_field = sorted(
        rows,
        key=lambda r: (
            -(r["field_accuracy"] if r["field_accuracy"] is not None else -1.0),
            r["model"],
        ),
    )
    for rank, row in enumerate(by_field, start=1):
        row["field_rank"] = rank
    rows.sort(key=lambda r: (r["mean_cost_per_item"], r["total_cost"], r["model"]))
    for rank, row in enumerate(rows, start=1):
        row["action_rank"] = rank
    return {
        "rows": rows,
        "ranking_by_action_cost": [r["model"] for r in rows],
        "ranking_by_field_accuracy": [r["model"] for r in by_field],
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_action_report(
    named_reports: dict[str, dict[str, Any]], comparison: dict[str, Any]
) -> str:
    """Render the side-by-side field-vs-action comparison as plain text."""
    lines: list[str] = []
    n_items = next(iter(named_reports.values()))["n_items"]
    lines.append(
        f"Composed-action report over {n_items} item(s) "
        "(gold action = spec applied to the correct field values)"
    )
    lines.append("")
    header = (
        f"{'model':<16}{'field acc':>10}{'field rank':>11}"
        f"{'action acc':>11}{'cost/item':>10}{'action rank':>12}"
        f"{'false-inaction':>15}"
    )
    lines.append(header)
    for row in comparison["rows"]:
        fi = (
            "—"
            if row["false_inaction_rate"] is None
            else f"{row['false_inaction_rate']:.1%}"
        )
        lines.append(
            f"{row['model']:<16}{_fmt(row['field_accuracy']):>10}"
            f"{row['field_rank']:>11}{_fmt(row['action_accuracy']):>11}"
            f"{row['mean_cost_per_item']:>10.3f}{row['action_rank']:>12}{fi:>15}"
        )
    lines.append("")
    field_order = comparison["ranking_by_field_accuracy"]
    action_order = comparison["ranking_by_action_cost"]
    lines.append("Ranked by field accuracy: " + " > ".join(field_order))
    lines.append("Ranked by action cost:    " + " > ".join(action_order))
    if field_order != action_order:
        lines.append(
            "The rankings diverge: the most field-accurate model is not the "
            "one whose composed actions cost least."
        )
    for model in action_order:
        errors = named_reports[model]["costliest_errors"][:3]
        if not errors:
            continue
        lines.append("")
        lines.append(f"Costliest action errors -- {model}:")
        for error in errors:
            lines.append(
                f"  gold {error['gold']} -> did {error['pred']}: "
                f"{error['count']}x @ {error['cost_each']:.1f} = {error['total_cost']:.1f}"
            )
    return "\n".join(lines) + "\n"
