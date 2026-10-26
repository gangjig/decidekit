"""Composed-action evaluation: spec parsing, composition, costs, CLI."""

import json
from pathlib import Path

import pytest

from decidekit import (
    Case,
    DecisionValidationError,
    ResultRecord,
    build_action_comparison,
    compose_actions,
    load_action_spec,
    parse_action_spec,
    render_action_report,
)
from decidekit.cli import main

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INBOX_DATASET = str(DATA / "sample_inbox_dataset.jsonl")
INBOX_SPEC = str(DATA / "sample_inbox_spec.json")
FIELDSHARP = str(DATA / "sample_inbox_results_fieldsharp-1.jsonl")
ACTIONWISE = str(DATA / "sample_inbox_results_actionwise-1.jsonl")

SPEC = {
    "actions": ["task", "draft", "ignore"],
    "inaction": "ignore",
    "rules": [
        {"when": {"is_noise": True}, "action": "ignore"},
        {"when": {"needs_reply": True, "urgent": True}, "action": "task"},
        {"when": {"needs_reply": True}, "action": "draft"},
        {"action": "ignore"},
    ],
    "costs": {"task->ignore": 10, "draft->ignore": 6, "task->draft": 3},
    "default_error": 1,
}

FIELDS = {"is_noise": False, "needs_reply": True, "urgent": True}  # gold action: task


def _cases(n_items=3, gold=None):
    gold = gold or FIELDS
    cases = []
    for i in range(n_items):
        for field, correct in gold.items():
            cases.append(
                Case(
                    id=f"item-{i}:{field}",
                    kind="yes_no",
                    prompt="?",
                    correct=correct,
                    group=f"item-{i}",
                    field=field,
                )
            )
    return cases


def _records(cases, probs, model="m"):
    """probs: {(item_index, field): probability} overrides; default = correct."""
    records = []
    for case in cases:
        item = int(case.group.split("-")[1])
        p = probs.get((item, case.field), 0.9 if case.correct else 0.1)
        records.append(
            ResultRecord(case_id=case.id, model=model, decision={"probability": p})
        )
    return records


# --- spec parsing -----------------------------------------------------------


def test_parse_spec_ok():
    spec = parse_action_spec(SPEC)
    assert spec.actions == ("task", "draft", "ignore")
    assert spec.inaction == "ignore"
    assert spec.cost_of("task", "ignore") == 10
    assert spec.cost_of("ignore", "task") == 1  # default_error
    assert spec.cost_of("task", "task") == 0


def test_parse_spec_wildcard_costs():
    raw = dict(SPEC, costs={"*->ignore": 7, "task->ignore": 10})
    spec = parse_action_spec(raw)
    assert spec.cost_of("task", "ignore") == 10  # exact beats wildcard
    assert spec.cost_of("draft", "ignore") == 7
    assert spec.cost_of("draft", "task") == 1


def test_parse_spec_rejects_default_rule_not_last():
    raw = json.loads(json.dumps(SPEC))
    raw["rules"] = [{"action": "ignore"}, {"when": {"is_noise": True}, "action": "ignore"}]
    with pytest.raises(DecisionValidationError, match="must be the last rule"):
        parse_action_spec(raw)


def test_parse_spec_rejects_missing_default_rule():
    raw = json.loads(json.dumps(SPEC))
    raw["rules"] = raw["rules"][:-1]
    with pytest.raises(DecisionValidationError, match="default rule"):
        parse_action_spec(raw)


def test_parse_spec_rejects_unknown_action_and_cost_key():
    raw = json.loads(json.dumps(SPEC))
    raw["rules"][0]["action"] = "shred"
    with pytest.raises(DecisionValidationError, match="not one of the actions"):
        parse_action_spec(raw)
    raw = json.loads(json.dumps(SPEC))
    raw["costs"] = {"task->shred": 5}
    with pytest.raises(DecisionValidationError, match="not one of the actions"):
        parse_action_spec(raw)


def test_parse_spec_rejects_bad_threshold():
    raw = json.loads(json.dumps(SPEC))
    raw["thresholds"] = {"needs_reply": 1.5}
    with pytest.raises(DecisionValidationError, match="must be a number in \\(0, 1\\)"):
        parse_action_spec(raw)


# --- composition ------------------------------------------------------------


def test_compose_perfect_model():
    cases = _cases(3)
    report = compose_actions(cases, _records(cases, {}), parse_action_spec(SPEC))
    assert report["action_accuracy"] == 1.0
    assert report["field_accuracy"] == 1.0
    assert report["mean_cost_per_item"] == 0.0
    assert report["false_inaction_rate"] == 0.0
    assert report["confusion"] == {}


def test_compose_false_inaction_is_singled_out():
    # One item's is_noise flips to true: gold task -> predicted ignore.
    cases = _cases(3)
    records = _records(cases, {(1, "is_noise"): 0.95})
    report = compose_actions(cases, records, parse_action_spec(SPEC))
    assert report["confusion"] == {"task->ignore": 1}
    assert report["total_cost"] == 10
    assert report["false_inaction_count"] == 1
    assert report["false_inaction_rate"] == pytest.approx(1 / 3)
    assert report["costliest_errors"][0]["gold"] == "task"


def test_compose_field_error_without_action_change_costs_nothing():
    # urgent flips on one item: gold task -> predicted draft (cheaper error),
    # and field accuracy drops while no inaction happens.
    cases = _cases(2)
    records = _records(cases, {(0, "urgent"): 0.2})
    report = compose_actions(cases, records, parse_action_spec(SPEC))
    assert report["confusion"] == {"task->draft": 1}
    assert report["total_cost"] == 3
    assert report["field_accuracy"] == pytest.approx(5 / 6)
    assert report["false_inaction_count"] == 0


def test_compose_threshold_override_changes_calls():
    cases = _cases(1)
    records = _records(cases, {(0, "needs_reply"): 0.6})
    report = compose_actions(cases, records, parse_action_spec(SPEC))
    assert report["action_accuracy"] == 1.0  # 0.6 >= default 0.5 -> still "task"
    strict = json.loads(json.dumps(SPEC))
    strict["thresholds"] = {"needs_reply": 0.7}
    report = compose_actions(cases, records, parse_action_spec(strict))
    assert report["confusion"] == {"task->ignore": 1}  # now needs_reply reads false


def test_compose_incomplete_group_is_counted_not_scored():
    cases = _cases(2)
    records = [r for r in _records(cases, {}) if r.case_id != "item-1:urgent"]
    report = compose_actions(cases, records, parse_action_spec(SPEC))
    assert report["n_items"] == 1
    assert report["n_incomplete"] == 1


def test_compose_rejects_inconsistent_field_sets():
    cases = _cases(1)
    cases.append(
        Case(id="odd:is_noise", kind="yes_no", prompt="?", correct=True,
             group="odd", field="is_noise")
    )
    records = [
        ResultRecord(case_id=c.id, model="m", decision={"probability": 0.9})
        for c in cases
    ]
    with pytest.raises(DecisionValidationError, match="disagree on their field sets"):
        compose_actions(cases, records, parse_action_spec(SPEC))


def test_compose_rejects_spec_field_the_dataset_lacks():
    raw = json.loads(json.dumps(SPEC))
    raw["rules"].insert(0, {"when": {"is_spam": True}, "action": "ignore"})
    cases = _cases(1)
    with pytest.raises(DecisionValidationError, match="is_spam"):
        compose_actions(cases, _records(cases, {}), parse_action_spec(raw))


def test_compose_rejects_condition_shape_mismatch():
    raw = json.loads(json.dumps(SPEC))
    raw["rules"][0]["when"] = {"is_noise": "yes"}
    cases = _cases(1)
    with pytest.raises(DecisionValidationError, match="must be true/false"):
        compose_actions(cases, _records(cases, {}), parse_action_spec(raw))


def test_compose_unknown_case_id_fails_loudly():
    cases = _cases(1)
    records = _records(cases, {}) + [
        ResultRecord(case_id="nope", model="m", decision={"probability": 0.5})
    ]
    with pytest.raises(DecisionValidationError, match="unknown case_id"):
        compose_actions(cases, records, parse_action_spec(SPEC))


# --- comparison and the bundled demo ----------------------------------------


def test_demo_rankings_diverge_like_the_foyer_study():
    from decidekit import load_cases, load_results

    cases = load_cases(INBOX_DATASET)
    spec = load_action_spec(INBOX_SPEC)
    named = {
        "fieldsharp-1": compose_actions(cases, load_results(FIELDSHARP), spec),
        "actionwise-1": compose_actions(cases, load_results(ACTIONWISE), spec),
    }
    # The Foyer pattern: the field winner commits the expensive false-ignore.
    assert named["fieldsharp-1"]["field_accuracy"] > named["actionwise-1"]["field_accuracy"]
    assert named["fieldsharp-1"]["mean_cost_per_item"] > named["actionwise-1"]["mean_cost_per_item"]
    assert named["fieldsharp-1"]["false_inaction_rate"] == pytest.approx(1 / 3)
    assert named["actionwise-1"]["false_inaction_rate"] == 0.0

    comparison = build_action_comparison(named)
    assert comparison["ranking_by_field_accuracy"] == ["fieldsharp-1", "actionwise-1"]
    assert comparison["ranking_by_action_cost"] == ["actionwise-1", "fieldsharp-1"]
    text = render_action_report(named, comparison)
    assert "rankings diverge" in text
    assert "gold task -> did ignore" in text


def test_cli_actions_json(capsys):
    rc = main([
        "actions", "--dataset", INBOX_DATASET, "--spec", INBOX_SPEC,
        "--results", FIELDSHARP, ACTIONWISE, "--json",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["comparison"]["ranking_by_action_cost"][0] == "actionwise-1"
    assert out["models"]["fieldsharp-1"]["confusion"] == {"task->ignore": 1}


def test_cli_actions_text(capsys):
    rc = main([
        "actions", "--dataset", INBOX_DATASET, "--spec", INBOX_SPEC,
        "--results", FIELDSHARP, ACTIONWISE,
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Composed-action report" in out
    assert "false-inaction" in out
