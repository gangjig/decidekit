"""CLI smoke tests on the bundled sample dataset -- fully offline."""

import json
from pathlib import Path

from decidekit.cli import main

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATASET = str(DATA / "sample_dataset.jsonl")
CAREFUL = str(DATA / "sample_results_careful-1.jsonl")
LOUD = str(DATA / "sample_results_loud-1.jsonl")
BUDGET = str(DATA / "sample_results_budget-1.jsonl")


def test_score_sample_model(capsys):
    rc = main(["score", "--dataset", DATASET, "--results", CAREFUL])
    out = capsys.readouterr().out
    assert rc == 0
    assert "careful-1" in out
    assert "Accuracy" in out


def test_score_json_output(capsys):
    rc = main(["score", "--dataset", DATASET, "--results", BUDGET, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["model"] == "budget-1"
    # budget-1 gets 7 of 10 categorical decisions right
    assert payload["accuracy"] == 0.7
    assert payload["n_cases"] == 13
    # ...but is by far the cheapest per correct decision
    assert payload["cost_per_correct_usd"] < 0.002


def test_compare_ranking_and_story(capsys):
    rc = main(
        ["compare", "--dataset", DATASET, "--results", CAREFUL, LOUD, BUDGET]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("| Model |")
    # careful-1 (accuracy 1.0) first, loud-1 (0.8) second, budget-1 (0.7) third
    assert out.index("careful-1") < out.index("loud-1") < out.index("budget-1")


def test_validate_good_file(capsys):
    rc = main(["validate", "--dataset", DATASET, "--results", CAREFUL])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_validate_bad_file_fails(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"case_id": "yn-threat", "model": "x", "decision": {"probability": 1.7}}\n',
        encoding="utf-8",
    )
    rc = main(["validate", "--dataset", DATASET, "--results", str(bad)])
    assert rc == 2
    assert "INVALID" in capsys.readouterr().err


def test_score_unknown_case_fails(tmp_path, capsys):
    bad = tmp_path / "unknown.jsonl"
    bad.write_text(
        '{"case_id": "nope", "model": "x", "decision": {"probability": 0.5}}\n',
        encoding="utf-8",
    )
    rc = main(["score", "--dataset", DATASET, "--results", str(bad)])
    assert rc == 2
