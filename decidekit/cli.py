"""Command-line interface: score, compare, validate.

Exit codes: 0 = success, 2 = validation error (loud failure, message on
stderr), 1 = usage error (argparse).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import load_cases, load_results, model_name_for
from .metrics import evaluate
from .report import build_comparison, render_markdown
from .schemas import DecisionValidationError, parse_decision


def _fmt(value, digits=3, prefix=""):
    return "—" if value is None else f"{prefix}{value:.{digits}f}"


def _cmd_score(args: argparse.Namespace) -> int:
    cases = load_cases(args.dataset)
    records = load_results(args.results)
    name = model_name_for(records, args.results, args.model)
    metrics = evaluate(cases, records)
    if args.json:
        print(json.dumps({"model": name, **metrics}, indent=2))
        return 0
    print(f"Model: {name}")
    print(f"Cases scored: {metrics['n_scored']}/{metrics['n_cases']}"
          f" (missing: {metrics['n_missing']})")
    print(f"Accuracy (yes_no/choice): {_fmt(metrics['accuracy'])}"
          f"  ({metrics['n_correct']}/{metrics['n_categorical']} correct)")
    print(f"Brier score:  {_fmt(metrics['brier'])}   (lower is better; 0 = perfect)")
    print(f"Log loss:     {_fmt(metrics['log_loss'])}")
    print(f"ECE:          {_fmt(metrics['ece'])}   (calibration gap; 0 = perfectly calibrated)")
    print(f"Score MAE:    {_fmt(metrics['score_mae'])}   over {metrics['n_score']} score case(s)")
    print(f"Total cost:   {_fmt(metrics['total_cost_usd'], 4, '$')}")
    print(f"Cost/correct: {_fmt(metrics['cost_per_correct_usd'], 4, '$')}")
    print(f"Mean latency: {_fmt(metrics['mean_latency_ms'], 0)} ms")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    cases = load_cases(args.dataset)
    named: dict[str, dict] = {}
    for results_path in args.results:
        records = load_results(results_path)
        name = model_name_for(records, results_path, None)
        if name in named:
            raise DecisionValidationError(
                f"duplicate model name {name!r} across results files; "
                "set the 'model' field in each record to disambiguate"
            )
        named[name] = evaluate(cases, records)
    comparison = build_comparison(named)
    if args.json:
        print(json.dumps(comparison, indent=2))
    else:
        print(render_markdown(comparison), end="")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    cases = load_cases(args.dataset)
    records = load_results(args.results)
    case_by_id = {c.id: c for c in cases}
    errors: list[str] = []
    for record in records:
        case = case_by_id.get(record.case_id)
        if case is None:
            errors.append(f"unknown case_id {record.case_id!r} (not in the dataset)")
            continue
        try:
            parse_decision(record.decision, case)
        except DecisionValidationError as exc:
            errors.append(str(exc))
    if errors:
        for error in errors:
            print(f"INVALID: {error}", file=sys.stderr)
        print(f"{args.results}: {len(errors)} problem(s) in {len(records)} record(s)",
              file=sys.stderr)
        return 2
    print(f"OK: {len(records)} decision(s) in {args.results} parse cleanly "
          f"against {args.dataset}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decidekit",
        description="Evaluate decision models: accuracy, calibration, and cost per correct decision.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="Score one model's results against a dataset")
    p_score.add_argument("--dataset", required=True, help="Dataset JSONL file")
    p_score.add_argument("--results", required=True, help="Model results JSONL file")
    p_score.add_argument("--model", default=None, help="Override the model name")
    p_score.add_argument("--json", action="store_true", help="Emit metrics as JSON")
    p_score.set_defaults(func=_cmd_score)

    p_compare = sub.add_parser("compare", help="Compare 2+ models, print a Markdown ranking table")
    p_compare.add_argument("--dataset", required=True, help="Dataset JSONL file")
    p_compare.add_argument("--results", required=True, nargs="+",
                           help="Two or more results JSONL files, one per model")
    p_compare.add_argument("--json", action="store_true", help="Emit the comparison as JSON")
    p_compare.set_defaults(func=_cmd_compare)

    p_validate = sub.add_parser("validate", help="Check a results file parses against the schemas")
    p_validate.add_argument("--dataset", required=True, help="Dataset JSONL file")
    p_validate.add_argument("--results", required=True, help="Model results JSONL file")
    p_validate.set_defaults(func=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DecisionValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
