"""Loading datasets (JSONL of cases) and model results (JSONL of records).

Dataset file: one :class:`~decidekit.schemas.Case` per line::

    {"id": "yn-1", "kind": "yes_no", "prompt": "...", "correct": true}

Results file: one record per line::

    {"case_id": "yn-1", "model": "my-model", "decision": {"probability": 0.9},
     "cost_usd": 0.004, "latency_ms": 820}

``cost_usd`` and ``latency_ms`` are optional but, when present, must be
non-negative numbers.  A results file describes ONE model: duplicate
``case_id`` entries are an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import Case, DecisionValidationError


@dataclass(frozen=True)
class ResultRecord:
    case_id: str
    model: str | None
    decision: Any  # raw, unvalidated -- use schemas.parse_decision against the case
    cost_usd: float | None = None
    latency_ms: float | None = None


def _read_jsonl(path: str | Path) -> list[tuple[int, Any]]:
    rows: list[tuple[int, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append((lineno, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise DecisionValidationError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def load_cases(path: str | Path) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    for lineno, raw in _read_jsonl(path):
        try:
            case = Case.from_dict(raw)
        except DecisionValidationError as exc:
            raise DecisionValidationError(f"{path}:{lineno}: {exc}") from exc
        if case.id in seen:
            raise DecisionValidationError(f"{path}:{lineno}: duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise DecisionValidationError(f"{path}: dataset is empty")
    return cases


def load_results(path: str | Path) -> list[ResultRecord]:
    records: list[ResultRecord] = []
    seen: set[str] = set()
    for lineno, raw in _read_jsonl(path):
        where = f"{path}:{lineno}"
        if not isinstance(raw, dict):
            raise DecisionValidationError(f"{where}: result record must be a JSON object")
        case_id = raw.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise DecisionValidationError(f"{where}: missing non-empty string 'case_id'")
        if case_id in seen:
            raise DecisionValidationError(f"{where}: duplicate result for case {case_id!r}")
        seen.add(case_id)
        if "decision" not in raw:
            raise DecisionValidationError(f"{where}: missing 'decision'")
        model = raw.get("model")
        if model is not None and not isinstance(model, str):
            raise DecisionValidationError(f"{where}: 'model' must be a string")
        records.append(
            ResultRecord(
                case_id=case_id,
                model=model,
                decision=raw["decision"],
                cost_usd=_optional_non_negative(raw, "cost_usd", where),
                latency_ms=_optional_non_negative(raw, "latency_ms", where),
            )
        )
    if not records:
        raise DecisionValidationError(f"{path}: results file is empty")
    return records


def _optional_non_negative(raw: dict, key: str, where: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionValidationError(f"{where}: {key!r} must be a number, got {value!r}")
    number = float(value)
    if number < 0:
        raise DecisionValidationError(f"{where}: {key!r} must be non-negative, got {number}")
    return number


def model_name_for(records: list[ResultRecord], path: str | Path, override: str | None) -> str:
    """Resolve the display name for a results file.

    Priority: explicit override, then the records' shared ``model`` field,
    then the file name.
    """
    if override:
        return override
    names = {r.model for r in records if r.model}
    if len(names) == 1:
        return next(iter(names))
    if len(names) > 1:
        raise DecisionValidationError(
            f"{path}: results file mixes model names {sorted(names)}; "
            "one file must contain one model (or pass a name override)"
        )
    return Path(path).stem
