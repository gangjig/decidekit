"""decidekit -- evaluate decision models on accuracy, calibration, and cost.

In the spirit of TypeSafe AI's Jev and the Artificial Analysis Decision
Index: when a model's whole job is to emit a *decision* (a typed answer
with probabilities), per-token benchmark prices tell you almost nothing.
What matters is whether the decisions are right, whether the stated
probabilities mean what they say, and what each correct decision costs.
"""

from .data import ResultRecord, load_cases, load_results
from .metrics import evaluate, expected_calibration_error, reliability_bins
from .report import build_comparison, render_markdown
from .schemas import (
    Case,
    ChoiceDecision,
    Decision,
    DecisionValidationError,
    ScoreDecision,
    YesNoDecision,
    parse_decision,
)

__version__ = "0.1.0"

__all__ = [
    "Case",
    "ChoiceDecision",
    "Decision",
    "DecisionValidationError",
    "ResultRecord",
    "ScoreDecision",
    "YesNoDecision",
    "build_comparison",
    "evaluate",
    "expected_calibration_error",
    "load_cases",
    "load_results",
    "parse_decision",
    "reliability_bins",
    "render_markdown",
    "__version__",
]
