# decidekit

**Evaluate decision models the way decisions actually get used: are they right, do their probabilities mean what they say, and what does one correct decision cost?**

`decidekit` is a small, zero-dependency Python toolkit for evaluating models whose job is to emit a *typed decision* — a yes/no call with a probability, a choice among options with a probability distribution, or a numeric score with a confidence. It is built in the spirit of TypeSafe AI's [Jev](https://www.typesafe.ai/) decision model and the Artificial Analysis *Decision Index*: once a model's output is a decision rather than prose, accuracy alone is not enough — you need calibration and cost accounting too.

## Why per-token benchmarks mislead for decisions

Model leaderboards usually quote a price per million tokens and a generic benchmark score. For decision workloads both numbers mislead:

- **You don't buy tokens, you buy decisions.** A cheap model that needs a retry, or that is wrong 30% of the time, can cost more *per correct decision* than an expensive model that is right the first time. decidekit's headline cost metric is exactly that: total spend ÷ correct decisions.
- **A probability you can't trust is worse than no probability.** Downstream systems threshold, escalate, and auto-approve based on the model's stated confidence. If a model says "99% sure" and is right 80% of the time, every one of those automations is mispriced. Brier score and ECE measure that gap directly.
- **Latency is part of the decision cost.** An on-call or moderation pipeline cares about the mean latency per decision, so it is a first-class column, not an afterthought.

## Quickstart

No installation is needed to try the demo — the sample dataset and three fake model result files are bundled, and everything runs offline:

```bash
pip install -e ".[test]"     # or just PYTHONPATH=. — there are no dependencies

# Score one model
decidekit score --dataset data/sample_dataset.jsonl \
                --results data/sample_results_careful-1.jsonl

# Compare all three fake models
decidekit compare --dataset data/sample_dataset.jsonl \
                  --results data/sample_results_careful-1.jsonl \
                            data/sample_results_loud-1.jsonl \
                            data/sample_results_budget-1.jsonl

# Check a results file parses against the schemas
decidekit validate --dataset data/sample_dataset.jsonl \
                   --results data/sample_results_loud-1.jsonl
```

The comparison prints a Markdown table:

| Model | Accuracy | Brier | Log loss | ECE | Score MAE | $ / correct | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| careful-1 | 1.000 | … | … | … | … | … | … |
| loud-1 | 0.800 | … | … | … | … | … | … |
| budget-1 | 0.700 | … | … | … | … | … | … |

(Run it to see the real numbers.) The demo is designed so each metric tells part of the story: `loud-1` gets most calls right but states ~99% confidence even when wrong, so its Brier score and ECE expose it; `budget-1` is the least accurate yet by far the cheapest per correct decision — which model "wins" depends on what a wrong decision costs *you*.

Every command also accepts `--json` for machine-readable output.

## Data formats

**Dataset** — JSONL, one case per line:

```json
{"id": "yn-1", "kind": "yes_no", "prompt": "Does this post violate the threats policy?", "correct": true}
{"id": "ch-1", "kind": "choice", "prompt": "Pick an action", "options": ["allow", "flag", "remove"], "correct": "remove"}
{"id": "sc-1", "kind": "score", "prompt": "Severity 0-10?", "score_min": 0, "score_max": 10, "correct": 8}
```

**Results** — JSONL, one model's answer per line. Raw model output, exactly as the model emitted it:

```json
{"case_id": "yn-1", "model": "my-model", "decision": {"probability": 0.93}, "cost_usd": 0.004, "latency_ms": 820}
{"case_id": "ch-1", "model": "my-model", "decision": {"picked": "remove", "probabilities": {"allow": 0.05, "flag": 0.1, "remove": 0.85}}}
{"case_id": "sc-1", "model": "my-model", "decision": {"value": 7, "confidence": 0.8}}
```

`cost_usd` and `latency_ms` are optional. `probabilities` may be an object keyed by option or a list aligned with the case's `options`.

Parsing is **strict on purpose**: a picked choice that is not among the options, a probability outside [0, 1], a probability vector that doesn't sum to ~1, or a score outside its range all fail loudly with a validation error. Silently repairing malformed model output would corrupt the very numbers you're evaluating.

## The metrics, briefly

- **Accuracy** — fraction of `yes_no` (threshold 0.5) and `choice` decisions that match the known-correct answer.
- **Score MAE** — mean absolute error over `score` cases.
- **Brier score** — mean squared gap between the predicted probability distribution and the actual outcome. 0 is perfect; it punishes confident wrongness quadratically, so one 0.99-and-wrong answer hurts more than ten mild misses.
- **Log loss** — like Brier, but punishes confident wrongness *logarithmically* — even harder. A model that says 0.99 and is wrong pays ~4.6; at 0.999 it pays ~6.9.
- **ECE (Expected Calibration Error)** — plain language: *"When the model says it's X% sure, is it right about X% of the time?"* Predictions are grouped into confidence bins; ECE is the average gap between each bin's stated confidence and its actual accuracy, weighted by bin size. 0 means the probabilities mean exactly what they say. The full reliability-bin table is included in `score --json` output.
- **Cost per correct decision** — total cost of all scored calls divided by the number of correct `yes_no`/`choice` decisions. (Score cases are excluded from the denominator — "correct" isn't defined for a continuous score; their cost is included in the numerator.)
- **Mean latency** — average `latency_ms` over scored calls, when provided.

## Library use

```python
from decidekit import load_cases, load_results, evaluate, build_comparison, render_markdown

cases = load_cases("data/sample_dataset.jsonl")
metrics = evaluate(cases, load_results("my_results.jsonl"))
print(metrics["accuracy"], metrics["ece"], metrics["cost_per_correct_usd"])
```

## Honest limitations

- **The bundled dataset is a demo, not a benchmark.** 13 hand-written content-moderation cases and three invented models exist so the CLI runs offline and the metric story is visible. Drawing conclusions about real models requires a dataset that is larger, representative of your traffic, and labeled with care.
- **Calibration needs volume.** ECE over a handful of cases is mostly noise; treat it as meaningful only with hundreds of predictions per bin range you care about.
- **Score-case confidence is not calibrated here.** A score's self-reported confidence is not a probability of being correct, so score cases contribute MAE only — folding them into ECE would be a category error.
- **Ground truth is assumed correct.** decidekit measures agreement with your labels; if the labels are wrong or contested, the metrics faithfully measure the wrong thing.
- **One results file = one model**, and every result must reference a known case. Missing cases are excluded and counted (`n_missing`); unknown case IDs fail loudly.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
python -m pytest        # 47 tests, fully offline
```

Requires Python ≥ 3.10. The package itself has **zero runtime dependencies** — the evaluation math is plain Python on purpose, so it can drop into any pipeline.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Gang Ji.
