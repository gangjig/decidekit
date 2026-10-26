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

# Tune the yes/no threshold from the saved scores (no model calls)
decidekit tune --dataset data/sample_dataset.jsonl \
               --results data/sample_results_budget-1.jsonl

# Compose field decisions into actions and score what the actions cost
decidekit actions --dataset data/sample_inbox_dataset.jsonl \
                  --spec data/sample_inbox_spec.json \
                  --results data/sample_inbox_results_fieldsharp-1.jsonl \
                            data/sample_inbox_results_actionwise-1.jsonl
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

A case may also carry optional `group` and `field` strings — several cases that are fields of one real-world item (one email's `is_noise` / `needs_reply` / `urgent`) share a `group`, and `field` names each case's role. Only [composed-action evaluation](#composed-actions-score-the-decision-the-user-actually-feels) uses them; the other commands ignore them.

## The metrics, briefly

- **Accuracy** — fraction of `yes_no` (threshold 0.5) and `choice` decisions that match the known-correct answer.
- **Score MAE** — mean absolute error over `score` cases.
- **Brier score** — mean squared gap between the predicted probability distribution and the actual outcome. 0 is perfect; it punishes confident wrongness quadratically, so one 0.99-and-wrong answer hurts more than ten mild misses.
- **Log loss** — like Brier, but punishes confident wrongness *logarithmically* — even harder. A model that says 0.99 and is wrong pays ~4.6; at 0.999 it pays ~6.9.
- **ECE (Expected Calibration Error)** — plain language: *"When the model says it's X% sure, is it right about X% of the time?"* Predictions are grouped into confidence bins; ECE is the average gap between each bin's stated confidence and its actual accuracy, weighted by bin size. 0 means the probabilities mean exactly what they say. The full reliability-bin table is included in `score --json` output.
- **Cost per correct decision** — total cost of all scored calls divided by the number of correct `yes_no`/`choice` decisions. (Score cases are excluded from the denominator — "correct" isn't defined for a continuous score; their cost is included in the numerator.)
- **Mean latency** — average `latency_ms` over scored calls, when provided.

## Threshold tuning

Every `yes_no` metric above thresholds the model's probability at **0.50** — a default nobody chose, and often the wrong one. A model that scores real violations at 0.45 gets them all wrong at 0.50 no matter how good its ranking is. `decidekit tune` finds the threshold your saved scores actually support:

```bash
decidekit tune --dataset data/sample_dataset.jsonl \
               --results data/sample_results_budget-1.jsonl \
               --min-precision 0.8 --min-recall 0.8
```

```
           threshold  precision   recall      F1  accuracy     $/100  $/correct
default        0.500      1.000    0.667   0.800     0.833   $0.0800    $0.0010
tuned          0.450      1.000    1.000   1.000     1.000   $0.0800    $0.0008
```

How it works:

- **Saved scores only.** Tuning reuses the probabilities already in the results file and never calls a model — so the cost per 100 eval runs is identical at both thresholds; only decision quality moves. (Cost *per correct decision* does move: the denominator improves.)
- **Stratified cross-validation.** Candidate thresholds (the observed score values) are selected on each fold's training part — best F1 among those meeting your precision/recall floors — and scored on the held-out fold. The report shows both the full-data tuned row and the out-of-fold means, so you can see the honest generalisation estimate next to the optimistic one.
- **Floors are reported, not faked.** If no threshold meets both minimums, the best-F1 threshold is still shown, flagged as not meeting the floors.
- `yes_no` cases only: a `choice` distribution has no single threshold to tune, and `score` cases are excluded for the same reason score confidence is excluded from ECE.

This feature was motivated by Hamel Husain's [decision-model evals write-up](https://hamel.dev/blog/posts/decision-model-evals/), which found the same effect at larger scale: default thresholds understating models, and uncalibrated scores being trusted as probabilities.

## Composed actions: score the decision the user actually feels

Per-field metrics score each decision on its own — but production systems don't ship a field, they ship an **action** composed from several fields. An inbox triage agent drafts a reply, files a task, or ignores the mail, based on its `is_noise` / `needs_reply` / `urgent` calls. The field ranking and the action ranking can disagree, and the disagreement is where the money goes: in the Foyer inbox-agent study below, the system with the best per-field accuracy, composed into actions, most often committed the most expensive error — ignoring mail that needed action.

`decidekit actions` evaluates that layer. Cases that are fields of one item share a `group` id in the dataset (with a `field` name, or an `id` ending in `:<field>`), and an **action spec** (JSON) says how field values compose into actions, and what each kind of action error costs:

```json
{
  "actions": ["task", "draft", "ignore"],
  "inaction": "ignore",
  "rules": [
    {"when": {"is_noise": true}, "action": "ignore"},
    {"when": {"needs_reply": true, "urgent": true}, "action": "task"},
    {"when": {"needs_reply": true}, "action": "draft"},
    {"action": "ignore"}
  ],
  "costs": {"task->ignore": 10, "draft->ignore": 6, "task->draft": 3},
  "default_error": 1
}
```

Rules match first-to-last; the last rule has no `when` and is the default. Conditions are `true`/`false` for `yes_no` fields, an option name (or list) for `choice` fields, and `{"gte"/"lte": n}` for `score` fields. Cost keys are `"gold->pred"`; either side may be `"*"`, and a correct action always costs 0. The gold action is the spec applied to the known-correct field values; the predicted action is the spec applied to the model's saved decisions — still no model calls.

The bundled inbox demo reproduces the divergence with two invented models over six emails:

```
model            field acc field rank action acc cost/item action rank false-inaction
actionwise-1         0.833          2      0.667     0.833           1           0.0%
fieldsharp-1         0.944          1      0.833     1.667           2          33.3%

Ranked by field accuracy: fieldsharp-1 > actionwise-1
Ranked by action cost:    actionwise-1 > fieldsharp-1
```

`fieldsharp-1` wins every field ranking and loses where it counts: its single field miss (calling one actionable email "noise") becomes a `task -> ignore` error at cost 10, and a third of the mail that needed action gets ignored. The report shows, per model: field accuracy next to **action accuracy**, **mean cost per item**, each model's rank under both orderings, the **false-inaction rate** (of the items whose gold action was not the `inaction` action, how often the system did nothing — the expensive failure mode), and the costliest gold→pred confusions. Field conditions must be consistent across groups — a group missing a field the others have fails loudly rather than being quietly dropped, and a group the model only partially answered is counted as incomplete and excluded.

This feature was motivated by Federico Lucca's [“Measure Before You Train”](https://arxiv.org/abs/2610.32955) (the Foyer study), which measured exactly this on a real personal inbox agent: the most field-accurate system ignored 18% of actionable mail once its fields were composed into actions, vs 4% for the best LLM — and a cost-chosen threshold on one field halved that rate.

## Library use

```python
from decidekit import load_cases, load_results, evaluate, build_comparison, render_markdown, tune_threshold

cases = load_cases("data/sample_dataset.jsonl")
metrics = evaluate(cases, load_results("my_results.jsonl"))
print(metrics["accuracy"], metrics["ece"], metrics["cost_per_correct_usd"])

report = tune_threshold(cases, load_results("my_results.jsonl"),
                        min_precision=0.8, min_recall=0.8)
print(report["tuned"]["threshold"], report["cross_validation"]["mean_f1"])
```

## Honest limitations

- **The bundled dataset is a demo, not a benchmark.** 13 hand-written content-moderation cases and three invented models exist so the CLI runs offline and the metric story is visible. Drawing conclusions about real models requires a dataset that is larger, representative of your traffic, and labeled with care.
- **Calibration needs volume.** ECE over a handful of cases is mostly noise; treat it as meaningful only with hundreds of predictions per bin range you care about.
- **A tuned threshold is still an estimate.** Cross-validation keeps the tuning honest on small data, but a threshold tuned on last month's labels can drift with your traffic; re-tune on fresh labels and sanity-check on held-out data before shipping it to production.
- **Score-case confidence is not calibrated here.** A score's self-reported confidence is not a probability of being correct, so score cases contribute MAE only — folding them into ECE would be a category error.
- **Ground truth is assumed correct.** decidekit measures agreement with your labels; if the labels are wrong or contested, the metrics faithfully measure the wrong thing.
- **One results file = one model**, and every result must reference a known case. Missing cases are excluded and counted (`n_missing`); unknown case IDs fail loudly.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
python -m pytest        # 63 tests, fully offline
```

Requires Python ≥ 3.10. The package itself has **zero runtime dependencies** — the evaluation math is plain Python on purpose, so it can drop into any pipeline.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Gang Ji.
