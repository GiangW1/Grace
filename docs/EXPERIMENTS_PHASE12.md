# Phase 1/2 experiments

These two entries use the same frozen actor snapshot and keep the workload
fixed. They are intended to run on the server after the PR branch is checked
out; no training update is performed during either measurement.

## 1. Request-level scheduler probe

`scripts/scheduler_probe.py` compares the existing two-call `LLM.generate`
path with a bounded `LLMEngine.add_request`/`step` queue. In `fixed_ht`, every
predetermined start first receives `decision_tokens`; a fixed Bernoulli draw
then either enqueues the suffix or records an early stop. `p=0` is the
prefix-only m=0 baseline. `full_pg` is the full completion arm. Results are
written to `scheduler_trials.jsonl`, `scheduler_records.jsonl`, and
`scheduler_summary.json`.

Example server command:

```bash
python scripts/scheduler_probe.py \
  --config configs/hardware/a100_1.yaml \
  --config configs/experiments/scheduler_probe.yaml \
  --model-path "$MODEL_PATH" \
  --checkpoint "$CHECKPOINT" \
  --data-path "$DATA_PATH" \
  --n-problems 8 --starts-per-problem 16 --capacity 8 \
  --decision-tokens 512 --max-new-tokens 2048 \
  --p 0 --p 0.5 --p 0.75 --p 1 \
  --repeats 3 --mode both --workload both \
  --run-dir runs/phase12-scheduler
```

The main comparison is wall time for the same starts and actor snapshot.
Token counts, queue occupancy, continuation counts, finish reasons, and
request-level timing are saved alongside it. Both schedulers receive the same
preassigned request seeds and Bernoulli decisions; repeats alternate scheduler
order. `scheduler_comparisons.jsonl` reports semantic trajectory agreement and
the paired wall-time ratio. The run deliberately excludes model construction
and adapter setup from the reported scheduling scope.

## 2. Predictor diagnostics

First generate one common frozen audit. Set `store_features: true` so each
prefix stores the exact feature vector used by the predictor, while all
continuation labels and seeds remain those of the existing audit path. The
checkpoint must contain the frozen basis `U`; the supplied pilot config uses
32 problems, 2 paths per problem, and 8 suffixes per prefix:

```bash
python scripts/audit.py --generate \
  --config configs/hardware/a100_1.yaml \
  --config configs/experiments/predictor_diagnostic_audit.yaml \
  --model-path "$MODEL_PATH" \
  --checkpoint "$CHECKPOINT" \
  --data-path "$DATA_PATH" \
  --run-dir runs/phase12-predictor-audit
```

Then fit the candidates on that same `audit_bundles.jsonl`:

```bash
python scripts/diagnose_predictor.py \
  --bundles runs/phase12-predictor-audit \
  --models zero,simple,ridge,mlp \
  --seed 17 --test-fraction 0.3 \
  --run-dir runs/phase12-predictor-diagnostic
```

The split is by `problem_id`, so all prefixes and future suffixes from a
problem stay in one side. The report includes full-space residuals,
per-problem held-out rows, the subspace omission floor, coordinate prediction
error, a per-realized-suffix oracle geometry ceiling, and unbiased same-prefix
future variance. The MLP is fit with fixed hyperparameters; no test-set tuning
or allocation decision is mixed into this diagnostic.
