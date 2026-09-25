# Phase 1/2 experiments

The experiments use a frozen actor snapshot. The serving probe keeps its
workload fixed, while the predictor analyses reuse stored audit labels; none
of them updates the actor during measurement.

The fast prefix-conditioned scalar training-value audit is documented in
[EXPECTED_GAIN_DIRECTIONAL_AUDIT.md](EXPECTED_GAIN_DIRECTIONAL_AUDIT.md).

The later sparse-checkpoint serve experiment, including 512/1024 decisions,
global-barrier comparison, cost calibration, and frozen-predictor timing, is
documented in [TRUNCATION_REFILL_SERVE.md](TRUNCATION_REFILL_SERVE.md).

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

## 3. Shared predictor suite

The suite can run immediately on the existing full
`phase12-predictor-audit-20260922-1740/audit_bundles.jsonl` on the A100
server. The local results archive excluded this large JSONL, so run the
following command against the original server run directory. It averages
coordinate and scalar labels from each prefix's independent suffixes before
fitting, giving a Monte Carlo estimate of the conditional mean rather than
one suffix realization. It reports
same-prefix coordinate, reward, and token-cost variance separately, keeps all
problem IDs intact across train/test splits, and includes shuffled-label Ridge
and MLP controls.

Run the existing-data comparison on CPU (it does not allocate A100 memory):

```bash
python scripts/diagnose_predictor_suite.py \
  --bundles runs/phase12-predictor-audit-20260922-1740 \
  --split-repeats 5 --test-fraction 0.3 \
  --models zero,simple,ridge,mlp \
  --include-shuffled --device cpu \
  --run-dir runs/phase12-predictor-suite-existing
```

The suite also runs two independent CPU analyses on the same audit. Use
separate run directories if launching them concurrently. On the same server,
these fits and repeated JSONL reads can compete with vLLM for CPU time and
disk bandwidth. For clean refill wall-time measurements, run them on a
different host or after the timing trials, or reserve separate CPU cores and
verify the measured latency is unaffected.

```bash
python scripts/diagnose_predictor_suite.py \
  --bundles runs/phase12-predictor-audit-20260922-1740 \
  --split-repeats 3 --models zero,simple,ridge,mlp \
  --feature-sets all,last,middle,pooled,entropy,length_baseline,no_baseline \
  --device cpu --run-dir runs/phase12-predictor-features

python scripts/diagnose_predictor_suite.py \
  --bundles runs/phase12-predictor-audit-20260922-1740 \
  --split-repeats 5 --models zero,simple,ridge,mlp \
  --train-fractions 0.25,0.5,1.0 \
  --device cpu --run-dir runs/phase12-predictor-learning-curve
```

These feature sets only slice the already stored legacy vector. They do not
recompute `prompt`, `response`, or `decision` features; those require an actor
forward and are a separate experiment. Every training fraction uses the same
held-out problems for a given split seed. A shuffled model permutes training
prefix labels but keeps the held-out labels untouched.

After the refill timing experiment finishes, a larger shared audit can be
generated on the A100 for a stronger conditional-mean estimate:

```bash
python scripts/audit.py --generate \
  --config configs/hardware/a100_1.yaml \
  --config configs/experiments/predictor_suite_audit.yaml \
  --model-path "$MODEL_PATH" \
  --checkpoint "$CHECKPOINT" \
  --data-path "$DATA_PATH" \
  --run-dir runs/phase12-predictor-suite-audit
```

Then run the same suite on the larger audit:

```bash
python scripts/diagnose_predictor_suite.py \
  --bundles runs/phase12-predictor-suite-audit \
  --split-repeats 5 --test-fraction 0.3 \
  --models zero,simple,ridge,mlp \
  --include-shuffled --device cpu \
  --run-dir runs/phase12-predictor-suite
```

The suite writes `predictor_suite_summary.json`, per-split rows, and a
future-randomness block. Check `test_prefix_mean_coordinate_error` against
`zero` to assess conditional-mean prediction, then inspect
`test_suffix_residual_mean` and the basis omission floor. The scalar rows
report held-out reward and cost error against the mean of the training
prefixes. These offline fits do not measure wall-time savings or actor
learning quality. The prefix-mean error includes finite-suffix noise; the
summary separately reports a sample-variance-over-suffix-count estimate.
