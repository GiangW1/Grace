# Dynamic score-gradient experiments

This experiment sequence isolates the next hypothesis without changing the
online GRACE estimator.

Stage A asks whether the exact prefix score gradient can represent the
conditional gradient mean.  For each live prefix, the replay is randomly split
into independent halves.  The coefficient of the prefix score-gradient basis
is fitted on half A and evaluated on half B.  The report includes the
per-prefix oracle, a zero predictor, the full-space second-moment residual,
and the conditional-mean reconstruction error.  A positive result is a
representation result only; it does not establish a learned predictor or a
training gain.

Stage B keeps the score-gradient basis exact and learns only its coefficients
from prefix features.  Coefficients are still fitted on half A, while the
feature model is split by problem and evaluated against half B.  The report
selects models using the validation residual and leaves the diagnostic
problems untouched.  This is the first coefficient-learning test for the
dynamic form; the value-head cross term can be added after this stage has
evidence.

When `--store-half-means` is enabled, the replay records reward, baseline, and
advantage means for both halves.  Stage A and Stage B report the observed
half-A reward strata `q_hat=0`, `0<q_hat<1`, and `q_hat=1`; these are finite-sample
strata, not claims that the latent success probability is exactly zero or one.
Stage A also compares the reward-only oracle
`(q_hat-b) * g_h` with the freely fitted score-gradient coefficient and reports
the cross-fitted residual gap and orthogonal energy fraction.  These reports
are the mechanism audit that must precede any value-head or online experiment.

Every replay also has a fixed diagonal quadratic metric.  With no metric file
the metric is Euclidean.  To run an optimizer-aware audit, create the metric
weights from an independent calibration split before replay, record the
pre-registered name, and pass the same file to replay and both CPU reports:

```bash
python scripts/replay_expected_gain.py ... \
  --store-half-means --metric-file runs/calibration/adam_weights.npy \
  --metric-name adam_diagonal --run-dir runs/dynamic-replay
```

The replay stores metric-weighted second moments, so the reported residual
includes continuation noise rather than only the squared error between two
sample means.  A metric file is a diagonal quadratic form; Adam and Fisher
variants must be generated from data independent of the A/B continuation
labels.

The unweighted `half_mean_norm_sq_*` sidecars are retained as the primary
Euclidean result.  When a replay also contains a weighted metric, pass an
explicit all-ones `.npy` file as `--metric-file` to the oracle or predictor to
request the Euclidean report; omit the option to use the replay's recorded
metric.

The replay can replace the source-record baseline with an independently
calibrated problem baseline.  The JSON must contain the checkpoint hash and a
`problems` mapping, for example:

```json
{
  "checkpoint_sha256": "...",
  "problems": {"problem-001": {"baseline": 0.5}}
}
```

The calibration completions must use a separate seed and problem pool from
the A/B continuations.  Pass the file before replay so the gradient labels and
all baseline-dependent features use the same calibrated value:

```bash
python scripts/replay_expected_gain.py ... \
  --baseline-file runs/calibration/problem_baselines.json \
  --store-half-means --run-dir runs/dynamic-replay-calibrated
```

The GPU replay must save the independent halves:

```bash
python scripts/replay_expected_gain.py \
  --bundles runs/phase12-predictor-audit \
  --checkpoint "$CHECKPOINT" --model-path "$MODEL_PATH" \
  --store-half-means --run-dir runs/dynamic-replay
```

Extract the score-gradient basis with the same frozen checkpoint and replay
rows:

```bash
python scripts/extract_prefix_score_gradients.py \
  --replay-dir runs/dynamic-replay \
  --checkpoint "$CHECKPOINT" --model-path "$MODEL_PATH" \
  --run-dir runs/dynamic-score-gradients
```

Run the CPU-only oracle and coefficient predictor.  The `--score-gradients`
path is the `score_gradients.npy` file in the extraction run directory.

```bash
python scripts/expected_gain_dynamic_oracle.py \
  --replay-dir runs/dynamic-replay \
  --score-gradients runs/dynamic-score-gradients/score_gradients.npy \
  --metric-file runs/dynamic-replay/metric_weights.npy \
  --run-dir runs/dynamic-oracle

python scripts/expected_gain_dynamic_predictor.py \
  --replay-dir runs/dynamic-replay \
  --score-gradients runs/dynamic-score-gradients/score_gradients.npy \
  --metric-file runs/dynamic-replay/metric_weights.npy \
  --features legacy --models zero,constant,ridge \
  --run-dir runs/dynamic-predictor
```

For an existing fixed-direction audit, add the exact prefix score gradients to
measure how much of the scalar label is the shared term
`-0.5 * <g_h, d>`:

```bash
python scripts/expected_gain_directional_audit.py \
  --replay-dir runs/directional-replay \
  --direction file --direction-file runs/direction.npy \
  --prefix-score-gradients runs/dynamic-score-gradients/score_gradients.npy \
  --run-dir runs/directional-overlap
```

The code records the score-basis hash, actor/checkpoint identity, replay row
hash, split, and run configuration.  It rejects missing A/B sidecars, row or
layout mismatches, non-finite arrays, and overlapping problem splits.  No
minimum sample count or automatic go/no-go threshold is imposed; the observed
residuals are the experiment result.
