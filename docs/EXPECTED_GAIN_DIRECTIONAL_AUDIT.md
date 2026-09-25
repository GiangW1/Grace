# Prefix-conditioned scalar training-value audit

This is the first fast falsification experiment for the new GRACE route. It
does not train an online scheduler. It asks whether a prefix-only feature can
predict a scalar continuation value that is defined by the current training
direction:

\[
u(h)=\mathbb{E}_z[\langle G(h,z),d_t\rangle\mid h].
\]

`G` is the frozen-actor ascent gradient and `d_t` is a frozen ascent/update
direction. A `.npy` direction can be an AdamW update direction exported from a
training step. The `reference` direction is the equal-problem mean gradient
from a disjoint replay and is a cheap secondary proxy. `loo` is a diagnostic
leave-one-problem-out direction formed from the replay pool; it is useful for
checking leakage and direction sensitivity, but should not be presented as an
online update.

## 1. Generate a quick replay on A100

First make a disjoint reference replay at `t=0`. Then make the predictor
replay with the same frozen checkpoint and pass the reference directory. The
second command stores exact prefix means and a small
`directional_values.npy`; the latter contains one scalar per continuation,
not another copy of the 5,898,240-dimensional gradient.

```bash
python scripts/replay_expected_gain.py \
  --bundles runs/directional-reference/audit_bundles.jsonl \
  --checkpoint "$CHECKPOINT" --model-path "$MODEL_PATH" \
  --decision-tokens 0 --max-continuations 16 \
  --run-dir runs/directional-reference-replay

python scripts/replay_expected_gain.py \
  --bundles runs/directional-predictor/audit_bundles.jsonl \
  --reference-dir runs/directional-reference-replay \
  --checkpoint "$CHECKPOINT" --model-path "$MODEL_PATH" \
  --decision-tokens 512 --max-continuations 64 \
  --run-dir runs/directional-predictor-replay
```

The first pass intentionally uses 64 continuations. If the two-half result is
still promising, rerun only the predictor replay with
`--max-continuations 256`; no predictor fitting has to be repeated on the
GPU.

For an actual frozen lagged update direction, use the same replay command with
`--direction-file runs/step-XX-update-direction.npy` instead of
`--reference-dir`. The file must be a finite flattened LoRA ascent/update
vector with the replay's `layout_dim` entries.

## 2. Run the CPU audit

The fit is deliberately small: zero, constant, and Ridge are the only default
models. The split is by problem, and all reported predictors are evaluated on
held-out problems.

```bash
python scripts/expected_gain_directional_audit.py \
  --replay-dir runs/directional-predictor-replay \
  --direction reference \
  --reference-dir runs/directional-reference-replay \
  --features legacy --models zero,constant,ridge \
  --n-grid 16,64 --run-dir runs/directional-value-audit-64
```

For the 256-continuation follow-up, use `--n-grid 16,64,256` and a new
`--run-dir`. To check a supplied optimizer direction, use:

```bash
python scripts/expected_gain_directional_audit.py \
  --replay-dir runs/directional-predictor-replay-256 \
  --direction file --direction-file runs/step-XX-update-direction.npy \
  --features legacy --models zero,constant,ridge \
  --n-grid 16,64,256 --run-dir runs/directional-value-audit-adamw
```

`--features cheap` is a fast sanity baseline using prefix length, cost,
reward, and baseline metadata. `--direction loo` runs the same analysis with
per-problem leave-one-out replay directions; it is a diagnostic, not a claim
about an online direction.

## 3. Read the result

`directional_value_summary.json` contains:

- `half_reliability`: Pearson/Spearman and same-problem ordering agreement
  between independent continuation halves;
- `n_grid`: the same stability and held-out predictor metrics at each available
  continuation count;
- `predictors_on_mean_label`: held-out MSE, R² against the training constant,
  Pearson/Spearman, top-20% overlap, and top/bottom actual-value spread;
- direction norm and optional reference-check cosine.

The first decision is label stability. If the scalar target is not repeatable
with 256 continuations, a poor predictor cannot be blamed on model capacity.
If it is repeatable but held-out Ridge does not beat the constant and cheap
baselines, pause this prefix-only scalar route before adding more bases or a
larger predictor. This audit does not measure refill wall-clock or training
quality; those are separate same-capacity vLLM experiments.
