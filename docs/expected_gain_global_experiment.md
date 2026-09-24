# Frozen-actor global expected-gain experiment

This experiment is independent of the truncation/refill server experiment. It
tests whether a prefix predicts an **expected, signed training contribution**,
and whether 8 or 64 global LoRA directions are enough for a useful gradient
control variate. It does not change the PR4 training loop or scheduler.

## Inputs and isolation

Use the same Full-PG step-40 checkpoint and base model for **every** replay.
The audit JSONL must retain continuation token IDs, reward, baseline, exact
full-gradient norms and prefix features. The 256-dimensional saved JL sketch
is never used as a full gradient. `replay_expected_gain.py` recomputes the
q/v LoRA A/B gradient from the frozen actor and checks it against saved norms.
It writes one FP64 gradient mean per live prefix, not every full gradient.

Start from an existing training-source file and an independent final-evaluation
file. The preparation script excludes exact normalized evaluation prompts and
reserves disjoint problem IDs. Its default is 256 predictor problems, 64
reference-construction problems, and 64 separate reference-check problems.
Prior inspected problem IDs can be pinned to the **training** side using a
JSON array passed as `--prior-train-ids`. The old 64-problem audit can be
replayed instead of regenerated only if its frozen actor, LoRA layout, and
sampling protocol match; do not infer a full gradient from its JL vectors.
To reuse it, add `--prior-train-ids old_ids.json
--omit-prior-from-predictor-file` to preparation, replay the old audit with
`--max-continuations 16 --reference-dir ...` and the same checkpoint, then run
`python scripts/merge_expected_gain_replays.py --replay OLD_REPLAY
--replay NEW_REPLAY --run-dir MERGED_REPLAY`. Use `MERGED_REPLAY` in the suite.
The merge checks actor, LoRA order, decision point, continuation count and
duplicate problem IDs. It cannot correct an old audit sampled under another
policy or protocol.

```bash
python scripts/prepare_expected_gain_data.py \
  --data-path /data/train.jsonl --eval-data-path /data/final_eval.jsonl \
  --output-dir runs/expected-gain-data

python scripts/audit.py --generate --split audit \
  --data-path runs/expected-gain-data/predictor.jsonl \
  --checkpoint /checkpoints/full-pg-step40 --model-path /models/Qwen3-4B-Base \
  --backend gpu_verl --config configs/experiments/expected_gain_predictor_a100.yaml \
  --run-dir runs/expected-gain-predictor-audit

python scripts/audit.py --generate --split audit \
  --data-path runs/expected-gain-data/reference.jsonl \
  --checkpoint /checkpoints/full-pg-step40 --model-path /models/Qwen3-4B-Base \
  --backend gpu_verl --config configs/experiments/expected_gain_reference_a100.yaml \
  --run-dir runs/expected-gain-reference-audit

python scripts/audit.py --generate --split audit \
  --data-path runs/expected-gain-data/reference_check.jsonl \
  --checkpoint /checkpoints/full-pg-step40 --model-path /models/Qwen3-4B-Base \
  --backend gpu_verl --config configs/experiments/expected_gain_reference_a100.yaml \
  --run-dir runs/expected-gain-reference-check-audit
```

On one A100, run these audit jobs **sequentially**. The replay also loads the
actor for backward; stop vLLM between audit and replay if memory is tight.
This audit is expensive: 256 × 2 × 16 planned predictor continuations plus
two 64 × 1 × 8 reference pools, before early finishes. Each audit already
computes full gradients for its exact norm diagnostic; the replay recomputes
them to recover full directions. Plan disk space for about 24 GiB of FP64
predictor means at 512 live prefixes and 5.9M dimensions, plus roughly
2.8 GiB per 64-column basis. Actual sizes follow the live-prefix count.

```bash
python scripts/replay_expected_gain.py \
  --bundles runs/expected-gain-reference-audit/audit_bundles.jsonl \
  --checkpoint /checkpoints/full-pg-step40 --model-path /models/Qwen3-4B-Base \
  --max-continuations 8 --skip-prompt-features \
  --run-dir runs/expected-gain-reference-replay

python scripts/replay_expected_gain.py \
  --bundles runs/expected-gain-reference-check-audit/audit_bundles.jsonl \
  --checkpoint /checkpoints/full-pg-step40 --model-path /models/Qwen3-4B-Base \
  --max-continuations 8 --skip-prompt-features \
  --run-dir runs/expected-gain-reference-check-replay

python scripts/replay_expected_gain.py \
  --bundles runs/expected-gain-predictor-audit/audit_bundles.jsonl \
  --reference-dir runs/expected-gain-reference-replay \
  --checkpoint /checkpoints/full-pg-step40 --model-path /models/Qwen3-4B-Base \
  --run-dir runs/expected-gain-predictor-replay
```

The suite creates a 160/48/48 problem split when all 256 predictor problems
remain live at t=512. Early-finished problems are counted but cannot enter
the live-prefix fit; it scales the three groups down rather than silently
turning suffixes into independent test examples. It learns the basis and
feature scaling only from training problems. It compares mean-SVD and
problem-crossfit predictable directions at k=8/64 (recording actual rank),
zero/constant/Ridge/MLP heads, prompt-only and cheap length/baseline inputs,
out-of-problem predicted reward, exact full-space gradient residual, signed
reference-direction gain, risk, and continuation cost. The reference direction
weights reference **problems**, rather than their number of continuations.

```bash
python scripts/expected_gain_suite.py \
  --replay-dir runs/expected-gain-predictor-replay \
  --reference-dir runs/expected-gain-reference-replay \
  --reference-check-dir runs/expected-gain-reference-check-replay \
  --prior-train-ids runs/expected-gain-data/prior_train_ids.json \
  --device cuda \
  --n-start 8 --learning-rate 0.0001 \
  --run-dir runs/expected-gain-global-suite
```

Set `--n-start` to the actual saved run's fixed N; 8 is only this repository's
default. The suite chooses basis/head settings by validation residual and
keeps diagnostic problems separate. It splits validation problems again for
risk-head fitting and probability calibration. Its allocation rows are
**offline variance/cost diagnostics**, not measured vLLM wall-time speedups.
The predictor replay uses the already-built reference direction to retain
two disjoint half-suffix gain means as scalars, so the suite can report their
correlation and same-problem ordering agreement without saving all suffix
gradient vectors. `gain_direct` is a signed local SGD-direction label (up to η/N); it is not an
observed accuracy improvement. The second reference pool reports direction
cosine and should be inspected before interpreting gain prediction.

The separate update probe replays four independent continuations from each
of up to 16 validation prefixes and three training-side background batches.
It restores AdamW first/second moments, step, LR, weight decay and the same
gradient clipping rule for *each* counterfactual. The candidate is scaled by
the saved fixed N. It reports `g_ref · [A(b+G/N)-A(b)]`, alongside the simple
SGD-direction label. This remains a local proxy, not a reward re-evaluation.
It fails explicitly if the checkpoint lacks optimizer state.

```bash
python scripts/probe_expected_gain_update.py \
  --bundles runs/expected-gain-predictor-audit/audit_bundles.jsonl \
  --replay-dir runs/expected-gain-predictor-replay \
  --reference-dir runs/expected-gain-reference-replay \
  --split-manifest runs/expected-gain-global-suite/split.json \
  --checkpoint /checkpoints/full-pg-step40 --model-path /models/Qwen3-4B-Base \
  --n-start 8 --run-dir runs/expected-gain-update-probe
```

Interpreting the result still requires the later, separate step in the
reviewed plan: update copied models and regenerate answers on **independent**
evaluation problems, followed by the vLLM `serve` truncation/refill comparison
at equal A100 time. Neither result is produced by these scripts.
