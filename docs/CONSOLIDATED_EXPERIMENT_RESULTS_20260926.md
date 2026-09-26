# Consolidated experiment status (2026-09-26)

This document separates the four experiments consolidated into PR #8. It also
records the later directional audits already present on that branch. Code
readiness, completed measurements, and claims supported by those measurements
are reported independently.

## Result matrix

| Source PR | Experiment | Code status | Result status | Main result | Evidence supported now |
| --- | --- | --- | --- | --- | --- |
| #5 | Paired feature-batch serve comparison (`1` vs `4`) | Implemented with paired trial controls and latency, queue, observed-batch, selection, token-length, and trajectory records | CPU regression coverage passes; no paired A100 result is committed | The comparison is ready to run, but there is no measured throughput or latency outcome yet | Experimental protocol only; no runtime-efficiency claim yet |
| #6 | Expected-gain replay with numerical drift recorded instead of rejected | Implemented; replay drift is retained in per-prefix and run-level provenance | Completed on 382 live prefixes from 229 problems and 6,112 continuations | Validation selected the zero gradient predictor. Diagnostic residual-risk MSE was `1,888,759,523` versus `1,782,549,613` for the constant-risk baseline; independent reference-gradient cosine was `0.159` | The original predictor did not establish predictor benefit or training-reward gain; the audit exposes a concrete representation/estimation bottleneck |
| #7 | Layer/q-v gradient bases and parameter-free multiwindow pooling | Implemented, including four-shard extraction, deterministic merge, structured suite, and archived provenance | Completed on four A100s: shards `96/96/96/94`, 382 prefixes, 10,240 features | Validation selected `global64 + legacy + zero`. Global-64 and layer/q-v-64 oracle bases captured `12.15%` and `8.03%` of diagnostic mean-gradient energy. Best multiwindow direct-gain Ridge diagnostic R2 was `-0.193` | The tested layer/q-v basis and pooling features did not solve the predictor bottleneck |
| #8 | Fast single-layer gradient-predictability audit | Implemented with CPU regression coverage and merged into the consolidated branch | No single-layer A100 result artifact is committed | The audit can cheaply test whether one layer contains usable signal before another full-gradient run | Audit method only; no single-layer predictor claim yet |

The detailed #6 result snapshot is in
[`results/expected_gain_20260925`](results/expected_gain_20260925/README.md).
The detailed #7 result and its complete split archive are in
[`results/expected_gain_structured_pr7_20260925`](results/expected_gain_structured_pr7_20260925/README.md).

## Follow-up evidence already included in PR #8

### PR #9: directional-value label reliability

The directional target was evaluated on 382 prefixes from 229 problems and
6,112 continuations. Independent continuation halves produced
Pearson/Spearman correlations of `0.9012/0.8922` at `n=8` and
`0.9551/0.9480` at `n=16`; same-problem ordering agreement was `0.8562` and
`0.9085`. The diagnostic zero and constant MSEs were `3230.27` and `3220.54`,
while tested Ridge models were worse (`5256.55` to `5384.60`).

This isolates the failure: the scalar target is repeatable, while the legacy
prefix representation plus linear Ridge does not predict its ranking. See
[`results/expected_gain_directional_pr9_20260925`](results/expected_gain_directional_pr9_20260925/README.md).

### PR #10: counterfactual AdamW direction

The counterfactual direction used `n_start=16`, three background draws, and a
saved direction norm of `0.056804897458776864`. Label reliability remained
high: Pearson/Spearman was `0.8594/0.8470` at `n=8` and `0.9212/0.9182` at
`n=16`. Validation zero MSE was `0.0395157424`; the best tested Ridge result was
`0.0400486`. On the diagnostic split, zero MSE was approximately `0.024751`,
while the best Ridge MSE was `0.0282282523` (R2 `-0.1314236`, Pearson
`0.1575373`, Spearman `0.1565057`, top-20 overlap `0.2857143`). Maximum replay
norm drift across the three shards was `4.76%`, `18.24%`, and `6.89%` and is
recorded rather than hidden.

The direction is a CPU PyTorch counterfactual AdamW proxy reconstructed from
saved optimizer groups, moments, and shared clipping. It is not the historical
CUDA update or an observed online reward change. See
[`results/expected_gain_directional_pr10_20260926`](results/expected_gain_directional_pr10_20260926/README.md).

## Claim boundary and next falsification step

The project-level claim remains that expected marginal training utility can be
estimated cheaply enough to improve online scheduling. The completed evidence
now supports a narrower prerequisite with strong controls: the directional
utility label is repeatable and falsifiable. It also rejects the current
legacy-feature/linear-Ridge realization on held-out problems. It does not yet
support an online reward, throughput, or wall-clock improvement claim.

The next decisive experiment is therefore an online paired A100 run that uses
the stable directional target to train a stronger prefix representation, then
compares the learned scheduler against zero/constant and existing scheduler
baselines under identical trajectories. The paired batch-size serve protocol
from #5 should be run separately so systems efficiency is not conflated with
predictor quality.

## Verification

- `tests/test_phase12_experiments.py`: `12 passed, 1 skipped` in the local
  dependency environment.
- Expected-gain focused suite excluding the three Torch/Windows-environment
  cases: `10 passed`.
- `py_compile` passed for the replay, optimizer probe, serve comparison,
  single-layer audit, and structured-suite entry points.
- Structured #7 archive SHA256:
  `8999d765084a16b43075c680cab341f4272dedb99eed2ac4ca747d8550a73567`.
- Directional #9 archive SHA256:
  `8741ffb83e4955c615d7256651c1c443f03d6ed36e41723e6e3079f704bad529`.
- Directional #10 archive SHA256:
  `f7e1f2b7a7d2e0f1617b1e08e4321f2e1406a8535c8307b160e8529de5be8ebc`.

The #6 snapshot omits full-gradient arrays, basis arrays, and adapter weights;
it supports result review but not full recomputation from scratch. The skipped
or excluded local cases require PyTorch or hit Windows cleanup behavior for an
open NumPy memmap; neither changes the recorded experiment metrics.
