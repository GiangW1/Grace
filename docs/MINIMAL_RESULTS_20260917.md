# Partial GPU results: 2026-09-17

This records real runs of source `a1ca77b7a5c4a50fbaa50be1c76e34f878e217bc`.
No training, scoring, or algorithm code was changed locally for this experiment.
This handoff adds documentation only. It is not the final four-method report.

## Scope and status

- Run root: `runs/minimal-chain-20260917-071528`.
- One physical A100 80 GB, `CUDA_VISIBLE_DEVICES=1`, seed 17.
- Qwen3-4B-Base; official DAPO training data with conflicting groups removed by
  the reader; MATH-500 evaluation subset.
- Each method starts afresh, including 256 shared-format SFT steps. No previous
  experiment checkpoint was resumed.
- 40 RL steps; independent evaluations at steps 0, 20, 40, each with 16 problems
  and four samples; audit with four held-out problems, two paths, positions
  128/512/1024 and eight continuations per retained prefix.
- All 269 tests passed before launch (12.98 seconds).
- At 2026-09-17 23:04 HKT, Full-PG and GRACE are complete. Uniform-CV has reached
  step 19; GRPO has not started. The queue remains running in tmux.

## Accuracy

| Method | Initial avg@4 | Step 20 avg@4 | Final avg@4 | Final pass@4 | Final parse rate | Final truncation |
|---|---:|---:|---:|---:|---:|---:|
| Full-PG | 57.8125% | 64.0625% | 73.4375% | 87.5% | 93.75% | 6.25% |
| GRACE | 67.1875% | 73.4375% | 67.1875% | 87.5% | 100% | 0% |

GRACE's final accuracy equals its starting accuracy. Its midpoint improvement
did not persist. This coincides with selective continuation after warmup, but
these observations do not establish causation. Initial evaluation scores differ,
and the small, single-seed sample does not establish method superiority.

## Mechanism and audit

GRACE formed its first data-derived basis at step 4, refreshed it at step 32,
and finished with `basis_id=2`, `predictor_synced_basis_id=2`, reservoir size 64,
and `allocation_ready=true`. All 20 post-warmup steps report allocation ready.
It stopped 190 of 416 post-warmup starts; Full-PG had 320 starts in those steps.
Stopping proportion is not a speedup measurement.

GRACE's completed audit contains 20 prefix bundles and 160 continuations; all
stored gradients are finite. Its selected subset has one bundle. The selected
512-token point estimates are rho_L=1.834111 and rho_A=1.142857; these sparse
estimates do not establish the paper's phenomenon. Full-PG's selected subset is
empty (19 total bundles).

GRACE audit variance-times-token-cost facts:

| Quantity | Full-continuation reference | GRACE estimator |
|---|---:|---:|
| Variance | 45.157752 | 166.920579 |
| Token cost proxy | 1206.75 | 757.991877 |
| Variance times cost | 54494.116853 | 126524.442790 |

Cost falls about 37.2%, but variance increases about 3.70x. The product ratio is
2.321800, above the intended value below 1. These are audit estimates with a
token-cost proxy, not measured wall-clock speedup or a direct comparison of the
separately trained Full-PG and GRACE actors.

## Runtime diagnosis

Minutes from completed training artifacts:

| Component | Full-PG | GRACE |
|---|---:|---:|
| Baseline prescan | 22.094 | 24.834 |
| Prefix generation/features | 3.655 | 5.780 |
| Continuation | 11.194 | 13.092 |
| Actor backward | 3.163 | 2.809 |
| Predictor updates including basis work | 0.029 | 7.575 |
| Sum of recorded step wall times | 40.424 | 54.891 |
| End-to-end training | 48.918 | 207.701 |

Phase rows are nested inside step times; do not add them to the step total.
Step timers include adapter synchronization and logprob probes, but are sampled
before `persist_training_step`. The roughly 153-minute GRACE difference includes
initialization and persistence, and cannot be called pure algorithm overhead or
pure checkpoint time without a separate timer.

Persistence writes two compressed complete snapshots per step, followed by a
checkpoint hash. GRACE's final snapshot is about 1.19 GiB versus 0.061 GiB for
Full-PG; per-step snapshots total about 26.85 GiB versus 2.46 GiB. This is a clear
engineering bottleneck to measure. No checkpoint-frequency optimization was
applied during the running comparison.

Post-warmup actual generated suffix totals are 150100 for GRACE and 327755 for
Full-PG. These are different trained policies with different starts; the 54%
reduction is not a controlled speedup estimate. Smaller decode batches, sequence
lengths, additional starts, prescan and predictor work can prevent token savings
from translating proportionally into elapsed time.

## Comparison with the first run

The first run is `runs/minimal-chain-20260916-123400`.

| GRACE metric | First run | Current run |
|---|---:|---:|
| Initial avg@4 | 62.5% | 67.1875% |
| Final avg@4 | 65.625% | 67.1875% |
| Final pass@4 | 75% | 87.5% |
| Final truncation | 1.5625% | 0% |
| First nonzero basis step | 32 | 4 |
| Audit variance-times-cost ratio | 1.248188 | 2.321800 |
| Selected audit bundles | 0 | 1 |

The new implementation establishes and synchronizes the basis earlier, but has
not demonstrated accuracy or variance-efficiency gains in this run. Cross-run
differences are not an isolated ablation: predictor, RNG, scoring and audit code
changed, and initial sampled scores differ. Raw training losses under different
scales/heads are not directly comparable.

## Follow-up for implementation

Finish Uniform-CV and GRPO before attributing the outcome to adaptive allocation.
Measure persistence separately and reduce duplicate full-state compression in a
future run. Check held-out residual-risk ranking and low-probability/high-residual
cases. Compare time to a common accuracy or accuracy at equal compute, rather
than using stopping rates as evidence of benefit. These are diagnostics, not
execution gates, and do not change the current running experiment.

## Artifact handoff

The partial results archive includes completed Full-PG and GRACE directories,
their logs, run configurations, raw trajectories, evaluation answers, audit
gradients, and this report. It excludes active Uniform-CV data and queued GRPO,
all files above 20 MiB, conda, cache, and downloaded assets. Exact exclusions
and SHA256 checksums are included. Large original files remain on the server;
the archive alone cannot resume training.
