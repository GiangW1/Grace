# Minimal GPU Results: 2026-09-18

## Run and Reproduction

- Run root: `runs/minimal-chain-20260918-063159`; seed directory: `seed-17`.
- Experiment source: `3c03ce9c972cf25b7978c43c3abdb659f692652f`.
- No tracked source edits were made during this run. This handoff adds documentation only.
- One physical A100 80 GB, `CUDA_VISIBLE_DEVICES=1`, conda environment `grace`.
- Model: `Qwen/Qwen3-4B-Base`; torch `2.10.0+cu128`, vLLM `0.18.0`.
- Command, after loading model/data environment exports:

  ```bash
  SEEDS=17 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    bash scripts/run_minimal_gpu.sh runs/minimal-chain-20260918-063159
  ```

- Configuration stack: `default.yaml`, `experiments/minimal_gpu.yaml`,
  `experiments/minimal_gpu_repaired.yaml`, `experiments/minimal_gpu_deeper.yaml`,
  `hardware/a100_1.yaml`; audit stages additionally use the audit overrides.
  Each stage's saved `config.yaml` is the authoritative merged configuration.
- Shared format SFT was trained anew for this run. No previous RL checkpoint was resumed.
  All four methods have initial actor SHA-256
  `590e8b8087c54bcbe8e059dca47561402b6d324c1fb18c0ec5b3e9d659fa1886`.
- Official DAPO training data was loaded with conflict-group exclusion recorded in
  `data_conflicts.json`. Evaluation used the same seeded 128-problem MATH-500 subset,
  with four samples per problem and a 4096-token response limit.
- Training: 40 steps, four prompts per batch, adaptive start count, decision at 512
  tokens, 2048-token response limit, 20 predictor warmup steps, beta 0.75,
  `predictable_crossfit` basis. This was step-budget comparison, not matched wall time.
- The chain finished at `2026-09-19T00:12:52Z` (08:12 Hong Kong), exit status 0.
  All four training summaries report `run_status=complete` and step 40.

## Observed Results

Percentages below are from saved evaluation summaries. Train time excludes shared
SFT, independent evaluations and audits; it includes each training run's measured
setup, training and persistence envelope.

| Method | Initial avg@4 | Step 20 avg@4 | Final avg@4 | Final pass@4 | Train seconds |
|---|---:|---:|---:|---:|---:|
| Full-PG | 62.3047% | 64.8438% | 71.2891% | 85.9375% | 1486.43 |
| GRACE | 63.8672% | 71.0938% | 70.5078% | 88.2813% | 2392.41 |
| Uniform-CV | 62.5000% | 31.0547% | 53.9063% | 75.7813% | 2518.23 |
| GRPO | 62.1094% | 75.5859% | 54.8828% | 82.8125% | 1009.72 |

The identical initial actor hashes do not imply identical sampled evaluation
answers: initial measured scores differ. GRPO's best measured intermediate score
exceeds GRACE's, despite its worse final checkpoint. One seed does not establish
stable method rankings or equal-compute superiority.

GRACE trained on 340 main trajectories versus Full-PG's 224. After warmup, GRACE
stopped 24 of 67 eligible trajectories (80 total main trajectories in those steps).
Its 61% longer training time is an observed run cost, not an equal-work throughput
comparison. No wall-time advantage was demonstrated.

From step 20 to 40, GRACE's average response length rose from 557.47 to 709.46
tokens, parse rate fell from 99.0234% to 96.0938%, and length truncation rose from
1.3672% to 3.5156%. Final avg@4 was 0.5859 percentage points below step 20.

## GRACE Audits

Each ratio below compares the GRACE estimator with full continuation at the same
frozen GRACE checkpoint, not with the separately trained Full-PG checkpoint.

| Audit | Prefix bundles | Full variance | GRACE variance | Full token cost | GRACE token cost | Variance-cost ratio |
|---|---:|---:|---:|---:|---:|---:|
| Multi-position, 4096 tokens | 80 | 6673.8335 | 8837.8113 | 2091.5813 | 1688.5645 | 1.069085 |
| Training position 512, 2048 tokens | 30 | 5698.5450 | 7691.6197 | 1381.7333 | 1164.2230 | 1.137276 |

Lower variance-cost ratios are better. Neither audit showed an improvement.
These are pooled empirical prefix diagnostics with a token cost proxy, excluding
prompt prefill, baseline generation, backward, predictor, I/O and engine overhead.
They are not fixed-prompt training-batch variance or measured GPU speedups.

The batch audit used four fixed prompts, four starts each (N=16), and eight
replicate batches. The saved trace sample covariances were:

| Quantity | Full continuation | GRACE |
|---|---:|---:|
| Raw gradient | 862.096945 | 1392.396116 |
| Clipped gradient | 0.983065 | 0.986931 |
| Replayed optimizer update | 0.0008619094 | 0.0008584454 |

Mean paired cosine was 0.961232 for gradients and 0.988702 for optimizer updates.
Updates were directionally close, but the audit does not establish variance
reduction: the raw gradient sample variance was higher and update variance was
approximately unchanged. These eight replicates are conditional on one prompt
set, baseline, optimizer and checkpoint, not independent training seeds.

The selected-subset multi-position metric at t=512 was unavailable; it must not
be reported as zero or as meeting a paper target. All observations remain in the
audit files. GRPO batch audit is marked not applicable because its objective
differs from the R-minus-baseline HT/CV estimator.

## Predictor Limitations and Earlier Runs

The final step's predictor diagnostics report `m_shrink=0` and calibration gamma
0. All 30 training-position audit bundles have zero `m_pred` and zero effective
coordinates. Thus the final checkpoint's control variate contributes no correction
in this audit; this does not mean it was zero throughout training.

The final reservoir has 86 labels, but the configured freshness window leaves 22
active labels. The fit/basis split has 10 labels from eight problems, only two
with nonzero gradients; calibration has seven labels. This is an observed lack
of useful supervision, not a proposed minimum-sample requirement.

In the training-position audit, the basis captures 0.00528447 of realized stochastic
gradient energy. This is not a measurement of the predictable conditional mean.
At matched token cost, replacing the allocation by uniform probabilities gives a
variance-cost ratio of 1.128887 versus the actual 1.137276. Replacing predicted risk
with retrospectively observed report risk gives 0.842582. That hindsight result
suggests an allocation opportunity on these samples, not a deployable predictor or
evidence of out-of-sample savings.

| GRACE run | Evaluation problems | Final avg@4 | Multi-position variance-cost ratio |
|---|---:|---:|---:|
| `minimal-chain-20260916-123400` | 16 | 65.6250% | 1.248188 |
| `minimal-chain-20260917-071528` | 16 | 67.1875% | 2.321800 |
| `minimal-chain-20260918-063159` | 128 | 70.5078% | 1.069085 |

The ratios moved closer to parity in the latest run. These are descriptive
cross-run observations, not controlled ablations: sample selection, sample sizes,
beta, prompts per batch, basis/predictor implementation and measurement code
changed. The second run's GRACE training start/finish interval was 207.70 minutes
versus 39.87 minutes in the latest run; this is not an equal-work algorithm speedup.
The latest full chain completed without the runtime interruptions seen in earlier
chains, but its efficiency objectives remain unproven.

## Validation and Handoff

- Pre-launch tests: 421 passed, 1 skipped in 84.21 seconds; evidence in `logs/tests.log`.
- Real GPU train/eval/audit stages completed; the chain exited successfully.
- `seed-17/report.md`, `stages.json`, per-stage summaries, `health.json`,
  `steps.jsonl`, `persistence.jsonl`, manifests, trajectories and timing ledgers
  retain the detailed evidence. Training health files describe their final batch,
  not aggregate run quality.
- The current result supports similar final accuracy to Full-PG in this one run,
  but does not support the paper's wall-time or variance-cost efficiency claims.
- Potential engineering follow-up: audit continuations and gradient calls are
  currently serial per prefix; zero-coefficient gradient calls are not skipped.
  No such optimization was applied during this experiment.
- Results archive: `runs/minimal-chain-20260918-063159-results.tar.gz`.
  It includes all four methods and the shared initialization metadata, excludes
  individual files larger than 20 MiB, and contains `archive_manifest.json` with
  included/excluded paths and SHA-256 hashes. Original large states remain on the
  server. The archive is for analysis, not a self-contained resumable checkpoint.
