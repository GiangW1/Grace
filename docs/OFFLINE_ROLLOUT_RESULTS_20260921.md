# Offline predictor and parallel rollout: September 21 handoff

This GPU run completed all 12 RL training jobs and their final-checkpoint
evaluations. The user paused the remaining three-GPU seed-41 postprocessing
on September 21 at approximately 22:43 HKT. The partial GRACE step-20
evaluation is not a completed result.

The implementation starts from PR #2 at
`bc7978e4a83da4f7ebb1af66df652954adde1a07`. This follow-up also fixes
recursive nonfinite report serialization, duplicate vLLM shutdown, and
command launch during scheduler handoff. Raw audit observations are retained.

## Completed final evaluations

Qwen3-4B-Base, 40 RL steps, fixed global N=16, shared per-seed initialization
and frozen offline predictor. Evaluation uses 128 MATH-500 subset problems,
four answers per problem, temperature 0.6, top-p 0.95, and a 4096-token cap.
These are observed checkpoint scores, not equal-compute claims.

| Layout | Seed | Full-PG avg@4 | Offline GRACE avg@4 |
|---|---:|---:|---:|
| One GPU | 17 | 53.7109375% | 73.2421875% |
| One GPU | 23 | 61.1328125% | 73.8281250% |
| One GPU | 41 | 78.5156250% | 79.1015625% |
| Three GPUs | 17 | 79.8828125% | 77.1484375% |
| Three GPUs | 23 | 69.1406250% | 73.4375000% |
| Three GPUs | 41 | 75.9765625% | 65.0390625% |

The three-seed arithmetic means are 64.453125% versus 75.390625% on one GPU,
and 75.000000% versus 71.875000% on three GPUs. The differing outcomes do not
establish a stable GRACE quality advantage across layouts. Hardware layouts
produce different sampled trajectories and learned actors even with the same
seed; these scores are not an isolated hardware intervention.

## Completion and costs

All single-GPU chains and three-GPU seeds 17 and 23 have complete training,
step-0/20/40 evaluations, main audits, training-matched audits and batch audits.
Three-GPU seed 41 has complete training, final evaluations and Full-PG eval-20.
GRACE eval-20 is suspended; both eval-0 runs and all six audits are pending.
No scheduler failures were recorded before the pause.

One-GPU RL subprocess times average 19.50 minutes for Full-PG and 18.44 minutes
for GRACE. Offline fitting additionally takes 29-36 minutes per seed and is
reused by both layout consumers. RL times exclude this fitting, shared
initialization, evaluation and audit work. They include process setup and
shutdown. Some three-GPU runs share their actor GPU with other processes;
their durations are not exclusive-device speed benchmarks. The suspended
evaluation will include pause time in raw elapsed timings if resumed.

For one-GPU seed 17, the training-matched variance-times-token-cost ratio is
1.178187652; setting prediction m=0 at the same p gives 1.178203698. This
predictor has little measured control-variate benefit in that audit. These
ratios exclude offline fitting overhead and are not wall-clock speed ratios.

## Next implementation target discussed with the user

The current pool splits a finite batch between independent TP=1 workers and
waits for the entire batch before actor update. It has no persistent pending
queue to refill capacity freed by GRACE early stopping. Per-token savings
therefore need not shorten a batch dominated by a long surviving answer.

The proposed next experiment freezes an actor over a predetermined larger
sample window and uses a bounded, continuously replenished request queue.
Full-PG and GRACE must share the same queue policy and concurrency limit.
All predetermined starts, including stopped starts, contribute with the
correct estimator normalization. Do not select only the fastest completions
or replace stopped starts until an arbitrary number of full answers is reached.
Maintain the allocation protocol and snapshot consistency when pipelining.
Measure elapsed time, tail idle time and gradient variance times elapsed time.
This queue implementation and experiment have not yet been performed.

## Evidence locations

Server run root: `runs/offline-parallel-pr2-20260920-3gpu`.

- `seed-N/{one_gpu,multi_gpu}/seed-N/{full_pg,grace}` contains each method.
- `train/steps.jsonl` contains phases, nested generation timings, request
  execution, snapshot hashes and sampled process inventories.
- `eval-40/eval_summary.json` contains final scores and the evaluation manifest.
- Each chain's `command_timing.jsonl` records measured subprocess lifetimes.
- `seed-N/offline` contains predictor fitting and exported artifacts.
- `auto-rollout-20260920/status.json` and `PAUSED.md` record the user pause.

The exported package supplies an explicit completion manifest. Historical
aggregate files can predate the pause; raw run metadata and configuration
paths retain the original machine paths for provenance. Base model weights,
datasets and the conda environment are outside this run directory.
