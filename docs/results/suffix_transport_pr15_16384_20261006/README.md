# PR15 Suffix Transport, 16384 Tokens, 2026-10-06

The requested source was `1e4e496`; both completed audits ran clean source
`4675a348c361002448f9898c84832f82d79b20d7` after the execution optimizations.
This directory contains the new 16384-cap run, not the interrupted 8192 run.

## Configuration And Completion

- Qwen3-4B with thinking enabled, temperature 1, frozen shared zero-B q/v LoRA.
- The same 32 audit questions, two independent prefix groups per question,
  prefix position 512, and eight independent continuation draws per group.
- GPU 0: HF actor/gradients; GPU 2: vLLM generation. No OOM or retry was recorded.
- Independent Full-PG completions, paired donor-only/ST and expensive full-RB
  controls remain enabled. The actor hash is checked after the frozen audit.
- Full-RB reuses gradient forwards for suffix scores. Next-group generation
  overlaps actor work; half of the actor layers use activation checkpointing.
- m=2 completed at 15:02 HKT, taking 10972.68 seconds (3.05 hours).
- m=4 completed at 19:36 HKT, taking 16428.62 seconds (4.56 hours).
- Expanded multi-method/multi-seed training and MATH-500 evaluations were
  deferred at the user's request. They are not reported as completed.

## Results

Lower gradient trace variance is better. Conditional variance fixes the sampled
prefixes; population variance also includes fresh prefix/problem variation.

| Group Size | Full-PG Conditional Variance | ST Conditional Variance | ST / Full-PG | ST / Full-PG Population Variance |
| --- | ---: | ---: | ---: | ---: |
| m=2 | 60730.58 | 119572.56 | 1.9689 | 1.9084 |
| m=4 | 29790.84 | 138491.87 | 4.6488 | 4.3697 |

ST lambda=0 and lambda=1 have identical reported variances; full-RB agrees to
roundoff. The paired ST-minus-donor variance difference and bootstrap interval
are both zero. Full-PG-minus-donor intervals are [-70249.32, -48438.14] for m=2
and [-142400.27, -83797.67] for m=4. Population variances are point estimates;
the conditional intervals do not apply to population estimates.

Every one of the 512 donor draws in each audit assigns donor posterior mass
1.0 at stored floating-point precision, with ESS=1. Non-donor mass is at most
8.93e-29 for m=2 and 9.26e-20 for m=4. The donor's sequence-logprob advantage
over the best receiver has median 203.58 and 173.83, respectively. The prefixes
are distinct in every group. The correction is effectively zero, explaining
the identical lambda results: exact long-suffix likelihood leaves no useful
cross-prefix sharing in this configuration.

Full-PG averages m independent completions, while the degenerate ST estimator
uses one donor. This variance comparison is not an equal-cost training or
speed comparison. It does not establish final trained-model accuracy, nor does
it evaluate an offline predictor: this ST experiment uses exact suffix scores.

| Group Size | Actual Completions | Successful Samples | Truncated | Truncation Rate |
| --- | ---: | ---: | ---: | ---: |
| m=2 | 1024 | 883 (86.23%) | 52 | 5.08% |
| m=4 | 2048 | 1812 (88.48%) | 79 | 3.86% |

These success rates describe shared audit rollouts, not separate method
evaluations or trained accuracy. All truncated responses reached the 16384
response-token cap. The earlier 8192 run changed the reward horizon and was
interrupted; its observations are not pooled with these results.

## Evidence And Download

`analysis.json` records the derived statistics. Each audit retains its original
summary/configuration/source identity and compact group, draw and trajectory
records, including posterior weights, likelihoods, seeds, rewards, lengths and
token hashes. `export_manifest.json` records hashes of the original inputs and
the compact exports. `export_results.py` reproduces the export from the raw run:

```bash
python docs/results/suffix_transport_pr15_16384_20261006/export_results.py \
  --run-dir /path/to/grace-pr15-st-16384-optimized-20261006-run
```

Filtered archive:
https://github.com/GiangW1/Grace/releases/download/pr15-st-16384-results-20261006/grace-pr15-st-16384-20261006-results-small.tar.gz

The archive is 46,815,588 bytes (44.65 MiB), with 174 retained files and 140
excluded files totaling 24,387,750,426 bytes. All retained payload hashes and
the 64/64 groups, 512/512 donor draws, and 1024/2048 trajectories were verified.
SHA-256: `488c78602b0725a973b9b7594b3728acd46e1ecc8322e05bc985a8ab66b5e348`.

The archive retains original generated text/token IDs, complete posterior
draws/groups, request seeds, audit questions, metadata, logs and compact analysis.
Dense `moments/*.npz`, checkpoints, model/LoRA weights and the unused training
input are excluded. The compact export cannot recompute full-vector population
statistics without the original dense moments. Source files were not deleted.

The actor profile is a teacher-forcing execution benchmark; its 16384 fixture
repeats existing tokens. It is not a new rollout or whole-run speedup measurement.
The 16384-cap smoke naturally finished before the cap; actual audits did include
16384-token responses. Archive inventory, hash and verification are recorded
beside this README. The relevant code regression run passed all 53 tests in
17.75 seconds (`test_suffix_transport`, `test_audit_execution`, `test_rollout_pool`).
