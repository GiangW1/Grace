# PR11 dynamic score-gradient experiment results

This run used PR #11 source commit
`2131f912ab64df450699793ed8c5edeab60d69ad`. The GPU stages completed on
2026-09-27 and the final experiment status was written at
`2026-09-27T19:39:37+08:00`. A reporting-only follow-up made `q_strata()` read
the replay's two-element `half_mean_reward` field and corrected Stage A's
per-role stratum counts. Stage A and Stage B were then recomputed from the
unchanged replay and score-gradient arrays.

## Coverage

- Baseline calibration: 256 problems, 8 independent completions per problem.
- Dynamic replay: 382 live prefixes from 229 problems and 6,112
  continuations, with independent 8/8 A/B halves.
- Gradient dimension: 5,898,240; exact frozen-actor ascent gradients stored in
  FP64.
- Observed split after removing 27 problems without a surviving `t=512`
  prefix: train 142 problems/241 prefixes, validation 47/74, diagnostic 40/67.

## Stage A: representation ceiling

The exact one-vector score basis reduced residual relative to the zero
prediction by 7.17% on validation (`0.928296`) and 8.39% on diagnostic
(`0.916067`). The empirical half-A reward strata were:

| Role | Stratum | Prefixes | Free coefficient | Reward-only coefficient |
| --- | --- | ---: | ---: | ---: |
| validation | observed q=0 | 59 | 0.942604 | 0.941416 |
| validation | observed 0<q<1 | 14 | 1.057582 | 1.056490 |
| validation | observed q=1 | 1 | 0.091773 | 0.091807 |
| diagnostic | observed q=0 | 58 | 0.960860 | 0.976328 |
| diagnostic | observed 0<q<1 | 9 | 0.854025 | 1.179106 |

Ratios are residual mean divided by the zero-prediction residual mean within
the same role and stratum. Diagnostic has no observed-q=1 prefix. The
diagnostic uncertain stratum is the clearest difference between a freely
fitted score coefficient and the reward-only coefficient; it contains only
nine prefixes and is reported as a mechanism diagnostic rather than a training
benefit claim.

## Stage B: learned coefficient

Validation selected the zero predictor. The constant predictor's validation
ratio was `1.006277`; Ridge ratios were `1.133293` to `1.138779`. Diagnostic
Ridge ratios were `1.253977` to `1.263351`. The tested legacy prefix features
therefore did not learn the oracle coefficient on held-out problems.

The counterfactual AdamW directional labels had half-sample Pearson/Spearman
reliability `0.660650/0.683893` at n=8 and `0.723238/0.620291` at n=16.

## Files

- `summary/dynamic_oracle_summary.json`: final Stage A report with q strata.
- `summary/dynamic_predictor_summary.json`: final Stage B model reports.
- `summary/directional_value_summary.json`: directional overlap and label
  reliability.
- `summary/replay_summary.json` and provenance files: replay size, hashes,
  actor identity, and score-gradient identity.
- `summary/observed_split.json`: the fixed observed-problem split.
- `dynamic-pr11-20260927-review.tar.gz`: review archive containing 87 configs,
  logs, summaries, JSONL evidence files, and small arrays. Its embedded
  `archive_manifest.json` records hashes for every included and excluded file.

Review archive SHA256:
`fd2c3a1aa6f80c0b867e6927eab3774e696891192d20c446b2da03c3c6428e96`

The review archive excludes the four 18,025,021,568-byte full-gradient arrays,
the 47,186,048-byte metric vector, and the 23,612,264-byte calibration adapter.
Their hashes remain in the embedded manifest. The complete binary run archive
is kept separately because committing it would add tens of gigabytes to the
Git repository.
