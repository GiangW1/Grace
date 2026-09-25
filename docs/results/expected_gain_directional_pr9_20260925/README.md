# PR9 directional-value audit results

- Source commit: `945e0f1211355b0701a74d7cb130e4cfc6816356`
- Run date: 2026-09-25
- Predictor prefixes: 382 across 229 problems
- Predictor continuations: 6,112 (`16` per prefix)
- Reference and reference-check continuations: 512 each (`8` per prefix)
- Features: legacy prefix features, dimension 7,685
- Predictors: zero, constant, Ridge (`l2=1, 10, 100`)
- Direction: equal-problem reference gradient

## Result

The directional label is repeatable across independent continuation halves:

| Continuations | Pearson | Spearman | Same-problem ordering |
| --- | ---: | ---: | ---: |
| 8 | 0.9012 | 0.8922 | 0.8562 |
| 16 | 0.9551 | 0.9480 | 0.9085 |

The reference/reference-check direction cosine is `0.1581`. On held-out
problems, the zero baseline has diagnostic MSE `3230.27`, the constant baseline
has `3220.54`, and Ridge has diagnostic MSE between `5256.55` and `5384.60`.
Validation does not select Ridge. The scalar target is stable, but the legacy
prefix features do not provide predictive ranking signal.

The available bundles contain at most 16 continuations per prefix, so `n=64`
and `n=256` stability points were not available in this run.

## Files

`directional_value_summary.json` is the complete audit summary. The replay
metadata, prefix records, scalar matrices, logs, and split are included in the
filtered archive. The archive intentionally excludes all 5,898,240-dimensional
`mean_grads.npy` files and the original audit bundles.

The local archive is split into GitHub-sized parts in this directory. Rebuild
it in numeric order and verify SHA256 against:

`8741ffb83e4955c615d7256651c1c443f03d6ed36e41723e6e3079f704bad529`

The replay used the legacy record-only handling for the known BF16 norm drift
in these historical bundles. No source code in the PR branch was changed for
that runner compatibility adjustment.
