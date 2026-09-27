# PR10 counterfactual directional audit

This directory records the A100 run using the PR10 directional provenance
implementation on 2026-09-26.

- 382 predictor prefixes across 229 problems and 6,112 continuations.
- Counterfactual AdamW direction: `n_start=16`, three background draws,
  direction norm `0.056804897458776864`.
- Half-sample stability for the counterfactual direction: Pearson/Spearman
  `0.8594/0.8470` at `n=8` and `0.9212/0.9182` at `n=16`.
- Maximum replay norm drift by shard: `4.76%`, `18.24%`, and `6.89%`.
- Held-out legacy-feature validation MSE: zero `0.03952`, constant `0.04071`,
  best Ridge `0.04005`; the prefix predictor did not beat zero.

The historical BF16 bundles do not reproduce their saved gradient norms within
the old hard threshold. The run therefore records each relative norm error and
continues the directional calculation; the provenance and direction sidecar
checks remain enabled. `mean_grads.npy` files are intentionally omitted from
the filtered archive because they contain the 5,898,240-dimensional full
gradient matrix.

The filtered archive is split into `archive/part-00` and `archive/part-01` in
the PR. Reassemble them in numeric order with `cat archive/part-* >
directional-pr10-20260926-filtered.tar.gz`; its SHA256 is
`f7e1f2b7a7d2e0f1617b1e08e4321f2e1406a8535c8307b160e8529de5be8ebc`.
