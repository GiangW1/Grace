# Structured expected-gain results (2026-09-25)

These results were produced from PR commit `007b762` using the completed
expected-gain replay from the preceding audit. Multiwindow feature extraction
was split contiguously across four A100 GPUs as 96, 96, 96, and 94 prefixes,
then merged back into the original 382-prefix order. The merged feature matrix
has 10,240 columns. The four shard extraction times sum to 23.30 seconds; the
CPU structured suite took 37.07 seconds.

Validation selected `global64 + legacy + zero`. The structured layer/q-v basis
and the added multiwindow features did not improve validation full-space
residual over the zero predictor. On diagnostic problems, the global 64-D
oracle basis captured 12.15% of observed mean-gradient energy, while the
layer/q-v 64-D oracle basis captured 8.03%. Direct gain Ridge prediction with
multiwindow features had diagnostic R2 -0.193 at its best tested penalty. The
two independent reference estimates had cosine 0.159.

Files:

- `structured_gain_summary.json`: full scalar, basis, block, and per-layer results.
- `basis_coefficients.npz`: coefficients needed to reconstruct fitted bases.
- `feature_provenance.json`: actor, replay, shape, and four-shard provenance.
- `split.json`: fixed train/validation/diagnostic problem split.
- `expected-gain-structured-pr7-007b762.tar.gz.part-00` through `part-08`:
  complete 57 MB run output split for reliable GitHub upload. Concatenate the
  parts in numeric order to recover the gzip archive; it includes merged and
  per-shard feature matrices, logs, and configurations.

Archive SHA256:
`8999d765084a16b43075c680cab341f4272dedb99eed2ac4ca747d8550a73567`

The archive does not duplicate the preceding run's 17 GB full-gradient replay;
the suite requires that replay if the full analysis is rerun from scratch.
