# PR12 Dynamic Score Audit

This result bundle records the PR12 trajectory replay and the two CPU audit
stages run from commit `32d2c69622df2383ac086816e2cc54c90bc4cd4a`.

The replay contains 376 pre-answer prefixes, 6,016 continuations, and a
5,898,240-dimensional FP64 gradient. Three GPU shards were merged before the
CPU analysis. The raw replay arrays are intentionally excluded from this
bundle because they are about 50 GB; their provenance and metric hashes are in
`replay-metadata/`.

Stage A measured the representation ceiling. The oracle residual ratio to the
zero baseline was 0.791 on validation and 0.841 on diagnostic. Stage B tested
the legacy prefix features. The selected validation model was the zero
baseline (ratio 1.000); constant was 1.005 and ridge ranged from 1.131 to
1.137. The legacy feature predictor therefore showed no useful gain in this
run.

The analysis uses the closed-form one-column projection and blockwise residual
evaluation implemented in `grace_gc/audit/dynamic_score.py`. These avoid an
SVD and full dense copies of the 5.9M-dimensional arrays.

The accompanying archive contains summaries, configurations, split manifests,
logs, coefficient arrays, and replay provenance. It does not contain model
checkpoints, raw replay arrays, or input bundles.
