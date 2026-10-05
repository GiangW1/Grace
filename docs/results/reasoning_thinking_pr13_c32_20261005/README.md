# PR13 Corrected Thinking Audit, 2026-10-05

This is the new completed run based on requested commit `8cac23f`, with the
reward/probe, metric, resume, and execution fixes. It is separate from the
legacy 2026-10-01 results and the interrupted 64-continuation run.

- Qwen3-4B thinking, revision `1cfa9a7208912126459214e8b04321603b3df60c`;
  frozen rank-16 q/v LoRA, zero B, Euclidean metric, temperature 1.0.
- 32 problems, two paths, observation positions 1024 and 2048: 128 prefixes.
- 32 independent continuations per prefix, A/B halves of 16; 16 baseline
  answers per problem; four 32-token functional probes per prefix.
- 8192 response-token limit. GPU 0 computed gradients; GPU 2 served vLLM.
- Final execution configuration: grouped prefix continuations, 32 HTTP
  requests, vLLM 32 sequences / 65% memory, feature batch size 2.
  Activation checkpointing remains enabled after a real long-sequence benchmark.
- Collection completed at 06:53 HKT and all offline analyses at 07:02 HKT
  on 2026-10-05. The rollout budget was changed from 64 to 32 before this run;
  request-keyed caches were reused, while gradient statistics were recomputed.

## Results

All-prefix leave-one-problem-out comparisons at continuation probability 0.5:

| Method | Residual / zero prediction | Variance x token cost / Full-PG |
| --- | ---: | ---: |
| Zero | 1.000000 | 1.200859 |
| Cheap-feature Ridge, L2=100 | 1.002673 | 1.202465 |
| Hidden-feature Ridge, L2=100 | 1.056367 | 1.234718 |
| Independent A-fit/B-score oracle reference | 0.979904 | 1.188788 |

The fitted predictors did not improve on zero prediction in this comparison.
The independent-half oracle's product interval is [1.175601, 1.203954].
These are Euclidean variance/token-cost estimates, not measured training speedups.
Full model, position, subset, and bootstrap outputs are in `collection/analysis`
and `collection/benefit`.

## Evidence And Archive

The PR retains reports, compact per-prefix statistics, input questions, small
analysis arrays, probes, logs, configuration, source identities, and export hashes.
The filtered archive additionally retains original bundles, generated text,
token IDs, seeds, and raw problem records. Dense gradient/metric arrays,
model/LoRA weights, and duplicate HTTP caches are excluded. Source files have
not been deleted. The archive is suitable for inspecting the results; running
the full-space analysis again requires the original dense arrays or recomputation.

Combined download for both completed runs:
https://github.com/GiangW1/Grace/releases/download/pr13-pr14-results-20261005/pr13-pr14-results-small-20261005.tar.gz

`archive_manifest.json`, `archive_verification.json`, and `archive_download.json`
record the exclusions, hashes, and verified download.
