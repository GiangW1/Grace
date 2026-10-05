# PR14 Learning Completion, 36 Problems, 2026-10-05

This is the completed Experiment A based on PR14 commit `2244a7d`, with
completed-answer scoring, probe, trajectory/cache, crossing, resume, and
execution fixes. It uses the original seeded question order reduced to 36.

- Frozen Qwen3-4B thinking, revision `1cfa9a7208912126459214e8b04321603b3df60c`;
  rank-16 q/v zero-B LoRA, Euclidean metric, temperature 1.0.
- 36 problems, four paths, eight independent continuations per live prefix,
  A/B halves of four; 16 independent baseline answers per problem.
- Positions 256, 512, 1024, 2048, and 4096; 8192 response-token limit.
- 700 live prefixes: 144 at each of the first four positions and 122 at
  the last position, after paths finishing before observation are excluded.
- GPU 1 computed gradients; GPU 3 served vLLM. Final configuration groups
  four prefixes into up to 32 requests, with vLLM 32 sequences / 65% memory.
  Activation checkpointing remains enabled.
- Collection completed at 06:44 HKT and analysis at 06:46 HKT on 2026-10-05.
  Previously completed problems/prefixes and exact request caches were resumed.

## Results

For the all-prefix group, point estimates are:

| Position | Live prefixes | F_pop | residual_pop | rho_A |
| --- | ---: | ---: | ---: | ---: |
| 256 | 144 | unavailable | unavailable | 1.027704 |
| 512 | 144 | -0.582507 | 3.366471 | 0.938642 |
| 1024 | 144 | -0.783690 | 0.611232 | 0.889356 |
| 2048 | 144 | unavailable | unavailable | 0.781265 |
| 4096 | 122 | 0.830254 | 0.147652 | 0.456094 |

The population denominator estimate U_MM is negative at 256 and 2048, so
normalized point estimates there are unavailable. The 4096 point estimate
is descriptive and must be read with the problem-bootstrap intervals;
it alone does not establish population learning completion.
`analysis/learning_completion_summary.json` contains all groups, batch sizes,
crossing handling, confidence intervals, and defined-bootstrap counts.

## Evidence And Archive

The PR retains analysis, compact prefix statistics, probes, input questions,
logs, configuration, shard provenance, and export hashes. The filtered archive
also retains generated text, token IDs, original bundles and raw problem
records, plus small Gram caches. Dense gradient arrays, weights, and duplicate
HTTP caches are excluded. Source files have not been deleted. Full-space
analysis reruns require the original arrays or recomputation.

Combined download for both completed runs:
https://github.com/GiangW1/Grace/releases/download/pr13-pr14-results-20261005/pr13-pr14-results-small-20261005.tar.gz

`archive_manifest.json`, `archive_verification.json`, and `archive_download.json`
record the exclusions, hashes, and verified download.
