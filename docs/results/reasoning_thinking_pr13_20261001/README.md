# PR13 Thinking Audit, 2026-10-01

This run collected the original 32 problems using post-trained Qwen/Qwen3-4B
(revision `1cfa9a7208912126459214e8b04321603b3df60c`) with thinking enabled.
GPU 0 and GPU 2 computed frozen-actor gradients; GPU 3 served vLLM.
The fresh rank-16 q/v LoRA had zero B matrices and did not change the initial
model policy. It has no Adam optimizer history, so all comparisons are Euclidean.

## Completed Work

- 32 problems, two paths per problem, decision points at 1024 and 2048 tokens.
- 128 prefixes, 64 independent continuations per prefix: 8192 continuations.
- 16 independent baseline answers per problem and four 32-token forced-answer
  probes per prefix. The response budget was 8192 tokens.
- Collection completed on 2026-10-01 at 21:11 HKT; offline analysis completed
  at 21:21 HKT. Collection and analysis took about 20 hours 43 minutes.
- Frozen actor identity matched across both gradient workers.

## Original Reward Results

The following are leave-one-problem-out comparisons across all 128 prefixes,
at continuation probability p=0.5. Lower ratios are better.

| Method | Residual / zero prediction | Variance x token cost / Full-PG |
| --- | ---: | ---: |
| Cheap-feature MLP256 | 0.998521 | 1.200672 |
| Hidden-feature Ridge, L2=100 | 1.007057 | 1.205803 |
| Hidden-feature MLP64 | 1.013635 | 1.209756 |
| Independent A-fit/B-score oracle reference | 0.946868 | 1.169625 |

The token cost ratio is 0.600487. The oracle reference's product ratio has a
problem-cluster bootstrap 95% interval of [1.143267, 1.193918]. This noisy
oracle reference is not a theoretical ceiling. These are token-cost estimates,
not measured GPU training speedups. Original reports remain unmodified.

## Scoring Limitations

1. The collected reward implementation ignores the `truncated` argument and
   can extract answers inside thinking. It counted 2481/8192 outputs as correct
   (30.2856%), including 728 truncated outputs. There were 6133 truncated
   outputs (74.8657%). Normal finishes with a correct answer after `</think>`
   numbered 1753/8192 (21.3989%). Gradient labels and predictor comparisons
   have NOT been recomputed under this completed-answer definition.
2. Functional-probe scoring omitted the prefilled `Answer:` and counted every
   probe as wrong. Rescoring the saved output as `Answer:` plus generated text
   yields 84/512 correct probes, 19/128 majority-recoverable prefixes, and 13
   strict pre-answer undecided prefixes under the existing definition. The
   original mechanism report's empty strict subset is therefore invalid.

`checked-20261004/checked_summary.json` and
`checked-20261004/functional_probe_rescores.jsonl` record these separate
descriptive corrections. They do not replace original rewards or gradient labels.

## Filtered Archive And Cleanup

The filtered archive preserves original trajectories, text, token IDs, seeds,
features, probe outputs, summaries, small analysis arrays, logs, provenance,
source snapshots, input selection, and the small frozen LoRA checkpoint.
Dense full-space gradient arrays and duplicate HTTP rollout caches are omitted.
The archive manifest records original sizes and SHA-256 hashes for included
and excluded files. Each included source file is at most 20 MiB.

After archive integrity and GitHub upload are verified, this run's omitted
gradient arrays, duplicate caches, downloaded model weights and incomplete
download fragments are deleted. The exact deletion list and bytes are recorded
separately. Model configuration and tokenizer files are retained. Recomputing
full-space gradients requires downloading the pinned model and replaying the
saved tokens with the retained frozen LoRA checkpoint; the filtered archive
does not contain the dense arrays themselves.

The verified filtered archive is 213697486 bytes (203.80 MiB):

https://github.com/GiangW1/Grace/releases/download/pr13-thinking-results-20261001/grace-pr13-thinking-20261001-results-small.tar.gz

SHA-256: `114df9c59cd705aa81790bc097f97df4edfa1b406e09e88025a9b69ad524d26a`.
`archive_manifest.json` and `archive_verification.json` accompany the result
reports in this PR. `cleanup_plan.json` lists the exact files authorized for
removal; the subsequent cleanup receipt records actual deletion.
