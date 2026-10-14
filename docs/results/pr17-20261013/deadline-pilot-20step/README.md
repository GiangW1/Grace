# PR17 deadline pilot results

This directory records the completed 20-step pilot for PR17. It is a time-boxed pilot and does not replace the planned 100-step formal suite.

- Model: Qwen3-4B, thinking disabled, 8192-token limit.
- Layout: actor GPU 7; rollout GPUs 0, 2, and 3.
- Arms: LoRA16 Full-PG, LoRA16 GRACE, core8 Full-PG, and core8 GRACE.
- Evaluation: seeded 64-problem MATH-500 subset, four samples per problem (256 samples per arm).
- All four arms completed at step 20; the run completed on 2026-10-14 03:04 Asia/Shanghai.

`PILOT_SUMMARY.zh-CN.md` gives the headline table. `result-index.json` contains the compact machine-readable metrics, while `eval_summaries/` preserves the four full evaluation summaries, including per-problem counts and checkpoint hashes. `configs/` and `status-summary.json` record the effective pilot setup and completion ledger.

Large assets are intentionally excluded from GitHub: model weights, checkpoints, full trajectories, per-request outputs, and environment dumps remain under `/mnt/mydata/wja/grace`.
