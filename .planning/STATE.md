---
gsd_state_version: '1.0'
status: phase-complete
progress:
  total_phases: 2
  completed_phases: 1
  total_plans: 4
  completed_plans: 4
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md

**Core value:** 用简单、正确、可运行的代码实现 GRACE，随后做最小证伪。
**Current focus:** Phase 2 — 服务器最小证伪（尚未开始）。最后一次推送是 GitHub `master`（`bd5f64f`）；现役工作树还有未提交改动。

## Current Position

Phase: 1 of 2 complete（完整实现 GRACE 与实验代码）
Plan: 4 of 4
Status: Phase 1 code complete
Last activity: 2026-09-16 — 格式 SFT 只训推理开头；审计认最后一行交卷、截断短前缀保留 64/128、续写 finish_reason 按次记录；vLLM `max_model_len` 留 64 token 余量。本机 pytest 214 passed / 9 deselected。未提交。

Progress: 50% — 代码阶段完成；真实 GPU 实验未跑。

## Accumulated Context

### Decisions

- GSD 只记录 4 个短计划，不加审批或研究合同。
- CPU 与 GPU 共用 `grace_gc.core` 数学函数。
- GPU 缺失时明确 ImportError，不提供假后端。

### Next Work

按 README 全流程在目标 GPU 上单卡起步（`a100_1.yaml` 或 `rtx5090_1.yaml`）：先 smoke Full-PG，再 Pilot Full-PG，再 GRACE。不要用 4 卡/8 卡配置。RUN-01 到 RUN-03 仍未跑。

### Notes

- U6/真实两阶段 token 分布待 GPU。
- 不要把本机 tiny LoRA 数字当成实测。
- 2026-09-16 两次服务器 smoke 都不能当 Phase 2：第一次冲突金标硬失败且无格式 SFT；第二次把 `Answer:`+金标+EOS 训进 SFT，5–9 token 交卷。现役树已改成只训推理开头，需按 README 重跑。达标先看 `mean_response_tokens` 不常 <16、审计看得到 64/128、`n_continued` 非整批 0。
