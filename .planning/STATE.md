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
**Current focus:** Phase 2 — 服务器最小证伪（尚未开始）。最后一次推送是 GitHub `master`（712c9e6）；现役工作树还有未提交改动。

## Current Position

Phase: 1 of 2 complete（完整实现 GRACE 与实验代码）
Plan: 4 of 4
Status: Phase 1 code complete
Last activity: 2026-09-16 — 官方 DAPO 冲突丢掉、共享格式 SFT、奖励 `\text{}`/`\pi` 归一、GPU 阶段心跳与 startup.json、health 记 baseline/A-vs-B、README 全流程。本机 pytest 199 passed / 1 skipped。未提交。

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
- 2026-09-16 服务器 smoke 产物不能当 Phase 2 结果：当时官方 parquet 会因冲突金标硬失败，且 Base+ChatML 无格式 SFT。代码已改，需按 README 全流程重跑。
