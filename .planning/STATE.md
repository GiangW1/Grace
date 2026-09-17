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
**Current focus:** Phase 2 — 服务器最小证伪尚未用现役树重跑。GitHub 默认分支是 `master`。

## Current Position

Phase: 1 of 2 complete（完整实现 GRACE 与实验代码）
Plan: 4 of 4
Status: Phase 1 code complete；保真/方差接线在 master
Last activity: 2026-09-17 — Neyman 等真实 U 与同步头、||G−Uf||²、仿射/ridge/γ/换基对齐、剩余 token ĉ；审查后补同步门、审计就绪、γ 的 IPW、稳定去均值、缺失 logprob、λ 换算、账本不重叠。本机 `252 passed, 17 deselected`。PR #2/#3 的最小 GPU 脚本和 π 计分已并入这份，不再单独合那两个 PR。

Progress: 50% — 代码阶段完成；现役树上的真实 GPU 实验未跑。

## Accumulated Context

### Decisions

- GSD 只记录 4 个短计划，不加审批或研究合同。
- CPU 与 GPU 共用 `grace_gc.core` 数学函数。
- GPU 缺失时明确 ImportError，不提供假后端。

### Next Work

把当前工作树放到单卡服务器，跑 `scripts/run_minimal_gpu.sh`。看 GRACE 相对 Full-PG / Uniform-CV / GRPO 的质量与成本，不要看停止者比例。不要用 4 卡/8 卡。不要把 2026-09-16 两次 smoke 或那次 16 题链当现役结果。Pilot 仍按 README，在这次最小比较可读之后。

### Notes

- U6/真实两阶段 token 分布待 GPU。
- 不要把本机 tiny LoRA 数字当成实测。
- 2026-09-16 两次 smoke：冲突金标硬失败；以及 SFT 训进 `Answer:`+金标+EOS 导致短答交卷。`37fbcb2` 已改为只训推理开头。
