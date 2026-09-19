---
gsd_state_version: '1.0'
status: in-progress
---

# Project State

核对日期：2026-09-19。原两阶段的首轮实现和最小证伪已完成；目前按负结果继续修复与再验证，论文效果尚未成立。项目范围见 [PROJECT.md](PROJECT.md)。

## 代码与发布

- 仓库：[GiangW1/Grace](https://github.com/GiangW1/Grace)，默认分支 `master`。
- 本地分支：`codex/grace-efficiency-20260919`；本轮功能提交截至 `f20506e`。后续知识收尾仅同步文档和忽略规则。
- 修复已通过 fork 提交 [PR #2](https://github.com/GiangW1/Grace/pull/2)，核对时为 OPEN、未合并；未观察到 CI 检查结果。不能把 PR 发布当作服务器已部署或实测。
- 后端为单卡 HF actor + vLLM 两阶段；固定 U 是可选变体，预测头仍在线更新。`n_gpu>1` 继续拒绝。

## 实验与测试

- 最新已复核真实实验为 `minimal-chain-20260918-063159`，源码 `3c03ce9`，四方法均完成训练、评测及审计；[结果复核](../docs/MINIMAL_RESULTS_REVIEW_20260919.md)仍不支持“补全提高实际训练效率”。
- 本轮修复没有新的 GPU 结果。本机无 GPU，最近 CPU 回归及真实 CUDA 跳过项以 [TEST_STATUS.md](../docs/TEST_STATUS.md) 为准。
- 9月16/17日实验保留为历史，不混入9月18日链，也不作为当前候选的验证。

## 下一步

1. 在单卡服务器验证当前 PR 版本的生成/缓存/同步观测、CUDA 数值、固定 U 恢复和完整成本，再使用现有脚本比较机制与同预算质量。运行方式见 [README](../README.md)。
2. [实施方案§9](../docs/GRACE_RESEARCH_IMPLEMENTATION_PLAN_20260919.md#9-代码落点与验证)的顺序 1–4 已实现；5–7（条件噪声/基底遗漏/拟合误差与交叉能量、可选 prequential/轻特征、后续多 GPU）尚未实现。已有 realized-G 分解不能替代第5项。
3. [30项问题清单](../docs/GRACE_ISSUE_CHECKLIST_20260919.md)是问题状态的权威入口；没有新测量时，不把待实测或未闭合事项改成已解决。

## 现场保留

本地 `_minimal_review_*/`、`_smoke_review_*/`、`_runs_results/` 及 `.planning/phases/` 的历史材料保留，不提交、不清场。PR 尚未合并，当前分支和 worktree 仍用于复核。生成记忆未手工修改。
