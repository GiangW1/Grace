# GRACE fidelity

本文件记录 2026-09-17 `a1ca77b` 的保真/方差接线，供追溯历史设计。后续已修改成本控制和监督；下文当时的取舍（包括末节的墙钟限制）不作为当前规则。当前边界见 [AGENTS.md](../../AGENTS.md)，后续实现见 [问题清单](../../docs/GRACE_ISSUE_CHECKLIST_20260919.md)。

当时修正论文量算和残差风险 Neyman 的方差项；保持 R−b，停止者 reward=null。接线完成不代表分配或补全已获得收益。

## P0 保真

- F01：`use_allocation` 方法在 `basis_id=0` 或 heads 未同步到当前 U 时保持 p=z=1。
- F03：换基后回写 reservoir `basis_id`；有效秩 < k 时零填充；秩 0 不换基。
- F05：残差用 ||G−Uf||²，非有限值报错。
- F07：λ 在 r/mean(r) 上求解，风险整体倍乘不改 p。
- F09：prescan 用 `IsolatedRNG(seed+1)`，跨步复用并写入 checkpoint。
- F11：生成时尽量记 vLLM sampled logprob；不作 TIS。
- F16：单位兼容只认数字核，例如 5 vs 5cm，不认 x vs xy。
- F18：审计 actual 在无预测器时用 m=0。
- F20：同时报 Z-proxy token、完成轨迹实际 token、墙钟账本。
- F29：清单写明 GRPO 是组均值、无标准差。

## P1 方差（与 Uniform-CV 共享；GRACE 另有风险分配）

- E01：可逆输入仿射 + 坐标/风险同单位。只改预测器内部尺度，不改存盘 φ。历史 fit 统计，本 actor 批冻结，HT/风险前还原。
- E04：换基后按 |内积| 对齐列并统一符号；`basis_id==0` 的第一次 SVD 不对齐到 identity。
- E06：历史 γ 只乘在 m 上，γ*=E[a⟨G,m⟩]/E[a||m||²]，a=1/p−1。两流同一 γ。风险用 ||G−γUf||²。γ 裁到 [0,2] 只是数值篱笆。
- E07：`coord_kind=mlp|ridge|linear`（linear 即 ridge）。论文/default 仍是 mlp；`minimal_gpu.yaml` 用 ridge（小 reservoir）。
- E08 lite：`constant_cost` 时 ĉ 是剩余 token，不是 cost MLP。不改 next_n 的 Z-proxy。

## 审查后接线

- 换基后先作废 `predictor_synced_basis_id`；只有坐标头和该方法风险头都在当前 U 上训过才恢复。
- 审计复用同一就绪门；常数成本用 remaining，不用 `pred.c_hat=1`。
- γ 的分子分母乘 reservoir IPW `1/(p s)`。
- 题内去均值先减第一行再对差值去均值，避免重复 float 梯度伪秩。
- 任一必要 sampled logprob 缺失则整段记缺失；返回 λ 为 `lam * scale`；账本总量不把 phase/train_step 和 train 加在一起。

## 拒绝

- 默认 TIS / mask replay / DAPO clip-higher。
- 截断奖励全局置 0（与 GRPO-short 合同冲突）。
- 新增 STREAMS、Go/No-Go、GRACE 专属 SFT。
- LayerNorm 作默认尺度（E03）。
- 用 planned-p 或墙钟改 next_n。
