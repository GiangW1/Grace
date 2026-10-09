# 分析论文框架 v4（ACL 2027，替代 ICLR v3 的方法论文叙事）

2026-10-09 审查后转向。v3（`GRACE_ICLR论文框架_v3.md`）保留为历史，其中的 GRACE 在线方法、0.65× 算力等目标不再是本文主张。ACL 走 ARR，长文 8 页正文；具体截止日期投稿前核实。

## 0. 一句话与题目

在 RLVR 里，一条轨迹会给参数带来什么更新，基本上由它的前缀价值 q(h) 决定。因此"更新的不确定性"和"答案的不确定性"同步消退，无偏的前缀截断/梯度补全在原理上省不了多少算力；真正能省算力的信号只有结果信息。

候选题目：*Learning Does Not Finish Before Reasoning: Why Unbiased Early Stopping Cannot Save RLVR Compute*（备选：*The Update Is the Value: Prefix Value Determines What an RLVR Trajectory Teaches*）。

## 1. 主张与对应测量

| 编号 | 主张 | 测量（`synchrony_summary.json` 字段） |
|---|---|---|
| C1 | 前缀价值足够：μ(h)=E[G\|h]≈(q(h)−b)·g_h，后缀交叉项 c(h) 只占 μ 能量的小部分 | `suffix_cross_share`；`residual_reward_only` 与 `residual_free_coefficient`、`rho_L`（完整 oracle）的差距 |
| C2 | 同步消退：全参数空间的 ρ_L(t) 不低于 ρ_A(t)，即没有 LAG | `rho_L`、`rho_A`、`gap` 及题目层 bootstrap 区间；按 b 分层与 `pre_answer` 子集 |
| C3 | 无偏截断有上限：方差×成本比 ≥ (√(αγ)+√((1−α)(1−γ)))²（α≤γ 时为 1） | `alpha_upper`、`allocation.*.gamma`、`homogeneous_bound`、`oracle_in_sample`（含异质性的乐观天花板） |
| C4 | 能省的只来自结果信息：只用 q 的分配拿到 oracle 天花板的大部分；省下的算力集中在答案已定的前缀 | `q_only_reward_cv` 对 `oracle_in_sample`；`settled_fraction`、`settled_residual_share` |

附带检验：鞅差正交性 ⟨g_s, g_h⟩≈0（`mean_suffix_prefix_cosine`），这是 C2 推导用到的唯一结构性质。

## 2. 理论（正文 §3，证明进附录）

- **分解（精确）**：G=(R−b)(g_h+g_s)。在策略下 E[g_s\|h]=0，且不同 token 的 score 增量是鞅差，所以 E‖g_s‖² 等于逐 token 能量之和，E⟨g_h,g_s⟩=0。由此 μ(h)=(q(h)−b)g_h+E[(R−q)g_s\|h]（Lemma 2 原式）。
- **命题 1（同步）**：若后缀 score 能量与 R 的相关性可以忽略（只有稀疏的决策 token 违反，Lemma 1 正是描述这些 token），则 tr Cov(G\|h) ≈ Var(R\|h)‖g_h‖² + E[(R−b)²\|h]·E‖g_s‖²。对 h 取平均后第二项不会缩小（全方差公式），所以 ρ_L(t) ≳ ρ_A(t)。Lemma 1 的 (1−2q)² 衰减只作用于决策 logit 这一方向，占全空间能量的比例可忽略。
- **命题 2（上限）**：对任意预测 m 和续写概率 p，HT 估计量的方差×成本 ≥ (√(c₀A)+E√(rc))²，其中 A=V−E r，r 至少是 oracle 残差 tr Cov(G\|h)。同质情形化简为上面的闭式；要达到 0.65× 需要 α≈0.87。
- **推论（分配退化）**：在命题 1 的近似下，r(h)/c(h) 只依赖 q(h)、b 和剩余长度，所以 Neyman 最优分配是一个"结果预测"分配（GRACE ≡ Reward-CV）。

所有估计量都用 A/B 两半续写交叉配对，保证无偏；同题不同前缀的交叉内积给出 ‖E[G\|x]‖²。合成数据上的精确枚举检验见 `tests/test_synchrony.py`。

## 3. 实验设计

**模型与快照**：Qwen3-4B-Base，q/v LoRA r=16，生成上限 2048 token，T=1、不截断 top-p。快照取格式 warmup 结束（step 0，B 已非零）、step 50、step 150，来自同一条 Dr. GRPO 训练（N=128=16 题×8 起步，固定 N）。规模对比加 Qwen3-1.7B-Base 的 step 0 和后期快照。thinking 模型作为附录可选。

**审计**：论文 audit split 的 240 题。每题先用 16 次独立采样估 b(x)；16 次全对或全错的题跳过并记录（无答案方差，b∈{0,1} 时 G 基本为 0）。其余题每题 4 条 iid 前缀，决策点 {128,256,512,1024}，每个前缀 16 条续写（A/B 各 8）。生成阶段只记 token 与奖励（`tokens_only`），梯度由 streaming replay 只算一次，只存标量。

**统计**：曲线取跨题的和之比，题目是 bootstrap 单位；按 b 分层（hard<0.4≤medium<0.7≤easy）；同时报告全部存活前缀和 pre-answer 子集。成本权重 w_train（训练一 token 相对生成一 token）报告 {0.5,1,2} 三档，并用实测吞吐标定一档。

**不再做的**：GRACE 在线训练对比、预测器/基底、总体层面的实验 A、PR15 后缀传递。独立起步下批梯度方差就是单起步方差除以 N，单起步层面的方差×成本就是正确指标。

## 4. 运行命令（服务器）

```bash
# 1) 快照训练（约 1–2 A100-天；checkpoints/step_{0,50,100,150}.npz）
python scripts/train.py --config configs/default.yaml \
  --config configs/experiments/analysis_snapshot_train.yaml --config configs/hardware/a100_1.yaml \
  --model-path "$MODEL" --data-path "$DAPO" --eval-data-path "$MATH500" --run-dir runs/analysis-snap

# 2) 每个快照一次审计生成（只采样）
python scripts/audit.py --generate --backend gpu_verl --method full_pg \
  --config configs/default.yaml --config configs/experiments/analysis_snapshot_train.yaml \
  --config configs/experiments/analysis_audit.yaml --config configs/hardware/a100_1.yaml \
  --checkpoint runs/analysis-snap/checkpoints/step_50.npz --model-path "$MODEL" --data-path "$DAPO" \
  --run-dir runs/analysis-audit-s50

# 3) 每个决策点一次流式重放（只存标量；包含已给答案的前缀，分析时再分组）
for T in 128 256 512 1024; do
  python scripts/replay_expected_gain.py --bundles runs/analysis-audit-s50 \
    --checkpoint runs/analysis-snap/checkpoints/step_50.npz --model-path "$MODEL" \
    --decision-tokens $T --max-continuations 16 --statistics-only --include-answer-emitted \
    --skip-prompt-features --run-dir runs/analysis-replay-s50-t$T
done

# 4) CPU 分析
python scripts/analyze_synchrony.py --replay-dir runs/analysis-replay-s50-t{128,256,512,1024} \
  --bootstrap 1000 --run-dir runs/analysis-sync-s50
```

审计只用 DAPO 的 audit split（默认 240 题，按 split seed 固定），不涉及 MATH-500。

## 5. 算力（粗估，冒烟后按实测修正）

快照训练 1–2 A100-天；每个快照的审计加重放约 1 A100-天（生成只采样，梯度只算一次），三个快照约 3 天；1.7B 约 1 天。合计 5–7 A100-天。先跑冒烟：20 题、1 个快照、t=512，用来测吞吐和跳题比例。

## 6. 风险与应对

| 风险 | 信号 | 应对 |
|---|---|---|
| 审稿人认为结论显然 | — | 给出精确上限与适用条件；实测连 oracle 也赚不到；点名受约束的方法族（GPCV 型补全、Randomized Telescoping 型截断） |
| 真的测到 LAG（gap 显著为正） | `gap` 区间在 0 以上 | 如实报告：结论改为"存在但收益受 α、γ 限制"，C3 的上限仍然成立 |
| 跳题后可用题太少 | 冒烟时跳题比例超过一半 | 加大 audit split（`split_records` 的 n_audit）或从 calib split 补题；不放宽统计定义 |
| 成本权重有争议 | — | 报告三档 w_train 并用实测吞吐标定；上限对 γ 单调，结论方向不依赖具体权重 |
| 单一模型族 | — | 1.7B 规模对比；thinking 长 CoT 进附录 |

## 7. 代码落点

- `grace_gc/audit/synchrony.py`、`scripts/analyze_synchrony.py`：C1–C4 的估计量、上限与分配天花板。
- `grace_gc/audit/streaming_replay.py`：新增 `problem_cross.jsonl`（同题跨前缀交叉内积）。
- `grace_gc/audit/run.py`：`audit.tokens_only`、`audit.skip_settled_problems`。
- `configs/experiments/analysis_snapshot_train.yaml`、`analysis_audit.yaml`。
- 同轮修复（影响旧训练路径）：截断回答一律 R=0（奖励协议 v4）；HT 基线截断到 [0,1]；token 记账按实际后缀；自锚定墙钟目标不再把 N 压到配置值以下。
