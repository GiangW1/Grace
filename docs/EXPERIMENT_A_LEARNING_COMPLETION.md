# 实验 A：总体期望更新先于答案完成

本实验只新增文件：`grace_gc/audit/learning_completion.py`、`scripts/analyze_learning_completion.py`、`configs/experiments/learning_completion_overlay.yaml` 和 `tests/test_learning_completion.py`。audit、replay、训练、奖励和已有分析脚本都不改；已有 run 和结果不受影响。

## 要检验的主张

对决策位置 t 的每个仍在生成的前缀 h（题目 x），在二值奖励、后缀按策略采样时：

```
mu(h) = E[G | h] = P(h) + c(h),     P(h) = (q(h) - b) g_h
```

`P(h)` 是写完 h 时就已确定的期望更新（只差一个标量 q(h)），`c(h)` 是后缀还要贡献的期望方向。PR11–PR13 测到逐前缀 `c` 很小，但逐前缀能量主要来自 `P(h)`，而 `g_h` 在策略下均值为零，这些大项会在题目之间相互抵消。真正驱动学习的是总体更新 `grad J = E_x S_x`，其中 `S_x` 是题目 x 在位置 t 上所有存活前缀的 `mu(h)` 之和。所以逐前缀“c 小”不等于总体更新已由前缀决定，各题的小 `c` 可能同向累加。实验 A 直接测总体层面的量。

主假设：存在一段 t，总体更新的残差份额 `residual_L(t)` 明显小于答案方差的残差份额 `rho_A(t)`。也就是在答案仍有大量不确定性时，总体期望更新已经由前缀确定（到标量 q(h) 为止）。

两个曲线的端点一致：t=0 时 `g_h=0`，`residual_L=1`、`rho_A=1`；轨迹结束时两者都为 0。因此比较的是“哪个先降下来”。

## 统计量

记 `U_ab` 为不同题目有序对 `<S^a_x, S^b_y>`（x≠y）的平均，它无偏估计 `<E S^a, E S^b>`。记 `D_ab` 为同一题目 `<S^a_x, S^b_x>` 的平均；同一前缀的自项总是用相互独立的 A/B 两半交叉配对。

| 字段 | 定义 | 含义 |
|---|---|---|
| `F_pop` | `U_PM / U_MM` | 总体更新在前缀确定部分上的投影份额 |
| `residual_pop` | `(U_MM - 2U_PM + U_PP) / U_MM` | 用 `E P` 近似 `grad J` 的相对平方误差，即 `‖E c‖²/‖grad J‖²` |
| `cos_pop` | `U_PM / sqrt(U_PP U_MM)` | 方向一致性 |
| `rho_A` | `Σ_h Var(R\|h) / Σ_x k_x Var(R\|x)` | 存活前缀上尚未决定的答案方差份额 |
| `gap` | `rho_A - residual_pop` | 主统计量；为正表示更新比答案更早确定 |
| `F_batch_m` | `(D_PM+(m-1)U_PM)/(D_MM+(m-1)U_MM)` | m 道题的一个 batch 的同一份额；m=1 即单题 |
| `F_prefix` | `Σ<P,mu> / Σ‖mu‖²`（逐前缀） | PR11–PR13 的口径，用于对照 |
| `aggregation_survival` | `U_MM / D_MM` | 单题更新能量在聚合后剩下的比例，反映总体信号强度 |

`Var(R|h)` 用二值奖励计数的无偏式 `s(n-s)/(n(n-1))`；`Var(R|x)` 用同题不同前缀的交叉乘积估计 `(E q)²`。为了让两条曲线在同一批前缀上计算，每组只保留该组在该位置至少有 2 个前缀的题目；被排除的题数和行数写在 `excluded_single_row_problems` 和 `excluded_rows`。

Bootstrap 以题目为单位，所有位置和分组共用同一组重抽样，所以可以比较跨 t 的穿越时间。题目被重复抽到时不与自身配对，因此 U 统计量不会混入单题能量。小样本下 U 统计量可以为负，此时比值记为 `null`，`bootstrap_defined` 给出有定义的次数。

## 分组

- `all`：replay 中全部存活前缀，用两半合并估计。
- `text_pre_answer`：`answer_emitted` 为 false 且未结束。默认 replay 已排除已发答案的前缀，此时与 `all` 相同；用 `--include-answer-emitted` 重放时两者会不同。
- `functional_pre_answer`：在上一组中再要求 `functional_recoverable` 为 false。replay 行不带这个字段，需要用 `--probe-rescores` 传入含 `problem_id, path_id, t, functional_recoverable` 的 JSONL。
- `undecided_half_a`：用 A 半的奖励选出 `0 < q̂_A < 1` 的前缀，所有估计只用 B 半，避免按结果选样带来的偏差。这一组没有同前缀自项，所以只报告总体量和 `rho_A`。

这些筛选只用于分析报告，不是继续实验的条件。

## 前提

1. 奖励必须是二值，而且对 thinking 输出要正确判分。PR13 记录的两个问题仍在现有代码里：原始奖励不看截断，可能抽到 `<think>` 内的答案；功能 probe 判分漏掉了预填的 `Answer:`。本 PR 不修改 `reward.py` 和 probe，以免影响其他实验。所以 thinking 模式的实验 A 需要先单独修好这两处；非 thinking 的 Base 模型可以直接跑。
2. 生成续写的策略要和求 score 梯度的策略一致（完整 softmax、同一 checkpoint）。否则 `E[g_s|h] ≠ 0`，`c(h)` 会混入采样偏差。
3. baseline 在每个前缀内保持不变（audit 默认使用独立 prescan；也可以传 `--baseline-file`）。
4. `--epsilon` 和 `--delta` 要在看结果之前固定。下面的 0.1 和 0.5 只是示例。

## 运行

### 1. Audit（多个决策位置）

把 overlay 叠在所研究模型的 audit 配置之后。下面以 Base 为例，thinking 模型则改为叠在 `reasoning_mechanism_audit.yaml` 之后：

```bash
python scripts/audit.py --generate --data-path DATA.jsonl --model-path "$MODEL" \
  --checkpoint "$CKPT" --config configs/default.yaml \
  --config configs/experiments/learning_completion_overlay.yaml \
  --run-dir runs/expA-audit
```

overlay 默认设置是 96 题 × 4 个前缀 × 8 条续写，决策位置为 `[256, 512, 1024, 2048, 4096]`；超过 `max_new_tokens` 的位置会被 audit 丢弃并记录。总体统计量的精度主要取决于题目数，单个前缀的续写条数次要，所以预算优先用来增加题目数。

### 2. 每个位置做一次稠密 replay

实验 A 需要跨前缀的内积，所以不能用 `--statistics-only`。replay 会读取 audit 目录里保存的 `config.yaml`：

```bash
for T in 256 512 1024 2048 4096; do
  python scripts/replay_expected_gain.py --bundles runs/expA-audit \
    --checkpoint "$CKPT" --model-path "$MODEL" \
    --decision-tokens $T --max-continuations 8 --skip-prompt-features \
    --store-half-means --store-trajectory-decomposition \
    --run-dir runs/expA-replay-t$T
done
```

### 3. 分析（只用 CPU）

```bash
python scripts/analyze_learning_completion.py \
  --replay-dir runs/expA-replay-t256 runs/expA-replay-t512 runs/expA-replay-t1024 \
               runs/expA-replay-t2048 runs/expA-replay-t4096 \
  --gram-cache runs/expA-gram \
  --epsilon 0.1 --delta 0.5 --bootstrap 1000 \
  --run-dir runs/expA-analysis
```

加 `--metric-file RUN/adam_metric.npy --metric-name adam_diagonal` 可以在固定对角度量下重复分析；度量只作用于内积，不需要重新 replay。thinking 模型的功能分组需要加 `--probe-rescores <修正后的 functional_probe_rescores.jsonl>`。结果写在 `learning_completion_summary.json`，终端打印每组每个位置的 `F_pop / residual / rho_A / gap / F_prefix`。

## 磁盘

D = 5,898,240 时，FP64 每个向量约 47 MB。实验 A 每个前缀读 3 个向量（`prefix_score_gradients.npy`、`half_mean_full_grads_a/b.npy`），约 142 MB。replay 还会写出实验 A 不用的 `mean_grads.npy` 和 `half_mean_grads_a/b.npy`，可以在 replay 结束后删除。

各位置之间互不配对，所以可以逐个位置处理：replay 一个位置，用 `--gram-cache` 跑一次分析得到该位置的 Gram（3n×3n，很小），然后删除该位置的大数组，再处理下一个位置。最后一次分析只需要各目录的 `prefixes.jsonl` 和 Gram 缓存。缓存的指纹包括前缀键、`prefixes.jsonl` 的 hash 和度量 hash；输入变化时会自动重算。同一位置的全部行必须同时在盘上，因为跨分片的题目对也要计算。

如果 PR12 的稠密 trajectory replay 仍在服务器上，可以直接用它做一次单位置分析，看 `aggregation_survival` 和 `U_MM` 的置信区间，用来估计正式实验需要多少题。

## 读法

- 支持主张：在若干 t 上，`gap` 的题目 bootstrap 区间整体大于 0，同时 `F_pop` 接近 1、`rho_A` 仍明显大于 0。`crossings` 给出预注册阈值下的 `t_L` 和 `t_A`，以及 bootstrap 中 `t_L < t_A` 的比例。
- 不支持：`residual_pop` 和 `rho_A` 一起下降（`gap` 约为 0）。这说明更新的确定和答案的确定同步，前缀价值 q(h) 已经足以解释一切，与 ARRoL 一类 q-only 方法没有区别。
- `F_prefix` 高而 `F_pop` 低，说明各题的 `c` 在总体上同向累加。这是 PR11–PR13 的逐前缀测量回答不了的情况，也正是本实验要区分的情况。
- `U_MM` 的区间覆盖 0 时，说明这个题目数下总体更新本身还测不准（看 `aggregation_survival`）。结果照常输出；需要更多题目，而不是改指标。

## 限制

- 只覆盖在 t 仍在生成的前缀；已结束的轨迹完全确定，不参与两条曲线。两条曲线用的是同一批前缀。
- 每组要求同题至少 2 个前缀，否则 `Var(R|x)` 无法无偏估计。
- 这里的“确定”是指到标量 q(h) 为止的期望方向；q(h) 本身是否可以低成本预测，是另一个问题（方法部分的预测器）。
