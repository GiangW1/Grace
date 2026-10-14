# 小参数 GRACE：固定训练坐标，直接预测完整小核心梯度

本实现基于 PR #16 的 `6656fb53af616c8d3f5285f6678e9498b80c4a66`，继承其中
PR #12–#15 的实验/审计框架及修复。它是用户要求重新验证 GRACE 的独立、可选实验；
不改写既有分析论文结论，不把旧结果当作本次结果。PR17 后来上传了 20 步 pilot；
它不替代下文的正式对照或预测机制证据。PR18 三卡快速诊断已于 2026-10-14 完成，
见[本轮实测结果](results/pr18-20261014/followup-fast-3gpu/README.md)：三臂各 50 步，128 题 × 4 回答与冻结审计全部完成；
当前结果未显示 GRACE 的预测收益。

## 改了什么

每个 q/v 投影为 `W + (alpha/r) L C R`，仅训练 FP32 的 `C[r,r]`。
`L/R` 为固定非零矩阵，`C=0` 保持初始模型不变。默认取权重 SVD 的左右
正交奇异向量（不吸收奇异值），另有独立随机正交方向消融。
这是 [LoRA-XS](https://arxiv.org/abs/2405.17604) 类参数化的实验变体，
并非该论文的逐项复现，也不把适配器结构作为创新点。

预测目标为全部小核心展平后的 `g_z=(R-b)∇_z log π(完整轨迹)`。
保留原前缀特征，并可拼入当前完整 `z`；不用未来奖励作为预测输入。
`predictor.target_space: direct` 会覆盖旧 `k=8`，不做 PCA、不刷新梯度基底，
也不分配 D×D 单位矩阵。优化器、梯度收集、审计布局全部只识别可训练核心。

校正仍是 `m + Z/p (g_z-m)`，使用原始起步数 N、原有梯度上升/下降符号约定、
批内冻结和前缀选择。它的原始估计期望只针对**当前小核心参数化**；
不能称为普通 LoRA 的等价更新，也不能将无偏保证扩展到梯度裁剪和 Adam 后的更新。

冻结基底和核心一并存入训练检查点，恢复/初始化/评测/审计/streaming replay 会加载
实际保存的基底。快照哈希包含固定几何。vLLM 导出使用 `A=R, B=L@C`，沿用
标准 LoRA 加载、adapter id 和 prefix-cache reset；GPU 上仍须检查行为 logprob probe。
FP32 代数等价不意味着不同 BF16 运算顺序逐 bit 相同。

## 对照与配置

| 参数化 | Full-PG | GRACE | 额外对照 |
|---|---|---|---|
| 普通 LoRA rank 16 | `lora16_full_pg` | `lora16_grace` | 原 PCA 路径 |
| 小核心 rank 8 | `core8_full_pg` | `core8_grace` | `core8_uniform_ht` |
| 小核心 rank 16 | `core16_full_pg` | `core16_grace` | `core16_uniform_ht` |

若模型恰为 36 层、每层 q/v 两模块，小核心维数分别为 4,608、18,432。
实际数目从 live actor 枚举，见 `startup.json` 的 `trainable_dim`、`prediction_dim`，
不能用这个算术示例代替真实检查点的参数清单。

共享默认协议：显式指定同一 Qwen3-4B（非 Base）checkpoint，关闭 Thinking
（`enable_thinking: false`）；DAPO 训练、独立评测集、
8192 token、512-token 决策点、32 starts / 8 prompts、固定 baseline=0.5、
无格式 SFT、无动态补位、固定 N。训练采样温度 1，沿用截断奖励为 0 的协议。
Full-PG 与 GRACE 都从各自参数化的零适配器开始；GRACE 前 10 步全续写预热。
标签来自原 mandatory backward，`audit_s=1`；标签保存、预测器拟合和同步仍有真实开销。

## 服务器运行

交付的源码压缩包已包含本次改动（不含 Git 历史、模型和大型历史实验归档），可解压到新目录运行。
若使用补丁，在干净的 PR #16 基础上应用，不要直接往旧 master 或有未保存修改的目录里覆盖：

```bash
git clone https://github.com/GiangW1/Grace.git Grace-small-core
cd Grace-small-core
git fetch origin pull/16/head
git switch -c experiment/small-core-gradient 6656fb53af616c8d3f5285f6678e9498b80c4a66
git apply --check /path/to/grace-small-core-nonthinking.patch
git apply /path/to/grace-small-core-nonthinking.patch
```

先将代码放到已有 GPU 环境，沿 README 安装依赖；这里没有安装/升级服务器 CUDA 栈。
不自动下载模型，务必让 `$MODEL` 指向 Qwen3-4B（非 Base）权重目录。
默认使用非 Thinking 聊天模式；训练、评测和审计继承同一配置。
关闭 Thinking 不等于换成 Base，也不会把 8192-token 输出上限缩短。
以下命令在仓库根目录运行，`$TRAIN_DATA/$EVAL_DATA` 为已有真实数据路径。

```bash
python -m pytest tests/test_small_core.py -q

# 先小核心 Full-PG 链路 smoke；4 步，保留 8192 长度，不是效果实验。
python scripts/run_small_core_experiment.py \
  --model-path "$MODEL" --data-path "$TRAIN_DATA" --eval-data-path "$EVAL_DATA" \
  --arms core8_full_pg --seeds 17 --smoke --execute \
  --run-dir runs/core8-smoke

# 最小效果对照：四格、一种子；默认 100 步。去掉 --execute 仅生成配置和命令。
python scripts/run_small_core_experiment.py \
  --model-path "$MODEL" --data-path "$TRAIN_DATA" --eval-data-path "$EVAL_DATA" \
  --arms lora16_full_pg lora16_grace core8_full_pg core8_grace \
  --seeds 17 --execute --run-dir runs/core8-four-way

# 全部六臂 × 三种子；同一张卡顺序运行，不并发占卡。
python scripts/run_small_core_experiment.py \
  --model-path "$MODEL" --data-path "$TRAIN_DATA" --eval-data-path "$EVAL_DATA" \
  --seeds 17 23 42 --execute --run-dir runs/core-six-way
```

每个 arm 自动调用原 `train.py` 和 `evaluate.py`。可加 `--initial-eval` 自动对首个训练作业的
`checkpoints/step_0.npz` 评测一次，建立共享未训练基线；不能把单个最终准确率当作能力提升。
共享基线仅适用于相同预训练权重、零适配器、无格式 SFT/初始化检查点/续训。
所有评测显式使用共同的 `--eval-seed`（默认 17），与训练 seed 分离。
同名 suite 会另建目录，不覆盖旧结果。`suite.json` 保存精确命令、阶段状态和实际整条命令耗时。

统一覆盖文件通过 `--config my_override.yaml` 传入；例如改变 `num_steps`、学习率、
裁剪阈值或 `eval.n_problems`。rank/参数化/direct 由 arm 设置，
随机基底用 `--core-basis random`，移除当前 z 输入用 `--without-actor-state`。
普通 LoRA 不拼接百万维 z，所以必须报告这个输入消融，不能把全套改变的收益只归因于维度。

同墙钟对照可加 `--run-wall-seconds 7200`，但同时要用覆盖文件把 `num_steps` 设得足够大，
检查每臂是否实际因墙钟预算停止而非提前碰到步数上限。保留完整命令计时和原成本账本；
SVD 初始化、预测器拟合、检查点、权重同步都不能从收益成本中删去。

## 重复续写审计

`--audit` 会在每臂训练/评测后额外运行付费审计：16 题、2 个前缀、每前缀 16 个独立续写，
决策点 512/1024，保存完整 trajectory gradients，不使用 JL 压缩，也不排除“已解决”题。
普通 LoRA 的完整梯度文件很大；小规模 smoke 只减题数和续写数，不改变目标定义。
注意旧 `small_core_audit.yaml` 的 `predictor_diagnostic: true` 会关闭已训练预测器，
主要用于收集离线标签。不能把缺失预测的统计当作当前预测器误差；下一节使用 false。

也可以针对已经完成的单个检查点运行，避免重训：

```bash
python scripts/audit.py --generate \
  --config runs/core8-four-way/core8_grace-seed17.yaml \
  --config configs/experiments/small_core_audit.yaml \
  --model-path "$MODEL" --data-path "$TRAIN_DATA" \
  --checkpoint runs/core8-four-way/core8_grace-seed17/checkpoint.npz \
  --run-dir runs/core8-grace-audit
```

沿用 PR 的 `audit_summary.json`、完整轨迹梯度和预测残差。先看同参数化内
GRACE 相对 Full-PG 的准确率/成本，再比较小核心 Full-PG 与普通 LoRA Full-PG 的能力。
训练日志 `predictor.frozen_new_problems.residual_to_m0_ratio` 是更新预测器之前的
新问题诊断；拟合集 loss 降低不是泛化收益。报告同前缀续写波动、正确样本子集、
原始梯度与实际优化器更新、裁剪比例，不能跨参数化直接比较未校准的梯度范数。

至少补做：Uniform-HT、移除 z、随机固定基底、rank 8/16、学习率敏感性。
相同学习率不是等效步长；应在独立验证集以同等调参预算选择各参数化学习率，
再比较测试集和等墙钟结果。前缀噪声/预测误差的改善与表达能力保留是两个独立问题。

## PR18 快速诊断：三卡5090（2026-10-14）

新增 `--followup` 仍调用原训练、评测、分卡审计入口，不改策略梯度/HT 数学。
默认三臂：core8 Full-PG、core8 Uniform-HT、core8 GRACE；只运行训练种子 17。
每臂 50 步，其中 10 步 warmup；保存 step-0 与第 50 步，不默认评测中间快照。
使用相同 MATH-500 子集 128 题、每题四次、相同题序与评测采样种子。
默认硬件 `rtx5090_rollout_3.yaml`：第一张可见卡放 actor，其余两张各运行独立 TP=1
vLLM worker；训练、评测和审计继承同一布局。每卡仍是 32GB，actor 不分片。
先运行 Full-PG。共享 step-0 只评测一次，不重复支付每臂初始评测；它在首个训练作业
完成后读取保存的 step-0，绝不拿最终检查点充当基线。零适配器在模型函数上相同，
但 BF16 内核/采样仍存在数值波动，共享一次基线不是对这类波动的估计。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
export MODEL=/path/to/Qwen3-4B
export TRAIN_DATA=/path/to/dapo-math.parquet
export EVAL_DATA=/path/to/math500.jsonl

# 默认仅生成计划：检查目录内 YAML 和 suite.json 后再执行。
python scripts/run_small_core_experiment.py --followup \
  --hardware configs/hardware/rtx5090_rollout_3.yaml \
  --model-path "$MODEL" --data-path "$TRAIN_DATA" --eval-data-path "$EVAL_DATA" \
  --run-dir runs/core8-quick-plan

# 先验证新增流程（4步、2题；不能用于效果判断）。会包含初始评测及 frozen audit。
python scripts/run_small_core_experiment.py --followup --smoke --seeds 17 \
  --hardware configs/hardware/rtx5090_rollout_3.yaml \
  --model-path "$MODEL" --data-path "$TRAIN_DATA" --eval-data-path "$EVAL_DATA" \
  --run-dir runs/core8-quick-smoke --execute

# 快速诊断：3个训练作业、4次评测（1次共享初始+3次最终）、1次GRACE审计。
# 不自动运行LoRA16参考、多种子或25/50/75的中间评测。
python scripts/run_small_core_experiment.py --followup --seeds 17 \
  --hardware configs/hardware/rtx5090_rollout_3.yaml \
  --model-path "$MODEL" --data-path "$TRAIN_DATA" --eval-data-path "$EVAL_DATA" \
  --run-dir runs/core8-quick --execute
```

正常诊断计划只有 9 条命令：3 train + 1 initial_eval + 3 eval + 1 audit + 1 CPU prediction_report。
评测计划为 2048 条答案；审计最多 64 条续写及对应梯度，提前自然结束的路径不重复续写。
这只是任务量，不是实际耗时或速度收益；单种子、小审计也不替代正式验证。

`--arms` 可显式加入 `lora16_full_pg` 能力参考，或换成三个 core16 臂；
`--seeds 17 23 42` 可扩展种子；四卡仍可显式选择 `rtx5090_rollout_4.yaml`。
`--config` 可覆盖步数、checkpoint_every、LR、评测数量；中间评测只在显式传入
`--eval-steps` 时执行，例如先覆盖为 100 步/每 25 步保存，再指定 `--eval-steps 25 50 75`。
不因小样本/无信号禁止运行。常规步数实验不得把同 step 当作同成本。
第二轮等墙钟实验可给上述命令加 `--config configs/experiments/small_core_followup_wall.yaml`
并更换 run-dir：每臂训练预算 7200 秒、步数上限 10000，仍固定 N，不自动缩小 batch。
等墙钟时可用已有 `--run-wall-seconds` 配合更大的 `num_steps`；预算提前结束导致的
缺失中间快照会记录为 skipped，最终检查点仍评测。检查是否真的由时间预算停止、
是否超时，不能把提前撞到步数上限的两组称为等墙钟对照。

### 不重训，先审计已有 core8 GRACE 检查点

```bash
python scripts/audit.py --generate \
  --config /path/to/core8_grace-seed17.yaml \
  --config configs/hardware/rtx5090_rollout_3.yaml \
  --config configs/experiments/small_core_audit.yaml \
  --config configs/experiments/small_core_followup_audit.yaml \
  --data-path "$TRAIN_DATA" \
  --checkpoint /path/to/core8_grace-seed17/checkpoint.npz \
  --run-dir runs/core8-existing-frozen-audit
python scripts/analyze_small_core_prediction.py \
  --audit-dir runs/core8-existing-frozen-audit
```

原训练 YAML 必须保留非 Thinking 以及原模型路径；上面明确覆盖为当前三卡布局，
也可审计之前四卡训练产生的同一检查点。审计使用独立 audit split。
新配置加载检查点冻结预测器（包括 shrinkage），不在审计标签上重新拟合；默认只审计
GRACE 最终检查点，4 题 × 2 前缀 × 8 后缀，决策点 512。自然结束的路径不强行延长。
两份 small-core 审计配置的 `audit.baseline: null` 默认继承共享训练配置的固定 baseline，
同时用于梯度标签和预测器输入；不会重新写死为 0.5。若研究目的需要覆盖，最后加载的
YAML 可显式设置 `audit.baseline`。审计 manifest 保留该覆盖值与
`baseline_target_changed_from_configured_fixed`；为 true 时不能把误差解释为原训练目标上的
预测精度。非 fixed 模式仍走已有独立预扫，目标比较标志留 null，不宣称继承历史 baseline。

### 如何读新增输出

- `comparison_summary.json`：初始/各快照成绩、相对初始与同 seed/同 step Full-PG 的逐题配对差值、
  problem-bootstrap 区间、最终跨训练种子均值/标准差。评测清单不匹配就标记不可配对。
  区间仅条件于当前检查点与回答，不代表多种子训练不确定性，也不是等效性检验。
- 同一汇总中的训练诊断：warmup 后停止比例、CV 实际启用步数、m_shrink、阶段耗时。
  同时保留逐步 frozen-new-problem 诊断（更新预测器之前、带历史完整性标志），
  以及训练 stop_reason/成本控制记录；不把这些跨 actor 的逐步 ratio 简单平均成因果收益。
  学习曲线用 `checkpoints.json` 的发布时钟，且要求检查点 hash 一致；缺少记录就留空，
  不拿总命令耗时伪造中途快照时间。完整训练命令耗时含启动，评测/审计单列在 `suite.json`。
- `*-audit/direct_prediction_summary.json`：冻结预测器对新续写完整核心梯度的 MSE/零预测 MSE，
  同前缀续写噪声、减去 `trace(sample covariance)/n` 后的条件均值误差估计。
  负的去噪估计照实保留；缺少预测不当作零预测。直接坐标只需 O(nD) 运算，不构建 D×D Gram。
- 固定 p 下 `(1-p)/p E||G-f||²` 与零预测对照只隔离“随机停止额外带来的方差”，
  不含 Full-PG 自身方差、不含预测计算成本，不等于在线 GRACE 的实际方差或提速。
  表中同时报告所有前缀和可继续前缀，避免自然结束样本稀释差异。
  `fixed_p` 只用于未结束前缀；已结束前缀强制 p_i=1，附加方差为零，即使其梯度非零。
  all-prefix 汇总仍保留其梯度能量与原前缀平均分母，不把结束样本当作可随机停止的样本。

可在部分作业完成后手工重汇总（不触发 GPU）：

```bash
python scripts/summarize_small_core_experiment.py --suite runs/core8-quick/suite.json
```

先问 core8 Full-PG 相比初始有没有学到，再问 GRACE 相比 Uniform-HT 是否有额外收益；
最后看是否在同等实际成本下保留成绩。仍需区分参数化容量、学习率与预测可学习性，
不要只凭 GRACE 接近较弱基线就宣布降维成功。

## 当前边界

三/四卡5090的rank8/8192 Full-PG显存测试与分卡审计命令见
[RTX5090_MEMORY_TEST.md](RTX5090_MEMORY_TEST.md)。当前诊断用1张actor+2张生成卡，旧四卡配置仍可选；
新增合成满长度梯度探针与真实4步训练显存报告；仍需服务器GPU实测。

- PR18 已运行 Qwen3-4B 三卡 GPU 训练、评测与冻结审计；详见[结果报告](results/pr18-20261014/followup-fast-3gpu/README.md)。
  这是一轮单 seed 快速诊断，不能当作多 seed 效果或加速结论。独立的满长度显存探针尚未补测。
- CPU tiny 的 direct 模式只验证算法接线，不模拟真实小核心的语言建模能力。
- 小型随机初始化 Qwen3 的 PEFT 导出一致性测试不需要下载模型，不是 4B 效果实验。
- 固定基底构造一次用 FP32 完整 SVD；可能较慢，计入 setup 成本。恢复时仍有构造开销，
  但随后加载保存的基底。`random` 是消融而不是偷偷更换默认初始化。
- 首版不对接历史 offline predictor 文件；旧 PCA 预测器和新 direct 坐标不可混用。
- 高阶旧分析脚本若直接把 Gram 当 dense list，应先适配 `{identity_dim: D}`；
  本次支持的主路径是上面的 train/evaluate/audit 及 streaming replay。
- 结果判断无自动 Go/No-Go 门槛；零信号、小样本、宽区间也应照实记录。
