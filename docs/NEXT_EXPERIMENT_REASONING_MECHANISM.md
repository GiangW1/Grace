# 下一轮机制实验

这份配置对应 PR12 评审后的下一轮实验。它保留强主张作为主分析：在答案文本出现、并且固定预算强制作答仍不能可靠恢复答案之前，更新本身是否已经确定。风险调度、难题稀有成功和后缀成本是并行分析，不替代主分析。

## 设计

- 使用真正支持 thinking 的后训练模型；`enable_thinking: true` 会同时作用于训练、独立评测和 GPU audit 的 chat template。
- 用 8k 生成上限，在 1024 和 2048 两个位置采样；先用只采样审计检查截断率和剩余长度，再运行昂贵的全梯度 replay。
- 每个前缀保存独立的固定预算 probe：强制追加 `</think>\nAnswer:`，采样 4 次，每次 32 token。它只用于资格审计，不进入原始续写或梯度标签；`functional_recoverable` 采用预先固定的 `mean_reward >= threshold` 规则（默认阈值 0.5）。
- replay 的欧氏统计始终保留；Adam 度量必须在 replay 前由 checkpoint 导出，并用偏差修正后的二阶矩权重。流式模式只保存充分统计量，不落盘 `N×D` 全矩阵。
- `difficulty_manifest` 在分析阶段显式传入，结果按 easy/medium/hard、独立 q 分层、严格答案未定子集和题目聚类 bootstrap 报告。
- 单卡共置时默认把 vLLM 上限压到 30% 显存并把并发序列限制为 2；reasoning 配置进一步使用 25% 和单序列。HF actor 默认启用 gradient checkpointing，避免 8k 反向和 vLLM KV cache 同时挤满 A100。

## 运行顺序

先运行 `scripts/audit.py`，使用本文件和数据、模型、checkpoint 的实际路径，并传入预注册难度清单：

```powershell
python scripts/audit.py --generate --data-path DATA.jsonl --model-path MODEL `
  --checkpoint CHECKPOINT --config configs/default.yaml `
  --config configs/experiments/reasoning_mechanism_audit.yaml `
  --difficulty-manifest DIFFICULTY.json --run-dir RUN
```

检查 `audit_summary.json` 中 thinking 是否生效、截断率、probe 资格统计和两个 decision point 的覆盖。然后在 checkpoint 旁生成固定 Adam sidecar：

```powershell
python scripts/build_adam_metric.py --checkpoint CHECKPOINT --output RUN/adam_metric.npy
```

再用 `scripts/replay_expected_gain.py` 对每个 decision point 重放。对长推理实验建议使用 `--statistics-only --store-trajectory-decomposition --metric-file RUN/adam_metric.npy --metric-name adam_diagonal`；该模式逐条续写只写标量、半样本 Gram 量、后缀能量和 batch 累加量。

最后运行：

```powershell
python scripts/analyze_learning_mechanism.py `
  --replay-dir REPLAY `
  --run-dir ANALYSIS `
  --metric-name euclidean `
  --difficulty-manifest DIFFICULTY.json `
  --split-manifest SPLIT.json `
  --bootstrap 1000
```

使用 `--metric-name adam_diagonal` 重复同一分析。两个度量、所有 q 层、题目 bootstrap 和零梯度前缀都保留在输出中；没有自动 Go/No-Go 门槛。

## 解释边界

截断在奖励和梯度标签中仍然是 `R=0`；只有长度和答案位置的描述把它作为删失。奖励解析只读取闭合 `</think>` 后的答案通道，未闭合 thinking 中的 `\boxed{}` 或 `Answer:` 不会得分。功能 probe 的 decoded continuation 会恢复预填的 `Answer:` 再评分，并和普通续写 q 分开保存。`strict_pre_answer_undecided` 同时要求普通续写 q 位于 0 和 1 之间、文本尚未发出答案、功能 probe 未可靠恢复、且前缀未自然结束。旧 replay 没有显式普通 q 字段时回退到半 A 的普通续写均值；没有功能 probe 时仍可分析，但严格层不会把缺失资格误当成通过。
