# PR18 三卡快速诊断实测结果（2026-10-14）

9/9 项任务正常完成，退出码均为 0。北京时间 2026-10-14 15:24:57 开始，22:46:59 结束，合计 7 小时 22 分钟。

执行代码与 PR18 提交 `332b816c8385d4cbd0105aaa0ea2e7c9cec078af` 的源文件逐项一致；见 `source_identity.json`。训练时本地 Git 元数据较旧，不能直接用其 HEAD 作为实测版本。此次上传保留远端历史结果，不删除本地缺失的旧归档。

## 配置与范围

- Qwen3-4B，关闭 thinking，输出上限 8192 token；rank-8 small-core，梯度维度 4608。
- 物理卡 7 用作 HF actor，2、3 是独立 TP=1 vLLM rollout worker；共三张 RTX5090。vLLM memory utilization 0.75，max sequences 2，chunked prefill，max batched tokens 2048。
- core8 Full-PG / Uniform-HT / GRACE 各 50 步，训练 seed=17，固定每批 8 题、32 条轨迹，固定 baseline=0.5，predictor warmup=10 步。
- 同一个固定 128 题 MATH-500 子集（题目选择 seed=20261013，评测 seed=17），每题 4 条答案；共享一次初始零适配器评测，各方法仅做最终评测，共 2048 条答案。
- 仅冻结 GRACE 最终检查点做审计：4 题 × 2 前缀 × 8 条续写，决策点 512；另有一次 CPU 预测报告。
- 启动命令见 `operations/start-fast-single-seed.sh`；有效配置、依赖版本、数据哈希和检查点哈希随结果保存。实际训练/评测/审计耗时见 `suite.json`。

## 结果

| 检查点 | 正确答案条数 | 平均正确率 | pass@4 |
|---|---:|---:|---:|
| Initial zero adapter | 383/512 | 74.8047% | 89.0625% |
| Full-PG | 384/512 | 75.0000% | 90.6250% |
| Uniform-HT | 376/512 | 73.4375% | 87.5000% |
| GRACE | 371/512 | 72.4609% | 87.5000% |

平均正确率以 512 条答案为分母；pass@4 是 128 题中至少答对一次的比例。四个评测的题目与顺序哈希一致。`eval_summary.json` 保留原始汇总及按题统计，`eval_per_problem_compact.jsonl` 保留分数、提取答案、seed、长度及截断信息，省去整段推理和 token ID。

这轮结果没有证明 GRACE 的质量收益。其正确率低于初始零适配器及 Full-PG；仅一个训练 seed，不能据此作多 seed 或显著性结论。

冻结审计的 8 个前缀上，有效预测向量范数全部为 0。预测 MSE 与零预测 MSE 同为 762.3473580259277，比值为 1.0；本轮没有显示预测收益。末步健康记录显示 `m_shrink=0.0`。这里记录观测，尚未完成原因定位，不能把“流程完成”解读为“方法有效”。审计只有 4 道独立题，64 条续写不能当成 64 道独立题。

训练命令耗时：Full-PG 112.23 分钟、Uniform-HT 82.80 分钟、GRACE 85.85 分钟。三组顺序执行、只有一个 seed，这些是实际运行记录，不能作为重复实验证实的加速结论。审计中的 token 成本代理不等于 GPU 墙钟速度。

## 文件与校验

- `result-summary.json`、`comparison_summary.json`、`suite.json`：总览、对比与 9 个作业的完整时间。
- 三个训练目录：50 步指标、健康记录、计算账本、配置、精简版本信息和检查点索引。
- 四个评测目录：原始评测汇总与题目清单、精简逐题答案分数。
- `core8_grace-seed17-audit/`：原始审计汇总、题目清单及冻结预测报告。原始 `audit_summary.json` 沿用执行程序的 NaN 表示；派生汇总将非有限数写为 null。
- `validation-pytest.log`：本次上传前运行已有相关 CPU 测试，68 passed；未重新执行 GPU 训练。
- `export-manifest.json`、`SHA256SUMS`：文件清单和校验和；`export_results.py` 是此次筛选导出脚本。

权重、NPZ/NPY、LoRA 检查点、完整训练轨迹、完整审计 bundles、环境/源代码的重复大段转储均未上传。单个结果文件小于 1 MiB。原始文件仍在服务器 `/mnt/mydata/wja/grace/runs/pr18-20261014/followup-fast-3gpu`；早期已中断的 `followup-full` 不属于这次完成结果。
