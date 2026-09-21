# 离线预测器与多卡 rollout 对照

2026-09-19 实施记录。对应用户提供的 `GRACE_offline_predictor_gpu_parallel_plan.md`，本轮授权范围为“离线对照和多卡并行一起实现”。这是待服务器验证的实验变体；9月18日负结果与论文证据状态不变。

## 已实现

| 环节 | 本轮实现与复用 |
| --- | --- |
| 离线监督 | `scripts/train_offline_predictor.py` 从现有 `calib` 划分取题，先排除外部评测题；保留原 ChatML 消息。复用训练入口的完整 G、基底、坐标、风险、成本与校准实现。 |
| 固定 actor | 从同 seed 共享 SFT 的 step-0 初始化；actor 的学习率与 weight decay 为零，结束后核对 actor/actor_full 参数未变。拟合仍支付生成、反传、优化器与保存成本。 |
| 采集与校准 | 整个离线拟合处于完整续写 warmup，审计概率为 1。部署的 beta 保留用于 holdout γ 校准；不把采集的 p=1 当成部署设计。固定 actor 下停用策略年龄淘汰/衰减。最后冻结 U。 |
| 推理 artifact | `predictor-<内容 hash>.npz` 保存 U、坐标/风险/成本头、scaler、γ、basis/sync ID、特征/LoRA/模型协议及来源 hash；不带 actor、actor 优化器、reservoir 或预测器优化器状态。 |
| 完全离线消费 | `train.py --offline-predictor PATH` 同时支持 CPU 与 GPU。在线不收集监督、不生成 fresh、不更新 U/heads/scaler/γ；仍按完整空间 HT/CV 更新 actor，停止者 reward 为 null。普通在线 GRACE 入口保留。 |
| 恢复与归档 | checkpoint 引用同目录 artifact；保存时复制依赖并校验字节 hash，避免重复压缩/解压 U。迁移完整目录后可续训，不依赖旧 donor 路径。续训与首次加载共用协议校验，拒绝不兼容的决策位置、生成上限或特征语义；复用加载时取得的协议，不额外解压 U。显式归档 checkpoint 会带上 artifact。离线 U 只读，日志复用加载时计算的 U hash。 |
| 多卡训练 | `rollout.workers=n_gpu-1`：第一张可见卡放 HF actor，其余卡各一个独立 TP=1 的 vLLM worker，使用 spawn 隔离进程。Full-PG 与 GRACE 复用相同并行入口。 |
| 请求及同步 | 父进程集中抽请求 seed、决定 p/Z/global N，worker 只生成。前缀与选中的续写分片并发，结果按原顺序还原。LoRA 保存一次后在所有 worker 同步加载、清空旧缓存，完成后才能继续。 |
| 故障及费用 | worker 异常/退出、同步返回失败与输出数量错误会报出，不能混用部分 snapshot。记录分片索引、worker RPC 时间、兼容回退尝试，时间已在父阶段内，不能再相加。退出时关闭本任务的 worker。 |
| 审计 | 离线 checkpoint 从第 0 步即可按其真实 basis/sync 状态审计，不误套在线 warmup。γ=0 或无真实基仍如实保留，不强制补全。 |
| 四组统一脚本 | `scripts/run_offline_comparison_gpu.py`：每 seed 共用一次格式 SFT、一次离线拟合；运行单卡/多卡 × Full-PG/离线 GRACE，固定 N=16、4 题、b=0.5、prescan=0、fresh=0。评测与审计统一单卡，并复用原始结果和多 seed 分析函数。 |

## 服务器用法

先沿 README 完成安装及数据/模型准备，在已激活的 GPU 环境运行。只填写真正可用的设备 ID。

```bash
export MODEL=/path/to/Qwen3-4B-Base
export TRAIN_DATA=/path/to/train.parquet
export EVAL_DATA=/path/to/math500.jsonl
python scripts/run_offline_comparison_gpu.py \
  --devices 0,1,2,3 --seeds 17 23 41 \
  --steps 40 --fit-steps 40 \
  --run-dir runs/offline-parallel-controls
```

默认先比较固定步数与固定主采样量。相同消费阶段墙钟预算使用 `--wall-seconds 2400`；它复用已有批次边界预算与 checkpoint 可用时间分析，实际超时仍会记录。追加 `--hardware-config configs/hardware/rtx5090_1.yaml` 可切换硬件配方，前提是每卡能装下对应角色的模型。设备数可以是 2、4、5 等：始终一张 actor 卡，其余为 rollout 卡；不是多卡 actor/FSDP 训练。

若只需在已有实验链加载离线 artifact，可设置 `OFFLINE_PREDICTOR=/path/to/predictor-<hash>.npz` 后调用原 `run_minimal_gpu.sh`，只作用于该链的 GRACE。单独多卡训练的配置为 `configs/hardware/a100_rollout_4.yaml`；评测/审计仍使用单卡配置。不要仅把旧硬件配置中的 n_gpu 改大而不设置 worker 布局。

输入数据需要非空 `calib` 划分。DAPO 规模数据由已有划分器产生，小数据可使用顶层 `split: calib` 显式标注；没有离线训练题时会报输入错误，不会偷偷从评测集取题。样本少、γ=0、秩不足或效果差都不是启动门槛。

## 成本与输出解释

- `matrix_summary.json` 保留四组逐 seed 结果、主 starts/步数、avg@4 均值与 seed 标准差、同硬件 GRACE−Full-PG 的配对 seed 区间；初始化或评测协议证据缺失时保留观测，不生成有效配对结论。
- `quality_endpoint=at_budget` 时，均值、标准差及配对区间来自 `selected_evaluation`：按命令起点时钟选择预算内最后可用且已评测的检查点，并核对来源、hash 和评测协议。任意步数的评测从 `stages.json` 读取；缺失有效预算证据时输出 null，不回退到结束分数。原始 `comparison.final_*` 和完整实际费用继续保留。固定步数模式为 `quality_endpoint=final`，使用实际结束分数。
- 训练 subprocess 墙钟含启动、加载、检查点和退出；GPU 秒按该运行配置使用的卡数乘墙钟，包含角色空闲时间，并非硬件利用率积分或云账单。
- cold 总成本额外计入共享 SFT、共享 actor 加载和一次完整离线拟合。每个假设独立部署的 GRACE 都计一次拟合；实际四组实验只拟合一次，不能把各组 cold 假设总计再次当成真实实验总费。
- 拟合内部 `offline_summary.json` 还记录数据准备、训练、导出范围；主报告采用外层 subprocess 实测时间，包括进程启动/退出。不要把两层时间相加。
- `--wall-seconds` 对齐的是消费训练阶段。cold 总成本是另列的完整账本，**不是相同 cold 总预算下的质量终点**。离线成本的摊销次数未预设；严格 cold 同预算还需按已测成本安排相应剩余训练预算。
- `commands.jsonl` 和每条 chain 的 `command_timing.jsonl` 保留测试、评测、审计及失败命令费用；它们与内部 ledger、worker 分项存在包含关系。方差×token 与 LAG 仍来自原审计，不能用停止比例或 worker 数代替加速。

## 明确保留的限制

本机无 GPU。CPU、进程通信替身和 tiny 回归不能证明 A100/5090 显存足够、实际吞吐提高或论文成功；最新检查记录见 [TEST_STATUS.md](TEST_STATUS.md)。

当前仍通过 HF 重放前缀提取特征，只复用此前的设备端特征归约；没有接入 vLLM hidden-state tap，也没有实现独立轻量文本编码器。多 worker 按当前请求索引轮转分片，选中的续写不保证回到原前缀 worker，因此不能预设跨阶段缓存命中或理想倍数加速。actor 特征和反传仍集中在一张卡。

完全冻结会失去策略漂移适应能力。γ、风险泛化、真实残差、U 能量覆盖、方差×成本、LAG 联合条件和多 seed 同算力质量均需新实验验证；没有人为调整这些数值来追求达标。可将已有在线固定 U 方案作为后续独立对照，不能把不同算法的结果混记。

API 核对：vLLM 的 [LLM.generate 文档](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/llm/#vllm.entrypoints.llm.LLM.generate)支持逐请求 SamplingParams，并说明输出保持输入顺序。本实现仍保存实际运行版本、接受的 engine 参数和兼容回退；多进程下 RequestOutput/LoRARequest 的真实版本兼容性待 GPU 服务器验证。
