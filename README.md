# GRACE-GC

实现《GRACE：ICLR 论文框架 v3》的主方法、核心对照和实验脚本。先完成本机代码与 CPU 测试，再到服务器做最小证伪。

显示名用 **GRACE-GC**，避免和另外两篇同名工作混淆。

## 范围

已实现：

- GRACE 全空间 HT、续写分配、双流更新
- Full-PG、Uniform-HT、Uniform-CV、Reward-CV、Prompt-CV
- GRPO、GRPO-short（独立实用目标）
- CPU tiny LoRA 参考训练
- reservoir / basis / 预测器 / IPW
- 数学数据分割、规则奖励、独立评测、前缀审计、方差×成本
- 运行目录、账本、A100/5090 配置

未在本机运行：真实 4B 训练和 GPU 集成测试。缺 vLLM/verl/CUDA 时 GPU 入口会报错，不提供占位后端。

## 安装

```bash
pip install -e ".[cpu,dev]"
```

服务器 GPU 请按实际采用的 **同一** verl 源码版本安装 vLLM/FSDP，不要混用不同日期的接口。候选参考：verl v0.9.0。

## CPU 检查

```bash
python -m pytest tests
python scripts/train.py --config configs/experiments/minimal.yaml --num-steps 2 --run-dir runs/cpu-smoke
python scripts/evaluate.py --generate --data-path data.jsonl --checkpoint runs/cpu-smoke/checkpoint.npz --run-dir runs/eval
python scripts/audit.py --generate --data-path data.jsonl --checkpoint runs/cpu-smoke/checkpoint.npz --run-dir runs/audit
```

## 服务器最小证伪（代码完成后）

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --config configs/experiments/pilot.yaml \
  --config configs/hardware/a100_4.yaml \
  --method grace \
  --backend gpu_verl \
  --model-path /path/to/Qwen3-4B-Base \
  --data-path /path/to/dapo.jsonl \
  --run-dir runs/grace-seed17
```

训练步数必须大于 `predictor.warmup_steps`（default/pilot 为 20），否则 named HT 方法不会离开 warmup。对照把 `--method` 换成 `full_pg`、`uniform_ht`、`uniform_cv`、`reward_cv`、`prompt_cv`、`grpo`、`grpo_short`。

评测与审计用训练快照自行生成（不要拿别的模型的 answers/bundles 当该方法的结果）。评测文件是 MATH-500，不要把 DAPO 训练语料再切 10% 当 eval。审计与训练用同一 DAPO 文件和同一 seed，这样会拿走论文预留的 240 题并从训练流排除。

```bash
python scripts/evaluate.py --generate --data-path /path/to/math500.jsonl \
  --config configs/default.yaml \
  --config configs/experiments/pilot.yaml \
  --config configs/hardware/a100_4.yaml \
  --backend gpu_verl \
  --model-path /path/to/Qwen3-4B-Base \
  --checkpoint runs/grace-seed17/checkpoint.npz \
  --run-dir runs/eval
python scripts/audit.py --generate --data-path /path/to/dapo.jsonl \
  --config configs/default.yaml \
  --config configs/experiments/pilot.yaml \
  --config configs/hardware/a100_4.yaml \
  --backend gpu_verl \
  --model-path /path/to/Qwen3-4B-Base \
  --checkpoint runs/grace-seed17/checkpoint.npz \
  --run-dir runs/audit
```

已有 `eval_per_problem.jsonl` / `audit_bundles.jsonl` 时可以改用 `--answers` / `--bundles` 只重算统计。

数据用官方 DAPO-Math-17k 的 JSONL 或 parquet：`prompt` 可以是对话列表，答案读 `reward_model.ground_truth`。源语料上的 `extra_info.split=train` 不会把全部题标成本项目的 train。未标记且达到语料规模的 DAPO 按论文切出校准 256、审计 240，其余训练；评测是单独的 MATH-500 文件（整份使用）。缺金标会报错，不会按空答案把奖励全记成 0。GPU 编码在 tokenizer 带 chat_template 时走 `apply_chat_template`（`enable_thinking=False`），与 verl/DAPO 对话格式对齐。审计集目前按 problem_id 可复现抽样，尚未按 base 模型 16 样本 pass rate 分层。

`hardware.n_gpu>1` 在真实 allreduce 接上前会拒绝启动；4 卡试跑先加 `hardware.n_gpu: 1`。5090 机器改用 `configs/hardware/rtx5090_8.yaml`。两种硬件分别记时，不要把 5090 小时折成 A100-hours。

## 约定

- G 是梯度上升方向；优化器 `.grad` 为 `-Ghat`
- 目标参数是全部 q/v LoRA A/B；token log-prob 求和
- 批内冻结 actor / baseline / basis / 预测器；选择只读前缀；停止者 `reward=null`
- 分母是预先确定的起步数 N
- 论文筛选条件只用于分析，结果不足也照常保存
- 已跑/未跑测试见 `docs/TEST_STATUS.md`

## 配置

| 文件 | 用途 |
|------|------|
| `configs/default.yaml` | 论文默认超参 |
| `configs/experiments/minimal.yaml` | 本机小规模 |
| `configs/experiments/pilot.yaml` | Pilot 规模 |
| `configs/hardware/a100_4.yaml` | 4×A100 |
| `configs/hardware/rtx5090_8.yaml` | 8×5090 |
