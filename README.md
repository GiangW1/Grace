# GRACE-GC

Qwen3-4B-Base 上的 GRACE 训练、评测和前缀审计。本机先做 CPU 检查；服务器按下面全流程做。

显示名用 **GRACE-GC**，避免和另外两篇同名工作混淆。多卡 allreduce 还没接上，**先单卡**。

下面是一条能按顺序做完的全流程。第 1–7 节是同一条线的细节，卡住时再翻。

## 全流程

顺序：**环境 → 资源 → 绑卡/tmux → smoke Full-PG → 最小比较 → Pilot Full-PG → Pilot GRACE → 其余对照**。先看 Full-PG 会不会学，再跑 GRACE。不要一上来就用 `grace` 做第一份 GPU 作业。

GitHub 默认分支是 `master`。clone 后确认 `git log -1` 含保真接线，不要停在只训推理开头的 `37fbcb2`。

5090 把下面所有 `configs/hardware/a100_1.yaml` 换成 `configs/hardware/rtx5090_1.yaml`。两种卡分别记时，不要折成 A100-hours。

### A. 本机（没有 GPU 也做）

```bash
pip install -e ".[cpu,dev]"
python -m pytest tests -k "not complete_final_expression and not u6_gpu"
```

Windows 上 `complete_final_expression` 会踩 math-verify 的 WinError 6；`u6_gpu` 要 CUDA。

### B. 服务器环境

```bash
conda create -n grace python=3.11 -y
conda activate grace
```

先装 vLLM（或 `pip install "verl[vllm]==0.9.0"`），再进仓库：

```bash
git clone https://github.com/yiweinanzi/Grace.git
cd Grace
pip install -e ".[gpu,dev]"
pip install pyarrow
python -c "import torch, vllm, transformers, peft, math_verify; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); print('vllm', vllm.__version__); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

`torch.cuda.is_available()` 必须是 `True`。`backend=gpu_verl` 只是配置别名，实现是 **HF actor + vLLM 两阶段**，不是 verl PPO。启动时会打印 `implementation=gpu_vllm_hf`，并写 `startup.json`。

### C. 模型和数据

```bash
python scripts/fetch_assets.py --root data
# 把脚本打印的三行 export 贴进当前 shell
test -d "$MODEL" && test -f "$TRAIN_DATA" && test -f "$EVAL_DATA" && echo ok
```

官方 DAPO parquet 可直接加载。同一题面金标冲突的整组丢掉，写入 `data_conflicts.json`（大约 12 组），不会硬失败。MATH-500 套与 DAPO 相同的 `Answer:` 指令。`format_warmup` 是共享格式 SFT：只训推理开头，不训 `Answer:`、金标和 EOS。它不是 `predictor.warmup_steps`。最后一行格式靠 DAPO 指令。

### D. 绑卡和会话

```bash
export CUDA_VISIBLE_DEVICES=0
tmux new -s grace
```

`hardware.n_gpu` 保持 1。训练、评测、审计都在这个 tmux 里做；断 SSH 用 `Ctrl-b d`，回来 `tmux attach -t grace`。同一张卡不要并行开两份作业。

### E. Smoke：先 Full-PG（4 步）

短跑是 1024/2048 token + 32 步格式 SFT，评测/审计各 8 题。只确认链路，不是 Pilot。

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py \
  --config configs/default.yaml \
  --config configs/experiments/smoke.yaml \
  --config configs/hardware/a100_1.yaml \
  --method full_pg \
  --backend gpu_verl \
  --model-path "$MODEL" \
  --data-path "$TRAIN_DATA" \
  --num-steps 4 \
  --run-dir runs/smoke-fullpg

python scripts/evaluate.py --generate \
  --config configs/default.yaml \
  --config configs/experiments/smoke.yaml \
  --config configs/hardware/a100_1.yaml \
  --backend gpu_verl \
  --model-path "$MODEL" \
  --data-path "$EVAL_DATA" \
  --checkpoint runs/smoke-fullpg/checkpoint.npz \
  --run-dir runs/smoke-eval-fullpg

python scripts/audit.py --generate \
  --config configs/default.yaml \
  --config configs/experiments/smoke.yaml \
  --config configs/hardware/a100_1.yaml \
  --backend gpu_verl \
  --model-path "$MODEL" \
  --data-path "$TRAIN_DATA" \
  --checkpoint runs/smoke-fullpg/checkpoint.npz \
  --run-dir runs/smoke-audit-fullpg
```

看 `run.log` 里有没有 `phase=prescan` / `phase=prefix` / `phase=continue_done` / `step N`。第一份适配器叫 `lora/step-2`，那是 vLLM adapter id（首次 sync），**不是**已经训完 2 步。`health.json` 里 `lora_A_grad_zero_expected` 在 B≈0 时可以为真（PEFT ΔW=B·A）。`all_reward_zero` 且 `n_baseline_zero>0` 时，`baseline_collapsed_with_zero_reward` 为真，优势会是 0。审计里 `Var(R)=0` 时 ρ 是 NaN，这是公式，不是崩了。

### E2. 最小比较（现役工作树）

16 题、40 步、单卡。用来看 GRACE 分配是否压过 Full-PG / Uniform-CV / GRPO，不是 Pilot。不要把 2026-09-16 那次链当这份代码的结果。

```bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_minimal_gpu.sh
```

默认方法是 `full_pg grace uniform_cv grpo`。看质量和实际成本，不要看停止者比例。

### F. Pilot：先 Full-PG 30 步

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py \
  --config configs/default.yaml \
  --config configs/experiments/pilot.yaml \
  --config configs/hardware/a100_1.yaml \
  --method full_pg \
  --seed 17 \
  --backend gpu_verl \
  --model-path "$MODEL" \
  --data-path "$TRAIN_DATA" \
  --num-steps 30 \
  --run-dir runs/fullpg-seed17

python scripts/evaluate.py --generate \
  --config configs/default.yaml \
  --config configs/experiments/pilot.yaml \
  --config configs/hardware/a100_1.yaml \
  --backend gpu_verl \
  --model-path "$MODEL" \
  --data-path "$EVAL_DATA" \
  --checkpoint runs/fullpg-seed17/checkpoint.npz \
  --run-dir runs/eval-fullpg-seed17
```

完整 Pilot 审计（240 题 × 16 前缀 × 8 个 t × 16 续写）很大，确认训练和评测没问题后再开：

```bash
python scripts/audit.py --generate \
  --config configs/default.yaml \
  --config configs/experiments/pilot.yaml \
  --config configs/hardware/a100_1.yaml \
  --backend gpu_verl \
  --model-path "$MODEL" \
  --data-path "$TRAIN_DATA" \
  --checkpoint runs/fullpg-seed17/checkpoint.npz \
  --run-dir runs/audit-fullpg-seed17
```

### G. Pilot：再 GRACE 30 步

GRACE 的 warmup 是 20 步，`--num-steps` 必须大于 20。评测/审计必须用**该方法自己的** `checkpoint.npz`。

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py \
  --config configs/default.yaml \
  --config configs/experiments/pilot.yaml \
  --config configs/hardware/a100_1.yaml \
  --method grace \
  --seed 17 \
  --backend gpu_verl \
  --model-path "$MODEL" \
  --data-path "$TRAIN_DATA" \
  --num-steps 30 \
  --run-dir runs/grace-seed17

python scripts/evaluate.py --generate \
  --config configs/default.yaml \
  --config configs/experiments/pilot.yaml \
  --config configs/hardware/a100_1.yaml \
  --backend gpu_verl \
  --model-path "$MODEL" \
  --data-path "$EVAL_DATA" \
  --checkpoint runs/grace-seed17/checkpoint.npz \
  --run-dir runs/eval-grace-seed17
```

### H. 其余对照

每个方法单独目录，换 `--method` 和 `--run-dir`。机制对照：`uniform_ht`、`uniform_cv`、`reward_cv`、`prompt_cv`。实用对照：`grpo`、`grpo_short`。续训的 `--num-steps` 是再跑多少步，不是累计到多少。

已有生成结果时只重算统计：

```bash
python scripts/evaluate.py --answers runs/eval-fullpg-seed17/eval_per_problem.jsonl --run-dir runs/eval-recompute
python scripts/audit.py --bundles runs/audit-fullpg-seed17/audit_bundles.jsonl --run-dir runs/audit-recompute
```

### 读目录时别搞错

| 现象 | 含义 |
|---|---|
| `implementation=gpu_vllm_hf` | 实际是 HF+vLLM，不是 verl PPO |
| `lora/step-2` 在第 1 步前就出现 | adapter id，首次 sync；不是 `state.step` |
| `phase=prescan` 很久没新日志 | 未见过的题各抽 4 条满长样本写 baseline；现在会打心跳 |
| `data_conflicts.json` | 官方 DAPO 冲突题面已丢掉 |
| `format_warmup.json` | 共享格式 SFT：只训推理开头，不训 `Answer:`/金标/EOS。`mean_response_tokens` 经常 <16 是交卷 |
| `n_continued` / `n_prefix_finished` | 续写条数；整批 0 说明决策点前已 EOS |
| 审计 `decision_grid_unobserved` | 该 t 没有存活前缀；不是 yaml 删了 64/128 |
| `baseline_collapsed_with_zero_reward` | 全零奖励且 b=0，优势为 0 |
| `lora_A_grad_zero_expected` | B≈0 时 A 梯度应为 0 |
| 审计 ρ 为 NaN | 该子集 `Var(R)=0`，按公式就是 NaN |

同一个 `--run-dir` 再跑一次（非续训）会改写成 `原名-时间`，不会盖掉上一次。不写 `--run-dir` 时目录是 `runs/train-YYYYMMDD-HHMMSS`。

## 1. 环境

单独建一个环境，不要装进系统 Python。先装 vLLM（它会带上匹配的 PyTorch），再装本仓库。不要先随手装一个 torch，再装另一个日期的 vLLM。

**verl 0.9.0 和 vLLM 0.9.0 不是同一个东西。** 本仓库 GPU 路径实际用的是 Hugging Face actor + vLLM 两阶段生成；`verl` 包不是硬依赖。如果要对齐论文栈，选定 **一个** verl 版本，vLLM 跟该版本走。候选：verl `v0.9.0`（其 `vllm` extra 要求 vLLM ≥ 0.18，不是 vLLM 0.9）。

### 1.1 机器上先看什么

```bash
nvidia-smi
python3 --version
```

- Python 3.10–3.12。
- 驱动要能跑你准备装的 CUDA 版 PyTorch / vLLM。A100 常见是 CUDA 12.x；5090 用更新的 vLLM，并确认该 wheel 支持 Blackwell。
- 驱动不够新时，先升级驱动，不要降 vLLM 去凑旧驱动。

### 1.2 新建环境

conda：

```bash
conda create -n grace python=3.11 -y
conda activate grace
```

venv：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 1.3 先装 vLLM，再装本仓库

**做法 A（推荐，跟 verl 对齐）**

到 [verl 安装说明](https://verl.readthedocs.io/en/latest/start/install.html) 按你选定的 tag 装。选定 `v0.9.0` 时：

```bash
pip install "verl[vllm]==0.9.0"
```

这会装上该版本指定的 vLLM 和 PyTorch。然后再进本仓库：

```bash
git clone https://github.com/yiweinanzi/Grace.git
cd Grace
pip install -e ".[gpu,dev]"
pip install pyarrow
```

**做法 B（不装 verl）**

按 [vLLM GPU 安装](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) 为当前 CUDA 装官方 wheel，再执行上面的 `pip install -e ".[gpu,dev]"` 和 `pyarrow`。

装完检查 torch 有没有被后装的包改掉：

```bash
python -c "import torch, vllm, transformers, peft, math_verify; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); print('vllm', vllm.__version__); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

`torch.cuda.is_available()` 必须是 `True`。若 torch 版本被后装包覆盖，回到该 vLLM 文档把推理栈重装一次，不要混两个日期的 wheel。

读 DAPO parquet 需要 `pyarrow` 或 `pandas`。缺 `math-verify` 时 GPU 训练/评测/审计会直接报错。下载资源还需要 `huggingface_hub`。

本机没有 GPU 时不要走上面的 GPU 栈：

```bash
pip install -e ".[cpu,dev]"
python -m pytest tests -k "not complete_final_expression and not u6_gpu"
```

## 2. 准备模型和数据

官方来源：

| 用途 | Hugging Face | 大约体积 |
|------|----------------|----------|
| 模型 | `Qwen/Qwen3-4B-Base` | 8–10 GB（bf16） |
| 训练 | `BytedTsinghua-SIA/DAPO-Math-17k` | 几十到几百 MB |
| 评测 | `HuggingFaceH4/MATH-500` | 很小 |

另外预留 Hugging Face 缓存和 `runs/`。短跑几个 GB 就够；完整 Pilot 审计的 `audit_bundles.jsonl` 可能到十几 GB 以上，训练盘建议空出 **50 GB+**。

下载默认走 [HF Mirror](https://hf-mirror.com)。`fetch_assets.py` 会自己设置 `HF_ENDPOINT`；能直连官网时加 `--official`。

```bash
pip install huggingface_hub pyarrow
python scripts/fetch_assets.py --root data
```

脚本结束会打印三行 `export`，复制进当前 shell。手动下也先开镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-4B-Base --local-dir data/Qwen3-4B-Base
huggingface-cli download BytedTsinghua-SIA/DAPO-Math-17k --repo-type dataset --local-dir data/dapo
huggingface-cli download HuggingFaceH4/MATH-500 --repo-type dataset --local-dir data/math500
export MODEL="$PWD/data/Qwen3-4B-Base"
export TRAIN_DATA=$(find "$PWD/data/dapo" -name '*.parquet' | head -n 1)
export EVAL_DATA=$(find "$PWD/data/math500" \( -name '*.parquet' -o -name '*.jsonl' \) | head -n 1)
```

- 训练文件是官方 DAPO 的 JSONL 或 parquet。`prompt` 可以是对话列表，答案读 `reward_model.ground_truth`。官方 parquet 可直接加载；同一题面金标冲突的整组会丢掉，并写入 `data_conflicts.json`（大约 12 组）。
- 评测必须是单独的 MATH-500，字段用 `problem`/`prompt` 和 `answer`。不要把 DAPO 再切 10% 当考卷。MATH-500 会套上与 DAPO 相同的 `Answer:` 格式指令。
- 未标记且够大的 DAPO 会按 seed 17 切出校准 256、审计 240，其余训练。缺金标会报错。
- `format_warmup` 是论文要求的共享格式 SFT：只训推理开头，不训 `Answer:`、金标和 EOS。默认 256 步；续训会跳过。看 `mean_response_tokens`：经常小于 16 是交卷，不是学会了。

检查路径存在再往下：

```bash
test -d "$MODEL" && test -f "$TRAIN_DATA" && test -f "$EVAL_DATA" && echo ok
```

5090 机器把下面命令里的 `configs/hardware/a100_1.yaml` 换成 `configs/hardware/rtx5090_1.yaml`。两种卡分别记时，不要折成 A100-hours。

## 3. 单卡、会话、显存

目标机器即使是 4×A100，现在也只能用 **一张卡**。不绑定时，vLLM 和 HF actor 可能同时看见多张卡，显存会乱。

```bash
export CUDA_VISIBLE_DEVICES=0
```

换卡就把 `0` 改成 `1`/`2`/`3`。`hardware.n_gpu` 仍然是 1，不要改成 4。

训练和审计要挂着跑，断 SSH 会停：

```bash
tmux new -s grace
# 环境、export、训练都在这个窗口里做
# 断开：Ctrl-b d
# 回来：tmux attach -t grace
```

同卡上要同时放下 4B 的 HF actor 和 vLLM。默认 `gpu_memory_utilization=0.5`，A100 80GB 一般够短跑。OOM 时：

1. 确认只有这一份作业在用这张卡：`nvidia-smi`
2. 在硬件 yaml 里把 `vllm.gpu_memory_utilization` 降到 `0.35` 或 `0.4`
3. 不要在同一张卡上再开评测或第二个训练

完整 Pilot 审计先别和训练抢盘、抢卡；确认训练和评测完成后再开。

## 4. 先打通（短跑）

第一份短跑用 Full-PG（见上面全流程 E）。下面是同一套命令换成 `grace` 的写法。

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py \
  --config configs/default.yaml \
  --config configs/experiments/smoke.yaml \
  --config configs/hardware/a100_1.yaml \
  --method grace \
  --backend gpu_verl \
  --model-path "$MODEL" \
  --data-path "$TRAIN_DATA" \
  --num-steps 4 \
  --run-dir runs/smoke-grace
```

```bash
python scripts/evaluate.py --generate \
  --config configs/default.yaml \
  --config configs/experiments/smoke.yaml \
  --config configs/hardware/a100_1.yaml \
  --backend gpu_verl \
  --model-path "$MODEL" \
  --data-path "$EVAL_DATA" \
  --checkpoint runs/smoke-grace/checkpoint.npz \
  --run-dir runs/smoke-eval
```

```bash
python scripts/audit.py --generate \
  --config configs/default.yaml \
  --config configs/experiments/smoke.yaml \
  --config configs/hardware/a100_1.yaml \
  --backend gpu_verl \
  --model-path "$MODEL" \
  --data-path "$TRAIN_DATA" \
  --checkpoint runs/smoke-grace/checkpoint.npz \
  --run-dir runs/smoke-audit
```

训练、评测、审计都过了，再放大。短跑是 1024/2048 token 加 32 步格式 SFT，评测 8 题（`eval.n_problems`）；不设这个字段时评测仍是 MATH-500 整份。短跑数字只用来确认链路，不是 Pilot。不写 `--run-dir` 时目录是 `runs/train-YYYYMMDD-HHMMSS`（评测/审计同理）。同一个 `--run-dir` 再跑一次（非续训）会改写成 `原名-时间`，不会盖掉上一次。

## 5. 最小证伪（Pilot 规模、单卡）

GRACE 的 warmup 是 20 步，`--num-steps` 必须大于 20。对照把 `--method` 和 `--run-dir` 一起换。续训的 `--num-steps` 是再跑多少步，不是累计到多少。

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py \
  --config configs/default.yaml \
  --config configs/experiments/pilot.yaml \
  --config configs/hardware/a100_1.yaml \
  --method grace \
  --seed 17 \
  --backend gpu_verl \
  --model-path "$MODEL" \
  --data-path "$TRAIN_DATA" \
  --num-steps 30 \
  --run-dir runs/grace-seed17
```

```bash
python scripts/evaluate.py --generate \
  --config configs/default.yaml \
  --config configs/experiments/pilot.yaml \
  --config configs/hardware/a100_1.yaml \
  --backend gpu_verl \
  --model-path "$MODEL" \
  --data-path "$EVAL_DATA" \
  --checkpoint runs/grace-seed17/checkpoint.npz \
  --run-dir runs/eval-grace-seed17
```

完整 Pilot 审计（240 题 × 16 前缀 × 8 个 t × 16 续写）很大。确认训练和评测没问题后再开：

```bash
python scripts/audit.py --generate \
  --config configs/default.yaml \
  --config configs/experiments/pilot.yaml \
  --config configs/hardware/a100_1.yaml \
  --backend gpu_verl \
  --model-path "$MODEL" \
  --data-path "$TRAIN_DATA" \
  --checkpoint runs/grace-seed17/checkpoint.npz \
  --run-dir runs/audit-grace-seed17
```

评测和审计必须用**该方法自己的** `checkpoint.npz`。已有生成结果时，可以只重算统计：

```bash
python scripts/evaluate.py --answers runs/eval-grace-seed17/eval_per_problem.jsonl --run-dir runs/eval-recompute
python scripts/audit.py --bundles runs/audit-grace-seed17/audit_bundles.jsonl --run-dir runs/audit-recompute
```

续训：

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --config configs/experiments/pilot.yaml \
  --config configs/hardware/a100_1.yaml \
  --method grace \
  --backend gpu_verl \
  --model-path "$MODEL" \
  --data-path "$TRAIN_DATA" \
  --num-steps 30 \
  --resume runs/grace-seed17/checkpoint.npz \
  --run-dir runs/grace-seed17
```

## 6. 对照

每个方法单独目录。机制对照：`full_pg`、`uniform_ht`、`uniform_cv`、`reward_cv`、`prompt_cv`、`grace`。实用对照：`grpo`、`grpo_short`。

先看 Full-PG 或 GRPO 会不会学，再跑 GRACE。命令顺序见上面全流程 F–H。

## 7. 看哪些文件

| 目录里的文件 | 内容 |
|---|---|
| `config.yaml` / `environment.json` / `run_meta.json` / `startup.json` | 实际配置、包版本、进程、开始时间、实际目录；GPU 启动时写明 `gpu_vllm_hf` |
| `summary.json` | `complete` 或 `failed`，以及开始/结束 UTC。同名再跑写成 `原名-YYYYMMDD-HHMMSS` |
| `run.log` | 标准输出和异常；GPU 训练有 `phase=prescan` / `prefix` / `continue_done` |
| `steps.jsonl` / `health.json` | 每步汇总；全零奖励/优势、baseline b、LoRA A/B（缺梯度 vs 零梯度）、未训预测器分配、nvidia-smi、logprob 对照 |
| `data_splits.json` / `data_conflicts.json` / `format_warmup.json` / `tokenizer.json` / `vllm_engine.json` / `sampling.json` | 切分、丢掉的冲突题、格式 SFT、tokenizer stop、实际引擎参数 |
| `logprob_probe.jsonl` | 同序列 HF vs vLLM logprob（不可用则记原因） |
| `trajectories.jsonl` | 每条起步：p/Z/f/r̂/ĉ/优势/长度/结束原因/答案/token |
| `compute_ledger.json` / `compute_ledger.jsonl` | 按步、按阶段的墙钟；`summary` 总量只计 `train`/`eval`/`audit` 外壳，不把 `phase_*` 和 `train_step` 再加一遍 |
| `checkpoint.npz` / `checkpoints/step_k.npz` | 最新与逐步快照 |
| `eval_summary.json` / `eval_per_problem.jsonl` | MATH-500 的 avg@k、答案、截断、token |
| `audit_summary.json` / `audit_bundles.jsonl` | ρ、ELF、方差×成本、前缀/后缀文本 |

```bash
python scripts/plot.py --summary runs/eval-grace-seed17/eval_summary.json --out runs/plot-eval
```

这只是把 `per_problem` 写成 CSV，不是论文图。

## 不要做的事

- 不要用 `configs/hardware/a100_4.yaml` 或 `rtx5090_8.yaml` 启动，`n_gpu>1` 会直接失败。
- 不要在没设 `CUDA_VISIBLE_DEVICES` 的 4 卡机器上直接开训。
- 不要漏 `--num-steps`。默认是 1，GRACE 出不了 warmup。
- 不要拿别人的 answers/bundles 当这个方法的结果。
- 不要把 tiny / smoke 数字写成实测。
- 不要在同一张卡上并行开两份训练/评测。

## 配置

| 文件 | 用途 |
|------|------|
| `configs/default.yaml` | 论文默认超参 |
| `configs/experiments/smoke.yaml` | 服务器短跑 |
| `configs/experiments/minimal.yaml` | 本机 CPU 冒烟 |
| `configs/experiments/minimal_gpu.yaml` | 单卡 16 题比较；经 `scripts/run_minimal_gpu.sh` 与 default 合并 |
| `configs/experiments/pilot.yaml` | Pilot 规模 |
| `configs/hardware/a100_1.yaml` | 单卡 A100（先用这个） |
| `configs/hardware/rtx5090_1.yaml` | 单卡 5090 |
| `configs/hardware/a100_4.yaml` | 4 卡，现在不能跑 |
| `configs/hardware/rtx5090_8.yaml` | 8 卡，现在不能跑 |

已跑/未跑测试见 `docs/TEST_STATUS.md`。
