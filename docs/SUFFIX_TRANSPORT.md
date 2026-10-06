# GRACE-ST 实验与 smoke

本入口实现“多个不同前缀共享一个真实后缀”的实验，不改旧 HT/GRACE 入口。
代码基于 PR14 的 `exp-a-learning-completion`，本 PR 是其上的独立增量。
2026-10-06 已在双卡服务器完成 16384-token 上限的 m=2/4 冻结审计，
结果与原始统计见 [PR15 实验记录](results/suffix_transport_pr15_16384_20261006/README.md)。
以下命令保留为运行说明；多方法、多 seed 预算训练尚未执行。

## 实际算子

同题独立生成 m 个固定位置前缀；自然 EOS 者保留其完整梯度。
从仍存活的前缀里均匀选 donor，选择先于后缀采样，使用独立 selection RNG。
活组大小 k，后验 `alpha = softmax(sum log p(suffix|prefix))`，不做长度平均、
温度平滑、裁剪或近似后缀奖励。停止 token 包含在序列概率中。

`G_ST = G_D + lambda*C`，其中
`G_D = A_D*(g_prefix,D + g_suffix,D)`，
`C = sum_i alpha_i*A_i*g_prefix,i - A_D*g_prefix,D`。
默认 lambda=1，lambda=0 是相同多前缀设置下的机制消融；lambda 在当前结果产生前固定。
baseline 默认固定 0.5，全部 q/v LoRA A/B、token logprob 求和，`.grad=-G`。
各活组贡献乘 `k/N`，自然完成者各贡献 `1/N`，N 是本批原始起步数。

实现只在无梯度 forward 中计算接收者的长后缀似然；其 backward 只包含短前缀。
donor 的前缀/后缀 token loss 共享同一个完整 forward，保留前缀激活图。
softmax 分块重算控制 FP32 临时内存。actor 在一批内冻结，更新后同步 adapter 并清缓存。
同题接收者按自己的拼接文本判奖；开启 thinking 的 prompt 若已经预填 `<think>`，
判奖时补回该标记。未 EOS 的长度截断回答奖励为 0，停止前缀的真实奖励保存为 null。
拼接反事实的成功不计为独立探索成功。

无偏性指 temperature=1、无采样截断、相同生成/计分策略下的截断目标的梯度期望，
不是 Adam 更新或最终准确率保证。HF/vLLM 的有限精度差异会记录。
短长度探针改变了奖励 horizon，不能代替完整 8192 长度结果。

## 服务器先检查

沿用 README 的 GPU 环境与版本，不重装已正常工作的 vLLM。设置真实路径：

```bash
git fetch origin
git switch codex/suffix-transport-experiments

export MODEL=/absolute/path/to/Qwen3-4B-Base
export DATA=/absolute/path/to/dapo/train.parquet
export AUDIT_DATA=/absolute/path/to/new_heldout_audit.jsonl
export EVAL=/absolute/path/to/math500.jsonl
export INIT=/absolute/path/to/shared/checkpoint.npz

COMMON=(--config configs/hardware/a100_1.yaml
        --config configs/experiments/suffix_transport.yaml
        --gpus 0,1 --model-path "$MODEL" --data-path "$DATA"
        --eval-data-path "$EVAL" --init-checkpoint "$INIT")

python -m pytest tests/test_suffix_transport.py -q -o addopts=''
python scripts/suffix_transport.py --mode smoke --cpu --run-dir runs/st-math-smoke
python scripts/suffix_transport.py --mode train "${COMMON[@]}" --method full_pg \
  --steps 1 --n-problems 2 --run-dir runs/st-fullpg-one-step
python scripts/suffix_transport.py --mode smoke "${COMMON[@]}" --run-dir runs/st-gpu-smoke
```

`INIT` 是共同初始 actor 或已有中间训练 checkpoint，加载 actor 权重，使用新的 optimizer/RNG。
要审计 zero-B 初始化，从 COMMON 中删除 `--init-checkpoint "$INIT"`；两种初始化均应保留。
模型路径与 LoRA rank/alpha/targets 会对照 checkpoint 校验。
不支持 ST optimizer/RNG 续训；不能把 `--init-checkpoint` 当作 resume。

`--gpus 0,1` 使用第一张可见卡的 HF actor 和第二张卡的 vLLM worker。
`--gpus 0,1,2,3` 自动设 workers=3、TP=1。单卡用 `--gpus 0`，共卡显存按已有规则限额。
这是单 actor、多 rollout，不是多卡 actor 归约。

CPU smoke 精确枚举无偏期望，检查长序列 logsumexp、lambda=0、组大小=1、
自然完成后的固定 N 与 RNG 隔离。
GPU smoke 检查实际生成/EOS路径、全 q/v LoRA loss 与参考梯度、梯度符号、
prefix+suffix 分解、checkpoint roundtrip、前缀和实际后缀的 HF/采样 token logprob 对齐；
还会执行小 SGD 更新、导出新 adapter 并再次 rollout，最后恢复初始权重与 adapter。
合成 advantages 会明确标记，不作为实验结果。实现检查失败会非零退出。
BF16 梯度分解按每个 q/v LoRA 参数张量的 L2 误差检查，避免相消后接近零的坐标
使逐坐标相对误差失真；相对容忍值仍为 0.03，绝对容忍值为 1e-5。
`autograd.decomposition_checks` 保存逐张量误差，非有限值和超出容忍值的误差仍会报错。
logprob 平均绝对误差容忍值默认 0.5，可在 `suffix_transport.smoke_logprob_mean_abs_tolerance`
配置；这是发现错误 adapter/映射的粗检查，不是数值无偏性的认证。
查看 summary 的逐 token 误差和 `signed_sequence_logprob_error`；小的逐 token 偏差可能在长后缀上
累积并改变后验。缺失的恢复 EOS 概率单独标记，不包含在可比较的序列误差中。

短 smoke 默认前缀 8、上限 40。真实长序列 backward 需单独检查。
上面已按 AGENTS 先运行 Full-PG；接着检查 ST。这些只做一次更新，不用它们的模型比较效果：

```bash
python scripts/suffix_transport.py --mode train "${COMMON[@]}" --method suffix_transport \
  --steps 1 --n-problems 2 --run-dir runs/st-one-step
# 也可把真实长度带入 smoke，它会对最长实际生成轨迹做一次完整 backward。
python scripts/suffix_transport.py --mode smoke "${COMMON[@]}" \
  --decision-tokens 512 --max-new-tokens 8192 --run-dir runs/st-long-smoke
```

smoke 能找实现错误，不能保证任意长度/题目不 OOM。自然短答时，长 smoke 实际反传长度
以 `summary.actual_trajectory_backward.response_tokens` 为准。

## 新留出题上的冻结机制审计

`AUDIT_DATA` 应单独准备为未用于选 lambda 或路线的题目，并与训练/最终评测题核对去重；
重新生成前缀或更换 RNG seed 不会自动得到新题。默认 DATA 的固定 audit 划分可能与 PR13/14
讨论过的题重合，不能直接称为新留出集。此入口不验证 checkpoint 的历史数据接触情况。
专用文件保留 `split: audit` 标记，或使用不带 split 的独立题目文件。

```bash
python scripts/suffix_transport.py --mode audit "${COMMON[@]}" \
  --data-path "$AUDIT_DATA" \
  --group-size 2 --n-problems 32 --groups-per-problem 2 --audit-draws 8 \
  --decision-tokens 512 --max-new-tokens 8192 --run-dir runs/st-audit-m2-t512
python scripts/suffix_transport.py --mode audit "${COMMON[@]}" \
  --data-path "$AUDIT_DATA" \
  --group-size 4 --n-problems 32 --groups-per-problem 2 --audit-draws 8 \
  --decision-tokens 512 --max-new-tokens 8192 --run-dir runs/st-audit-m4-t512
```

每组先固定同题的独立前缀，再重复独立续写。每次为各 live 前缀各生成一个后缀，
形成真实独立 Full-PG 均值；从预先均匀选择的 donor 后缀计算配对 donor-only、ST 和完整 RB。
RB 用所有接收者的完整拼接梯度，是昂贵机制对照；`--no-full-rb` 可以跳过。
整个审计冻结 actor，结束时核对权重 hash。

查看 `transport_draws.jsonl` 的完整序列似然、后验、真实/反事实奖励，
`groups.jsonl` 的 ESS、donor 后验质量和梯度二阶矩。
`summary.methods` 是固定前缀下的条件方差，区间按题目 bootstrap。
`summary.contrasts` 使用相同题目重采样，报告各方法减 paired donor 的方差差值及 95% 区间；
负值表示下降。不要用两条单独区间是否重叠代替配对差值检验。
`summary.population.methods` 包含前缀及题目变化，用每题独立前缀组的重复估计，
校正全局均值平方的采样噪声。每题只有一组时总体方差为 null，第二矩仍报告；
有限样本的校正值可为负，照常报告，不做启动门槛。
总体方差目前仅有点估计，不能把条件方差的区间套到总体方差上。
保留 `moments/*.npz` 的全参数向量均值，不保存每条密集梯度。

审计额外生成 Full-PG 后缀、接收者完整 backward；实际 replay 时间不是 ST 训练成本。
`lambda_optimum_in_sample_diagnostic_only` 只描述 donor/C 协方差。
若要据此调 lambda，必须用其他校准题固定值，再到新的留出题验证，
不能挑当前审计上最优 lambda 并把其下降当作留出收益。

第一轮只比较 m=2/4、固定 t=512。通过 CLI 可另做 t=1024 或更短 cap 的机制探针，
仍需原始完整长度结论。独立新数据采集不会直接复用 PR13/14 汇总或旧策略后缀。

## 分开实测的共同预算训练

```bash
for seed in 17 29 43; do
  for method in full_pg donor_only suffix_transport; do
    python scripts/suffix_transport.py --mode train "${COMMON[@]}" \
      --method "$method" --seed "$seed" --steps 100000 --budget-seconds 3600 \
      --n-problems 0 --run-dir "runs/st-${method}-seed${seed}"
  done
done
```

三方法相同起点、baseline、奖励上限、Adam 和 actor microbatch=1。
Full-PG 每题生成 m 条并全部反传；ST 生成 m 个前缀、只续写一个 donor；
高效 donor-only 每题只生成一条前缀和一条后缀，完全没有其他前缀/计分开销。
其 N 是实际起步数，绝不虚构为 m。不要将相同步数当作相同计算量。
`--strength 0 --method suffix_transport` 是配对机制消融，仍付多前缀成本。
本入口未增加 GRPO/ARRoL；正式论文仍需已有 GRPO 等竞争基线及吞吐调优。
默认 baseline=0.5；建议另用 `suffix_transport.baseline: 0.0` 的 YAML overlay
对所有方法重做配对机制对照，区分收益是否主要来自固定 baseline 的噪声抵消。
相同 actor microbatch 并不保证 Full-PG 吞吐最优，现阶段效率结论只适用于实测配置。
lambda=0 跳过后验计分，日志的 alpha、ESS、反事实奖励记 null，不能当作后验塌缩证据。

预算从 run 启动计时，包括加载、同步、checkpoint；已开始的 batch 会完成，
可能超预算一个 batch。有时间预算时每步保存 checkpoint，覆盖常规 checkpoint_every，
其磁盘空间与 I/O 成本也计入预算。`checkpoints.jsonl` 保存持久化后的时间与 reserved GPU-seconds，
`budget_checkpoint.json` 指向最后一个预算内保存的 checkpoint；`latest_checkpoint.json` 可能已超时。
若预算连初始化保存都不足，budget 文件的 path 为 null，不应把超时模型算作预算内模型。
质量曲线取预算内 checkpoint。GPU数×总 wall time 是本入口的保守计算账本，
不是 kernel GPU-active time。step-0 初始模型也留档；服务器运行成本与 smoke 分开。

现有独立完整解题评测直接加载本入口保存的 LoRA checkpoint：

```bash
CKPT=$(python -c 'import json; c=json.load(open("runs/st-suffix_transport-seed17/budget_checkpoint.json")); assert c["path"], c; print(c["path"])')
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py --generate --backend gpu_verl \
  --method full_pg --config runs/st-suffix_transport-seed17/config.yaml \
  --config configs/hardware/a100_1.yaml \
  --data-path "$EVAL" --model-path "$MODEL" \
  --checkpoint "$CKPT" --k 1 --run-dir runs/st-eval-seed17-budget
```

单卡 hardware overlay 覆盖训练的两卡账本，避免单卡评测被计成两卡。
`--method full_pg` 是旧评测入口的解码别名；结果 method 和 actor_source 从 checkpoint
读取真实训练算法，training_estimator 保存训练时的组大小、lambda 等配置。评测不共享后缀。
`first_success_problem_ids` 仅统计真实生成且核验通过的回答，未完成接收者不重复计数；
目前没有困难题标签，不能将这个总计直接当作“困难题首次成功/GPU-hour”主张。

配置、seed、源码 commit/diff、软件版本、选题 manifest、轨迹、梯度布局、
开始/结束/失败状态及实际时间写入 run 目录。同名重跑自动加 UTC 后缀，不覆盖旧结果。
中断日志保留已完成记录；下一次用新 run-dir，当前没有断点续采。

## 2026-10-06 双卡服务器运行

本轮基于 `1e4e496`，保留上一轮的 Qwen3-4B thinking、共同 zero-B actor、
BF16 forward 和 activation checkpointing。GPU0 为 HF actor，GPU2 为 vLLM；
vLLM memory utilization=0.65、max_num_seqs=32。直接批量 RPC 不需要 HTTP 并发参数。

`suffix_transport_execution_a100_2.yaml` 将八个独立 audit draw 按 draw-major 顺序
一起生成；m=4 时最多 32 条，m=2 时最多 16 条。selection RNG 与 continuation
RNG 分开，donor 在采样前确定；独立 Full-PG 后缀、自然完成者和固定 N 保持原定义。
`transport_draws.jsonl` 额外保存每条真实续写的 request seed。
49 项 CPU 回归通过，包括分批/逐次采样的 donor、seed、记录、全空间统计和 RNG 状态比较。
真实 CUDA/vLLM 的批量数值差异仍以 GPU smoke 及实际记录为准。

专用输入选择 32 道 audit 题，排除 PR13/PR14 已使用题面和 MATH-500；
与训练题面无精确重合。`input_manifest.json` 保留源数据、选题和共同 actor 的 hash。
这不检查基础模型预训练时的题目接触历史。

`supervise_suffix_transport.py` 持久化 CPU smoke、Full-PG 首次更新、GPU 短 smoke、
ST 首次更新、完整长度 smoke、m=2/4 审计、三方法预算训练和独立 MATH-500 评测队列。
每个新阶段启动前读取 run 根目录的 `queue_control.json`；`deferred_jobs` 中的阶段
记为暂缓，不伪装成已完成。主线先行时只保留 m=2/4 审计，其内部配对对照仍保留。
这能检查冻结 actor 上的后缀共享/梯度方差，不能证明训练后的准确率收益。
`--adopt-current` 可接管同一 run 的活动任务：暂停旧调度器，保留当前 GPU 子进程运行，
等子进程退出后读取实际退出码并释放旧调度器，再按控制文件调度余下阶段。

### 16384 主线执行优化

叠加 `configs/experiments/suffix_transport_optimized_16384.yaml`，并给调度器传入
`--max-new-tokens 16384 --mainline-only --execution-config configs/experiments/suffix_transport_optimized_16384.yaml`。
仍是同一模型、初始 actor、32 题、t=512、温度 1、m=2/4 和完整 full-RB。
8192 与 16384 属于不同奖励长度上限，必须使用独立 run 根目录，不合并方差统计。

full-RB 所需的完整轨迹梯度和后缀 likelihood 由同一次前向取得，保留每条轨迹的
独立梯度；其他需要后验评分的路径按最多 4 条批量评分，排除 padding 并保留 EOS。
激活检查点每隔一层启用，增加保存的激活、减少反向时的重复前向；FP32 LoRA masters
和 BF16 计算保持不变。GPU smoke 记录复用梯度误差和实际轨迹的批量评分漂移。

冻结审计使用单个生成线程预取下一组，让 vLLM 与当前组 HF 反向重叠。
全部生成 RNG、donor 选择和 vLLM RPC 仍由同一生产者按原顺序执行。
生成计时等待 GPU worker RPC 完成，不再同步 actor GPU；各阶段耗时有重叠，不能相加
当成整轮墙钟时间。训练路径没有启用该预取，因为更新 actor 后需要重新同步 adapter。

`scripts/benchmark_suffix_transport_actor.py` 使用明确标记的 teacher-forcing 压力样本
比较执行耗时、显存峰值及梯度误差；16384 压力样本重复已有 token，不是新生成结果。
整轮收益仍需要真实流水线和审计结果，不能由这个小基准直接推断。
训练沿文档使用 seed 17/29/43，每种方法各 3600 秒；三方法统一每步八题，
Full-PG 每步最多 32 条续写，ST 和高效 donor-only 每步最多八条。
质量比较读取最后一个预算内保存的 checkpoint，初始化和审计不计为训练预算收益。

队列保留每次失败目录；OOM 时等待 GPU 可用后用新目录重启当前任务。
已完成任务不重跑，但这不是单个 audit 或 ST optimizer/RNG 的断点续跑。
状态在 run 根目录 `supervisor_status.json`，命令和日志在 `queue.json` / `logs/`。
