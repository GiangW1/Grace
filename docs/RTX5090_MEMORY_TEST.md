# 三/四卡5090：rank8 / 8192-token Full-PG显存测试

本次是容量与接线测试，不是训练效果实验。代码已做本机CPU回归；没有在5090上实测。

## 布局与环境

- 当前三卡使用 `CUDA_VISIBLE_DEVICES=0,1,2` 与 `rtx5090_rollout_3.yaml`，第一张可见卡
  运行HF actor，其余两张各运行一个vLLM worker。四卡可选 `0,1,2,3` 与 `rtx5090_rollout_4.yaml`。
- 训练、前缀审计和批次审计复用这个布局。TP=1，actor不分片；单卡上限仍是32GB。
- 生成卡初始设置：显存利用率0.75、每worker最多2条序列、分块prefill预算2048。
  2048是一次调度的token预算，不是输出长度上限。引擎上下文为9280（1024+8192+64余量）。
- 使用Linux及已匹配5090的PyTorch/vLLM/驱动环境，不沿用不支持Blackwell的旧环境。
  GPU环境建立后，在源码根目录运行 `python -m pip install -e '.[gpu,dev]'`，
  DAPO parquet还需要 `pyarrow`。不要为了此配置单独覆盖vLLM要求的torch版本。
- 所有命令均在本次源码根目录运行。旧checkout需切换到包含本次改动的分支，或应用完整补丁，
  或将本次源码压缩包解压到新目录使用。不要对已应用旧版小核心补丁的目录重复应用完整补丁。
  完整补丁基于PR16底稿commit `6656fb53af616c8d3f5285f6678e9498b80c4a66`；
  PR18 在此基础上补充三卡快速诊断；使用 PR18 checkout 才能运行新增配置。

## 1. Full-PG显存测试（先跑这个）

将下面三个路径替换成服务器真实路径。MODEL是Qwen3-4B（非Base）权重目录；
本次使用`enable_thinking: false`关闭Thinking，训练和后续审计继承相同模式。
TRAIN_DATA为训练数据，EVAL_DATA为需要从训练池排除的独立评测集。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
export MODEL=/path/to/Qwen3-4B
export TRAIN_DATA=/path/to/dapo-math.parquet
export EVAL_DATA=/path/to/math500.jsonl

python scripts/train.py \
  --config configs/hardware/rtx5090_rollout_3.yaml \
  --config configs/experiments/small_core_comparison.yaml \
  --config configs/experiments/small_core_r8.yaml \
  --config configs/experiments/small_core_memory_smoke.yaml \
  --model-path "$MODEL" --data-path "$TRAIN_DATA" \
  --eval-data-path "$EVAL_DATA" --seed 17 \
  --run-dir runs/5090-core8-fullpg-memory
```

实际流程：

1. 加载小核心actor，启动两张生成卡（四卡配置则为三张），完成adapter同步。
2. 用固定普通token构造1024提示词+8192输出的合成序列，经过与Full-PG相同的
   `real_stream_backward_and_audit` 梯度路径（p=1、固定非零advantage）。
   不更新参数/优化器，恢复梯度与torch RNG；没有正确率意义。
3. 运行4步真实Full-PG，每步2题、4条轨迹，最大输出8192，真实奖励与AdamW更新。
   自然生成可能提前结束，报告另记实际用于logprob的最大序列/输出长度。
4. 保存checkpoint供审计使用。本命令不自动付费运行评测或审计。

生成器保持正常EOS行为；非Thinking可能较早结束，但仍保留8192的最大输出预算。
没有为了显存测试偷偷改变训练采样分布。
强制长序列探针是独立诊断，包含setup成本且显式标记synthetic。
开启诊断会同步CUDA并增加额外计算，不能用此run作吞吐/等墙钟效果比较。

## 2. 查看显存结果

运行目录中的主要文件：

- `memory_profile.json`：合成探针、真实algorithm_step、同步阶段的actor
  `peak_allocated_gib` / `peak_reserved_gib`，状态和实际logprob序列长度。
- `nvidia_smi.csv`：每500ms采样可由nvidia-smi看到的所有物理卡；JSON按UUID汇总
  `sampled_peak_used_mib`。这是板卡总占用，可能含其他进程，且可能漏掉瞬时峰值，
  不是actor分配器的精确峰值。用`cuda_visible_devices`和设备UUID核对实际所用卡。
- `vllm_engine.json`：actor/worker的物理卡映射和实际接受的vLLM参数。
- `startup.json`：rank对应的真实可训练维度、checkpointing、BF16等设置。
- `checkpoint.npz`：4步结束后的检查点；`trajectories.jsonl`：真实采样记录。

发生OOM会失败退出，不自动降低长度或切换参数化；尽可能保留显存与失败报告。
强制探针通过只说明这一条完整反传可执行；还需4步真实训练通过，验证生成、同步和优化器。
本测试通过也不是更大N、长时间运行或GRACE预测器模式的显存保证。
若在actor卡OOM，减少生成卡的并发不能解决actor显存；应先检查具体失败阶段。

同名目录已有结果时框架会加时间后缀，不覆盖旧实验。以控制台最后打印的`run_dir`为准；
后续审计的RUN必须指向那一份实际目录。

## 3. 分卡审计短测试（训练通过后再跑）

```bash
export RUN=runs/5090-core8-fullpg-memory

python scripts/audit.py --generate \
  --config "$RUN/config.yaml" \
  --config configs/experiments/small_core_audit.yaml \
  --config configs/experiments/small_core_audit_smoke.yaml \
  --model-path "$MODEL" --data-path "$TRAIN_DATA" \
  --checkpoint "$RUN/checkpoint.npz" \
  --run-dir runs/5090-core8-fullpg-audit-smoke
```

保留前面的`CUDA_VISIBLE_DEVICES=0,1,2`。审计会继承训练配置中的三卡设置，
第一张卡只做HF特征/梯度，其余两张生成；不再在第一张卡加载vLLM。
本次覆盖2题、每题1前缀、每前缀4个独立续写、决策点512、最大输出8192，
保存完整轨迹梯度。它只检查接线，不保证生成会达到8K，也不是充分的统计证据。
查看`vllm_engine.json`确认设备映射，`audit_summary.json`和梯度sidecar确认完成。

正式审计删除`small_core_audit_smoke.yaml`这一行，即使用16题、2前缀、16续写、
512/1024两个决策点。耗时明显增加，不建议第一条命令就跑。
普通前缀审计自动关闭pool；编程调用传入外部`cache`时，需在finally中调用
`close_gpu_audit_cache(cache)`。批次审计入口已内置这个清理。

正式训练不用`small_core_memory_smoke.yaml`，否则仍只有4步且带合成探针。
PR18 的 `--followup` 默认三卡；仍可用 `--hardware configs/hardware/rtx5090_rollout_4.yaml`
选择四卡训练/审计（同时暴露四张卡）。
