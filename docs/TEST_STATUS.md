# 测试状态

记录已跑与未跑检查。不虚构 GPU 数字。

## 本机 CPU（应随实现跑通）

- `tests/test_estimator.py`：U1 均值、U2 方差恒等式、U3 正交分量、U4 双流与 p=1 回归、梯度符号
- `tests/test_allocation.py`：p_min、自然结束、空集合、零风险/成本
- `tests/test_rng.py`：流隔离与状态恢复；缺流的 checkpoint 不能静默落到 seed 0
- `tests/test_tiny_lora.py`：token-sum（含 EOS）、固定 N、RNG 计数、零幸存、后缀泄漏负例、prefix-only p 不变、actor 必移动
- `tests/test_predictor.py`：全空间残差 Gram 项、IPW 同题同侧、新题 50/50 进 fit/hold、prompt_slice 对齐、reproject(g=)、最后 64 token 池化、熵不再取负、reservoir 挤掉 fit 题时不拿 hold 训坐标头
- `tests/test_methods.py`：八个方法入口、assemble_ghat 数值、Reward-CV 不同 q 不同 p、成功头随前缀特征变化、GRPO-short 为每题 16 起步
- `tests/test_layout_and_reduce.py`：LoRA 布局、缺名报错、全局 N、非零 f 校正
- `tests/test_data_eval_audit.py`：去重分割互斥、标记 split、嵌套 boxed、parse_rate=2/3、ρ_L≠1、DAPO 对话列表与 ground_truth、JSON 字符串字段还原、GPU chat_template 编码、保留 role/content 消息、未闭合 thinking 硬失败、超长对话先缩短 user 文本以保住 assistant 头，模板本身超长才退回截尾、未标记 DAPO 规模语料预留校准 256/审计 240 且评测拒绝整库 DAPO、审计 PLC 用 t_L 后 token 代理、verl 风格 `extra_info.index=0` 保留为题号以免 256/240 划分被静默打乱
- `tests/test_config_loop.py`：配置、checkpoint、CPU 训练写日志
- `tests/test_grpo.py`：组均值优势、GRPO 必续写、抽题在池够大时无放回以免同题两组合成一组
- `tests/test_multistep_and_eval.py`：多步训练、续训、评测/审计自行生成、Prompt-CV 用题目特征
- `tests/test_gpu_entry.py::test_u7_amp_reduce_clip_order_cpu_reference`：clip/N 的 CPU 参考
- `tests/test_gpu_entry.py`：`n_gpu>1` 在 `train()` 入口硬失败；FSDP 无 process group 硬失败；GPU 审计入口不再是 stub；vLLM `finish_reason=stop` 且 token 里没有 EOS 时标 finished，只在 `stop_reason` 属于 stop 集合时才补 token（不把 Qwen3-Base 的 `<|endoftext|>` 当成 ChatML `<|im_end|>`）；`finish_reason` 枚举 `STOP` 也标结束，`LENGTH` 不标；`SamplingParams` 带上 `stop_token_ids`、空 `stop`、`repetition_penalty=1`、`min_p=0` 且显式 `top_k=-1`；拒收 `stop_token_ids` 时硬失败，只允许丢掉 `min_p`/`repetition_penalty`；Qwen ChatML 同时停 `<|im_end|>` 与 `<|endoftext|>`；输出条数必须等于 prompt 数，且必须带 `prompt_token_ids` 且不能乱序；空 completion 报错；同题多起步若 rollout 完全相同则硬失败（list SamplingParams 只生效第一条种子会塌掉 GRPO/GRACE）；GPU 评测 avg@4 若四次生成完全相同也硬失败，避免忽略种子后把 avg@1 报成 avg@4；LoRA 同步后必须 `add_lora`、清 prefix cache 且写入新目录，请求名带 id，路径为绝对路径，保存后必须有 `adapter_config.json` 和权重文件；多步训练先 `remove_lora` 再加新适配器，避免 `max_loras=4` 挤掉当前快照；`add_lora`/`reset_prefix_cache` 不得返回未等待的 awaitable；`require_gpu_stack` 不再把没用到的 verl 当作硬依赖；GPU 评测没有 checkpoint 会报错；vLLM 拒收 seed 时硬失败而不是无种子采样；GPU 训练入口会套上方法默认预算；actor/vLLM 加载带 `trust_remote_code`；vLLM `generation_config="vllm"` 以免 Qwen3-Base 的 2048 上限裁掉评测 4096，且 TypeError 时不得丢掉该参数；vLLM 默认 `gpu_memory_utilization=0.5`、`max_model_len=5120` 以便和同卡 HF actor 共存且评测 4096 放得下；GPU 评测先写 LoRA 再释放 actor 后才建 vLLM；yaml 里过短的 `max_model_len` 也会被抬到 prompt+eval；logprob 前向不取 hidden states
- `tests/test_review2.py`：方差×成本、answer 0、split 冲突、pass@k、assemble_ghat、EOS、轨迹级 natural_finish、GRPO-short 每题 16 起步、审计 grid/嵌套前缀、frozen predictor、LoRA-only ckpt、fit/eval 按 path 切开、残差长度混合、GRPO-short 审计丢掉超过方法预算的 grid、续训检查 U 维、Reward-CV 风险头/成功头和 LoRA 参数名顺序、提前结束的路径不进入更大的 t、方差×成本用实际前缀长度、next_n 把前缀算进 token 预算、t_A 用 Var(R)、ρ_L 用 M/(M-1)、门内曲线、ρ 分母用最早前缀、审计 G 用独立 16 样本通过率、EMA 从 0.5 起步、题级 EMA 用批均值、未见题 4 样本预扫初始化 b(x)、headline t_L 用判定半边上置信界、曲线按题平均、答案已写出的前缀退出主曲线、GRPO-short 截断仍保留已解析奖励、续写剩余步数按实际前缀长度、审计 JL 在生成时降维且大 d 用 count-sketch、GRPO-short 评测默认 4096 不继承训练 1024、GPU 无 eval 段时也用 4096 以免和 GRPO-short 对不齐、next_n 改变 N 时仍保持每题 16 起步、审计/预扫/评测空续写硬失败、真实流逐条 backward 与整批 backward 的 `.grad` 一致、评测 CLI 打印 avg@k 而不是「n 次里至少对一次」、`--answers` 无配置时用 jsonl 的 n 作 k、vLLM abort 不记成自然结束、length 提前停也算截断、Wilson 与 per-t λ

## 需要服务器 GPU 再跑

- U6：FP64 参考梯度 vs 分布式实现
- U7：真实 FSDP AMP / DDP 下校正不被乘以卡数
- vLLM 两阶段与单阶段 token 分布、prefix cache 下 RNG 隔离
- 真实 math-verify 与 DAPO revision 记录
- 4×A100 / 8×5090 墙钟账本

## 本机最近一次结果（2026-09-16）

```text
186 passed, 1 skipped
```

本次覆盖审查报告的代码修复，以及主曲线的 Wilson / 答案出现前筛选、按决策点 t 求解 λ。跳过项：`test_u6_gpu_fp64_entry_is_defined`（无 CUDA/verl）。

跳过项：`test_u6_gpu_fp64_entry_is_defined`（无 CUDA/verl；stack 存在时会跑 FP64 对照，不再无条件 skip）。

## 如何跑

```bash
python -m pytest tests
python scripts/train.py --config configs/experiments/minimal.yaml --run-dir runs/cpu-smoke
```
