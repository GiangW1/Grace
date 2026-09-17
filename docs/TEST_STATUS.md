# 测试状态

记录已跑与未跑检查。不虚构 GPU 数字。

## 本机 CPU（应随实现跑通）

- `tests/test_estimator.py`：U1 均值、U2 方差恒等式、U3 正交分量、U4 双流与 p=1 回归、梯度符号
- `tests/test_allocation.py`：p_min、自然结束、空集合、零风险/成本、风险倍乘不改 p
- `tests/test_rng.py`：流隔离与状态恢复；缺流的 checkpoint 不能静默落到 seed 0
- `tests/test_tiny_lora.py`：token-sum（含 EOS）、固定 N、RNG 计数、零幸存、后缀泄漏负例、prefix-only p 不变、actor 必移动
- `tests/test_predictor.py`：全空间残差 ||G−Uf||² 与 Gram 展开一致、IPW 同题同侧、新题 50/50 进 fit/hold、prompt_slice 对齐、reproject(g=)、最后 64 token 池化、熵不再取负、reservoir 挤掉 fit 题时不拿 hold 训坐标头
- `tests/test_methods.py`：八个方法入口、assemble_ghat 数值、Reward-CV 不同 q 不同 p、成功头随前缀特征变化、GRPO-short 为每题 16 起步
- `tests/test_layout_and_reduce.py`：LoRA 布局、缺名报错、全局 N、非零 f 校正
- `tests/test_data_eval_audit.py`：去重分割互斥、标记 split、嵌套 boxed、parse_rate=2/3、ρ_L≠1、DAPO 对话列表与 ground_truth、JSON 字符串字段还原、GPU chat_template 编码、保留 role/content 消息、未闭合 thinking 硬失败、超长对话先缩短 user 文本以保住 assistant 头，模板本身超长才退回截尾、未标记 DAPO 规模语料预留校准 256/审计 240 且评测拒绝整库 DAPO、审计 PLC 用 t_L 后 token 代理、verl 风格 `extra_info.index=0` 保留为题号以免 256/240 划分被静默打乱；冲突金标整组丢掉；`\left`/`\dfrac`/`\frac{\pi}{n}` 规范化；`\text{Evelyn}` 字面匹配；verify 先看抽出的 pred；MATH-500 格式指令与 `looks_like_math500` 用加载数不是截断后的 8；health 记 baseline b 与 A 零梯度是否因 B≈0
- `tests/test_format_warmup.py`：tiny LoRA B 在格式 SFT 后会动；只编码推理开头；`format_warmup.json` 写 `trained=reasoning lead`、`gold_in_loss=false`、`eos_in_loss=false`
- `tests/test_config_loop.py`：配置、checkpoint、CPU 训练逐步写轨迹/`steps.jsonl`/`run.log`/账本/逐步快照；同名 run-dir 再跑加 UTC 后缀
- `tests/test_grpo.py`：组均值优势、GRPO 必续写、抽题在池够大时无放回以免同题两组合成一组
- `tests/test_multistep_and_eval.py`：多步训练、续训、评测/审计自行生成、Prompt-CV 用题目特征
- `tests/test_gpu_entry.py::test_u7_amp_reduce_clip_order_cpu_reference`：clip/N 的 CPU 参考
- `tests/test_gpu_entry.py`：`n_gpu>1` 在 `train()` 入口硬失败；FSDP 无 process group 硬失败；GPU 审计入口不再是 stub；vLLM `finish_reason=stop` 且 token 里没有 EOS 时标 finished，只在 `stop_reason` 属于 stop 集合时才补 token（不把 Qwen3-Base 的 `<|endoftext|>` 当成 ChatML `<|im_end|>`）；`finish_reason` 枚举 `STOP` 也标结束，`LENGTH` 不标；`SamplingParams` 带上 `stop_token_ids`、空 `stop`、`repetition_penalty=1`、`min_p=0`、`logprobs=1` 且显式 `top_k=-1`；拒收 `stop_token_ids` 时硬失败，只允许丢掉 `min_p`/`repetition_penalty`/`logprobs`；Qwen ChatML 同时停 `<|im_end|>` 与 `<|endoftext|>`；输出条数必须等于 prompt 数，且必须带 `prompt_token_ids` 且不能乱序；空 completion 报错；同题多起步若 rollout 完全相同则硬失败（list SamplingParams 只生效第一条种子会塌掉 GRPO/GRACE）；GPU 评测 avg@4 若四次生成完全相同也硬失败，避免忽略种子后把 avg@1 报成 avg@4；LoRA 同步后必须 `add_lora`、清 prefix cache 且写入新目录，请求名带 id，路径为绝对路径，保存后必须有 `adapter_config.json` 和权重文件；多步训练先 `remove_lora` 再加新适配器，避免 `max_loras=4` 挤掉当前快照；`add_lora`/`reset_prefix_cache` 不得返回未等待的 awaitable；`require_gpu_stack` 不再把没用到的 verl 当作硬依赖；GPU 评测没有 checkpoint 会报错；vLLM 拒收 seed 时硬失败而不是无种子采样；GPU 训练入口会套上方法默认预算；actor/vLLM 加载带 `trust_remote_code`；vLLM `generation_config="vllm"` 以免 Qwen3-Base 的 2048 上限裁掉评测 4096，且 TypeError 时不得丢掉该参数；vLLM 默认 `gpu_memory_utilization=0.5`；`max_model_len` 按 `prompt+生成+64` 抬（下限 5120，Pilot 评测 4096 为 5184）；GPU 评测先写 LoRA 再释放 actor 后才建 vLLM；`FinishReason.LENGTH` 规范名按截断计；logprob 前向不取 hidden states
- `tests/test_fidelity.py`：Neyman 要等真实 U+同步 heads；残差非有限报错；rank<k 零填、秩 0 不换基；prescan 不吃主 token RNG；审计无预测器时 actual m=0；单位兼容只认数字核；生成时记录 sampled logprob；GRACE 第一步 p=z=1 换基后才分配；仿射可逆且不改 φ；换基对齐符号/列；ridge 还原线性映射；γ 可还原并裁到 [0,2]；两流同一 γ；ĉ 用剩余 token；换基后空 fit/hold 不得同步；未就绪审计保持全续写；可变 p 的 γ 用 IPW；缺失 logprob 记缺失；重复 float 梯度零秩；账本总量不重叠累计
- `tests/test_review2.py`：方差×成本、answer 0、split 冲突、pass@k、assemble_ghat、EOS、轨迹级 natural_finish、GRPO-short 每题 16 起步、审计 grid/嵌套前缀、frozen predictor、LoRA-only ckpt、fit/eval 按 path 切开、残差长度混合、审计长度不绑 GRPO-short 训练 1024、续训检查 U 维、Reward-CV 风险头/成功头和 LoRA 参数名顺序、提前结束的路径不进入更大的 t、方差×成本用实际前缀长度、next_n 把前缀算进 token 预算、t_A 用 Var(R)、ρ_L 用 M/(M-1)、门内曲线、ρ 分母用最早前缀、审计 G 用独立 16 样本通过率、EMA 从 0.5 起步、题级 EMA 用批均值、未见题 4 样本预扫初始化 b(x)、headline t_L 用判定半边上置信界、曲线按题平均、答案已写出的前缀退出主曲线（只认最后一行交卷，过程 boxed 不算）、截断短前缀仍进入 64/128、审计多次续写各自记 finish_reason、GRPO-short 截断仍保留已解析奖励、续写剩余步数按实际前缀长度、审计 JL 在生成时降维且大 d 用 count-sketch、GRPO-short 评测默认 4096 不继承训练 1024、GPU 无 eval 段时也用 4096 以免和 GRPO-short 对不齐、next_n 改变 N 时仍保持每题 16 起步、审计/预扫/评测空续写硬失败、真实流逐条 backward 与整批 backward 的 `.grad` 一致、评测 CLI 打印 avg@k 而不是「n 次里至少对一次」、`--answers` 无配置时用 jsonl 的 n 作 k、vLLM abort 不记成自然结束、length 提前停也算截断、Wilson 与 per-t λ、smoke 不是 256/512 stub

## 需要服务器 GPU 再跑

- U6：FP64 参考梯度 vs 分布式实现
- U7：真实 FSDP AMP / DDP 下校正不被乘以卡数
- vLLM 两阶段与单阶段 token 分布、prefix cache 下 RNG 隔离
- 真实 math-verify（包能 import 不等于验证器在 4B 上对）
- 下载后的模型/数据 revision（代码会写 `download_meta.json`，要服务器上下完才有值）
- 单卡 4B+vLLM 墙钟；4 卡/8 卡尚未接线

## 本机最近一次结果（2026-09-17）

```text
252 passed, 17 deselected
```

命令：`python -m pytest tests -k "not complete_final_expression and not u6_gpu"`。  
`complete_final_expression` 在 Windows 上会踩 math-verify 的 WinError 6；`u6_gpu` 要 CUDA。有 stack 时 U6 会跑 FP64 对照。

本次在 master：P0/P1 之后补上审查接线（同步门、审计就绪、γ 的 IPW、稳定去均值、缺失 logprob、λ 换算、账本不重叠）。没有真实 GPU 数字。下一步是服务器上再跑 `minimal_gpu.yaml`。

## 如何跑

```bash
python -m pytest tests -k "not complete_final_expression and not u6_gpu"
python scripts/train.py --config configs/experiments/minimal.yaml --run-dir runs/cpu-smoke
```

当前工作树的 GPU 比较：`bash scripts/run_minimal_gpu.sh`。论文规模仍按 README：先 smoke Full-PG，再 Pilot。不把 tiny / 2026-09-16 两次 smoke / 那次 16 题链写成现役实测。
