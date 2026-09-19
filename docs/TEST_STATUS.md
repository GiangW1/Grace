# 测试状态

记录已跑与未跑检查。不虚构 GPU 数字。

## 离线对照审查修复（2026-09-19，本轮）

全量 CPU/替身回归：`755 passed, 1 skipped in 95.43s`。命令：`D:\Anaconda\python.exe -m pytest tests -q -o addopts='' --tb=short`。真实 CUDA 项因本机无 GPU 跳过；requests 仍有既有依赖版本警告。

新增10项覆盖两处审查问题：离线目录迁移后续训拒绝决策位置、生成上限和特征模式不兼容；墙钟矩阵的均值、pass、标准差及配对区间使用预算内检查点，同时保留实际结束分数和全部运行费用；预算不足、时间/hash/阶段证据缺失不回退最终分数；最终评测缺失不妨碍有效预算内结果。固定步数模式与原有迁移恢复、归档回归一起通过。上述矩阵记录均为合成数据，不是 GPU 实测。

先以新增用例复现失败，修复后的相关测试得到 `37 passed in 18.67s`，随后补充最终评测缺失场景并完成全量回归。矩阵 CLI 帮助和 `git diff --check` 通过。输出字段说明见 [离线并行对照](OFFLINE_PARALLEL_CONTROL_20260919.md)。

## 离线预测器与多卡 rollout 对照（2026-09-19，首次实现）

最终全量 CPU/替身回归：`745 passed, 1 skipped in 91.68s`，命令 `D:\Anaconda\python.exe -m pytest tests -q -o addopts='' --tb=short`。1项真实 CUDA 测试因本机无 GPU 跳过；requests 有既有依赖版本警告。

本轮覆盖完全离线权重/scaler/γ冻结、无 fresh/审计监督、p=1 与 Full-PG 相同 actor 更新、共享 actor 不变、calib 题隔离与 ChatML 保留、部署 beta 校准、artifact 校验及迁移恢复/归档。多卡侧使用 CPU 替身验证真实 spawn/Pipe 通信、并发分片、请求顺序/seed、兼容回退、同步失败、缺失输出、GPU入口资源清理；真实 Bash 包装器验证离线参数和评测配置传递、用户输入不被 setup 默认值覆盖。

补充了重复 seed、TP 设备账本、CPU 不记录虚构 GPU 用量、方法特征/风险协议校验；三个新增测试文件先单独得到 `24 passed in 11.85s`，随后完成上面的最终全量回归。CLI 帮助、Bash 语法、单卡/四卡配置加载与 `git diff --check` 已检查；README、新实施文档和 STATE 的23个本地文件链接均存在。没有 GPU 性能、显存、质量或 LAG 成功证据。实现及服务器四组命令见 [离线并行对照](OFFLINE_PARALLEL_CONTROL_20260919.md)。

## 此前在线固定 U 回归（2026-09-19）

```text
721 passed, 1 skipped in 80.39s
```

命令：`D:\Anaconda\python.exe -m pytest tests -q -o addopts=''`。本轮为固定基底在线监督及运行开销优化的全量 CPU/替身回归；1项真实CUDA测试因本机无GPU跳过，requests仍有既有依赖版本警告。没有新GPU训练、质量或加速数字。

新增143项覆盖：

- `test_feature_reduction.py`（85项）：legacy/response/decision、EOS/pad、批量和prompt特征与旧定义对照；CPU FP32/BF16/FP64输入，容差 `rtol=1e-6, atol=1e-7`。先复现完整hidden序列回传，再验证只回传最终特征；该样例传输元素数不代表GPU加速比。
- `test_fixed_basis_supervision.py`（8项）：一般/秩亏Gram残差、零G、FP32标签范数舍入、IPW/γ/风险拟合对照；真实tiny训练固定U且在线更新预测头，连续训练与恢复状态一致，p=1与Full-PG actor一致，停止者null、固定N和无信号时不冻结占位基。
- `test_basis_artifact.py`（23项）：固定U引用、旧内嵌格式、hash/维度/缺失文件、写失败保留旧状态、跨目录latest、目录迁移和归档依赖；显式包含NPZ也带同目录U，归档不解析pickle。
- `test_rollout_observability.py`（22项）：不吞无关TypeError、兼容回退保seed/失败费用、cached tokens缺失为null、续写全局索引、同步失败；GPU入口用CPU替身验证失败落盘及正常同步日志，日志写失败不覆盖原异常，已恢复回退不误报为后续失败。
- `test_array_hash_buffer.py`（5项）：直接连续buffer计算hash，不创建整份bytes副本；非连续/字节序/空数组/标量保持原hash协议。

完整回归后仅更新文档。训练/归档CLI帮助、候选配置按实际层叠顺序加载和 `git diff --check` 通过。服务器仍需验证CUDA数值、显存、真实生成批次/缓存、完整费用及同成本质量。代码与命令见 [实施状态§9.1](GRACE_RESEARCH_IMPLEMENTATION_PLAN_20260919.md#91-首批实施状态2026-09-19)。

同日知识收尾仅同步文档与忽略规则，未重跑全量 pytest。另核对仓库 Markdown 的本地文件链接、修改文档的锚点、README Bash 语法和六处手动训练示例的评测题排除参数；训练 CLI 帮助验证该参数有效。历史报告保留的机器绝对路径用于本机原件追溯，在 GitHub 上不能直接打开。

## 本机上一轮结果（2026-09-19）

```text
578 passed, 1 skipped in 81.64s
```

命令：`python -m pytest tests -q -o addopts=''`。这是汇总一致性与重复实现收敛后的全量CPU回归，比上一轮575项增加3项；1项CUDA测试因本机无GPU跳过。requests仍报告既有可选依赖版本警告，未影响结果。

本轮3项新增测试先复现失败，再修至通过：普通方法汇总对错配/缺失实际seed的过滤，以及实验名称不改变训练配方hash（学习率、保存频率、精度差异仍可检出）。已有训练动态、辅助成本缺损、配对统计、实际脚本入口和包装器回归同时覆盖共享函数迁移。四方法9/18历史all/post token总量重算均不变，两个baseline配置替换后的有效设置及预扫行为一致。末尾仅清理了fresh零生成判断中的未使用元组成员，另跑15项训练动态回归。

此前继续落实清单时增加的覆盖：

- `test_checkpoint_availability.py`、`test_cost_quality.py`：发布后时钟、命令起点传递、真实tiny训练→checkpoint→评测→成本曲线，预算按时间选择、重复评测冲突、缺失seed/方法、跨seed题集/硬件/配方与重复seed不混池。
- `test_minimal_budget_wrapper.py`：真实Git Bash包装器、外层命令计时和汇总器；仅训练/评测子命令为合成替身。验证不含Full-PG的显式预算、自动加评预算内checkpoint、最终别名去重、全保存点评测和硬件配置传递。
- `test_eval_repeatability.py`、`test_logprob_diagnostics.py`：actor/协议身份、逐token首分歧、串行回退、batch-size对照、失败子进程及数值误差位置/分段；另实际跑通两次全新CPU评测子进程，合成tiny输出逐token一致。
- `test_lag_joint_evidence.py`、`test_variance_cost_uncertainty.py`：边际均值达标却无联合前缀反例、独立选择、有效配对/缺失分母、整问题簇重采样、原点估计与充分统计一致、零方差抽样不补0。
- `test_training_dynamics.py`：PG与GRPO分开重算优势、停止者null、零/缺失G、题组与长度、辅助日志缺损和零生成证据；历史四方法原始日志已只读执行。
- `test_version_metadata.py`：直接读取发行包版本不导入GPU包，以及模块回退/缺失来源。

全量回归后仅将新成本字段改名为明确的“本链初始化+训练费用”（避免复用既有SFT时冒充完整部署费用），并同步文档；相关成本/时钟/包装器18项回归通过。

上一轮新增覆盖：

- `test_training_eval_exclusion.py`：外部评测题在SFT/采样前排除、不同题号/不同金标仍匹配、保留大小写及原split、来源报告不被eval读取覆盖、原文件不修改；过滤删空时仍保存排除证据。
- `test_stage_cost_timing.py`：20项可控时钟测试，覆盖准备/环境/成功/失败/空prefix路径、结果保存异常只记一次和硬件标签。
- `test_command_timing.py`：真实CPU子进程成功/非零退出/启动失败留证，阶段空隙不伪装成计算；四臂Bash替身测试验证最终独立审计也有外层计时。
- `test_archive_evidence.py`：大JSONL/审计NPZ保留、adapter/指定checkpoint选择、基础模型默认不越过大小限制、流式hash、逐项归档来源与CLI。
- `test_seed_summary_intervals.py`：配对seed Student-t区间、样本不足/缺SciPy的null、缺失错配可见、pass@4及完整starts汇总。

脚本帮助、Git Bash语法和 `git diff --check` 检查通过。这里的子进程/时钟/四臂数据均为CPU或合成测试，不是GPU效率结果。历史回答重判另列修复文档，不计入pytest数量。

此前9月19日已覆盖的效率实现：

- `test_backward_reuse_20260919.py`：一次求导复用真实G，HT权重/梯度符号、p=1、审计掩码隔离、零优势、停止者和unused参数；FP32与旧加权反向数值比较。
- `test_fixed_n_20260919.py`：关闭wall之外仍绕开旧token回收；固定N续训；四臂真实tiny训练的初始actor、warmup更新和输入序列一致，p1/m0/停止者null。
- `test_cost_feedback_20260919.py`：固定开销摊销、主成本反馈、预算/保存/恢复和配置模式；checkpoint无损紧凑存储、旧格式兼容及原子发布。
- `test_predictor_signal_fixes.py`：零残差冷启动风险尺度、Reward-CV当前q特征、γ诊断与监督组成；跨题预测交叉矩的反向预测反例、显式谱等价及退化输入。
- `test_batch_interventions.py`：同完整G的p1/m0/uniform冻结干预、训练checkpoint对应N、原始梯度方差×主token代理及零分母；开启额外干预不改变原审计样本/统计。
- `test_eval_execution_trace.py`：eval/audit引擎继承seed、显式覆盖、逐请求seed/token hash与batch回退轨迹。
- `test_mechanism_summary.py`、`test_mechanism_wrapper.py`：配对统计的actor/seed/评测协议/终点/完整输入与N核对；完整及后warmup成本、缺证据与已观测零生成的区分；真实Git Bash运行四臂编排，昂贵训练/审计以替身命令代替。

Git Bash分别对 `run_minimal_gpu.sh`、`run_mechanism_gpu.sh`、`run_matched_cost_gpu.sh` 做语法检查，批审计CLI帮助和 `git diff --check` 通过。没有新GPU训练、质量或端到端速度验证。确定修复、研究候选及服务器运行方法见 [9月19日实现说明](GRACE_EFFICIENCY_REMEDIATION_20260919.md)。

## 历史本机结果（2026-09-18）

```text
421 passed, 1 skipped in 47.24s
```

命令：`python -m pytest tests -q -o addopts=''`。没有按名称排除判分测试；Windows math-verify 由有超时的隔离进程执行。1项真实CUDA测试因本机无GPU跳过。安装环境仍报告 requests 的可选依赖版本警告，不影响本次测试结果。

吸收`GRACE_review.md`的本轮增加50项测试；包含冻结批次的真实CPU更新比较，未以训练曲线逐点相同代替p=1正确性验证。两个审计CLI帮助和五份新覆盖配置的加载检查通过。

新增回归覆盖：

- `test_predictor_remediation.py`：dual/primal一致、加权精确基、固定尺度、独立校准、Uniform/Reward风险设计、空校准、冻结新题诊断、旧状态兼容和极端数值报告；按题交叉拟合、监督年龄权重、过期监督处理和部分秩建基。
- `test_backend_remediation.py`：零优势省算且保留Adam语义、BF16运算与FP32参数、更新前原始行为token比较、probe异常/非有限状态、非有限梯度拒绝更新。
- `test_runtime_remediation.py`：原子写入故障、一次压缩、回退恢复、完整RNG/学习状态一致、强制评测/终点快照、失败与累计成本、HT baseline、路径迁移、源码快照和协议错配；用合成时钟验证墙钟反馈、保存成本、批边界超额及续训预算。
- `test_measurement_remediation.py`：投影与尺度、detect/report隔离、原始审计可重判、判分反例、seeded题目选择、批量seed保持、失败成本、CLI方法标签；本机还实际执行了9月17日归档的只读回归。
- `test_experiment_remediation.py`：四方法共享actor的完整CPU链、seed为单位的汇总、来源缺失不配对、含/排除文件清单与归档字节hash。
- `test_deeper_remediation.py`：当前策略监督采样隔离、当前批更新不变、Reward-CV的q与baseline分离、停止者前缀行为比较、缺行为分数不补造、深层配置恢复一致、实际终点评测及跨seed/链来源校验。
- `test_batch_audit.py`：固定批全空间HT/CV、条件选择方差、独立预扫baseline、历史Adam状态回放、冻结状态污染检测、真实tiny CLI及模拟GPU入口；这些不代替真实CUDA验证。
- `test_config_loop.py`：仅覆盖warmup的短运行正常保存，并报告实际分配状态。
- `test_review_baseline.py`：固定baseline、独立prescan先验平滑、历史HT不裁剪、旧状态兼容。
- `test_review_allocation.py`：异质成本下uniform收缩保预算、零风险p=1、枚举全部选择状态验证全空间HT期望、m=0与停止者null。
- `test_review_integration.py`：GRPO移除预扫后同批更新不变、冻结R−b、补全开关不改当批风险分配、全组件p=1回归、最终p接入抽样/记录、baseline恢复与有效配置。
- `test_review_audit.py`：非正交/降秩基底的全空间误差分解、旧包缺证、审计开关接线、固定/平滑baseline和三阶段方向误差；CPU回放不等于CUDA逐位验证。

另已通过 Git Bash 对 `scripts/run_minimal_gpu.sh`、`scripts/run_matched_cost_gpu.sh` 的语法检查，训练与批审计CLI帮助检查，以及 `git diff --check`。实际GPU训练、质量和加速均未验证。改动与50项发现的对应关系见 [修复记录](GRACE_REMEDIATION_20260918.md)。

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
- `tests/test_gpu_entry.py`：`n_gpu>1` 在 `train()` 入口硬失败；FSDP 无 process group 硬失败；GPU 审计入口不再是 stub；vLLM `finish_reason=stop` 且 token 里没有 EOS 时标 finished，只在 `stop_reason` 属于 stop 集合时才补 token（不把 Qwen3-Base 的 `<|endoftext|>` 当成 ChatML `<|im_end|>`）；`finish_reason` 枚举 `STOP` 也标结束，`LENGTH` 不标；`SamplingParams` 带上 `stop_token_ids`、空 `stop`、`repetition_penalty=1`、`min_p=0`、`logprobs=1` 且显式 `top_k=-1`；拒收 `stop_token_ids` 时硬失败，只允许丢掉 `min_p`/`repetition_penalty`/`logprobs`；Qwen ChatML 同时停 `<|im_end|>` 与 `<|endoftext|>`；输出条数必须等于 prompt 数，且必须带 `prompt_token_ids` 且不能乱序；空 completion 报错；训练和评测允许独立请求 seed 恰好产生相同回答，不据重复样本设置硬失败；保留逐请求 seed 以核对生成协议；LoRA 同步后必须 `add_lora`、清 prefix cache 且写入新目录，请求名带 id，路径为绝对路径，保存后必须有 `adapter_config.json` 和权重文件；多步训练先 `remove_lora` 再加新适配器，避免 `max_loras=4` 挤掉当前快照；`add_lora`/`reset_prefix_cache` 不得返回未等待的 awaitable；`require_gpu_stack` 不再把没用到的 verl 当作硬依赖；GPU 评测没有 checkpoint 会报错；vLLM 拒收 seed 时硬失败而不是无种子采样；GPU 训练入口会套上方法默认预算；actor/vLLM 加载带 `trust_remote_code`；vLLM `generation_config="vllm"` 以免 Qwen3-Base 的 2048 上限裁掉评测 4096，且 TypeError 时不得丢掉该参数；vLLM 默认 `gpu_memory_utilization=0.5`；`max_model_len` 按 `prompt+生成+64` 抬（下限 5120，Pilot 评测 4096 为 5184）；GPU 评测先写 LoRA 再释放 actor 后才建 vLLM；`FinishReason.LENGTH` 规范名按截断计；logprob 前向不取 hidden states
- `tests/test_fidelity.py`：Neyman 要等真实 U+同步 heads；残差非有限报错；rank<k 零填、秩 0 不换基；prescan 不吃主 token RNG；审计无预测器时 actual m=0；单位兼容只认数字核；生成时记录 sampled logprob；GRACE 第一步 p=z=1 换基后才分配；仿射可逆且不改 φ；换基对齐符号/列；ridge 还原线性映射；γ 可还原并裁到 [0,2]；两流同一 γ；ĉ 用剩余 token；换基后空 fit/hold 不得同步；未就绪审计保持全续写；可变 p 的 γ 用 IPW；缺失 logprob 记缺失；重复 float 梯度零秩；账本总量不重叠累计
- `tests/test_review2.py`：方差×成本、answer 0、split 冲突、pass@k、assemble_ghat、EOS、轨迹级 natural_finish、GRPO-short 每题 16 起步、审计 grid/嵌套前缀、frozen predictor、LoRA-only ckpt、fit/eval 按 path 切开、残差长度混合、审计长度不绑 GRPO-short 训练 1024、续训检查 U 维、Reward-CV 风险头/成功头和 LoRA 参数名顺序、提前结束的路径不进入更大的 t、方差×成本用实际前缀长度、next_n 把前缀算进 token 预算、t_A 用 Var(R)、ρ_L 用 M/(M-1)、门内曲线、ρ 分母用最早前缀、审计 G 用独立 16 样本通过率、EMA 从 0.5 起步、题级 EMA 用批均值、未见题 4 样本预扫初始化 b(x)、headline t_L 用判定半边上置信界、曲线按题平均、答案已写出的前缀退出主曲线（只认最后一行交卷，过程 boxed 不算）、截断短前缀仍进入 64/128、审计多次续写各自记 finish_reason、GRPO-short 截断仍保留已解析奖励、续写剩余步数按实际前缀长度、审计 JL 在生成时降维且大 d 用 count-sketch、GRPO-short 评测默认 4096 不继承训练 1024、GPU 无 eval 段时也用 4096 以免和 GRPO-short 对不齐、next_n 改变 N 时仍保持每题 16 起步、审计/预扫/评测空续写硬失败、真实流逐条 backward 与整批 backward 的 `.grad` 一致、评测 CLI 打印 avg@k 而不是「n 次里至少对一次」、`--answers` 无配置时用 jsonl 的 n 作 k、vLLM abort 不记成自然结束、length 提前停也算截断、Wilson 与 per-t λ、smoke 不是 256/512 stub

## 需要服务器 GPU 再跑

- U6：FP64 参考梯度 vs 分布式实现
- U7：真实 FSDP AMP / DDP 下校正不被乘以卡数
- vLLM 两阶段与单阶段 token 分布、prefix cache 下 RNG 隔离
- 真实 math-verify（包能 import 不等于验证器在 4B 上对）
- 下载后的模型/数据 revision（代码会写 `download_meta.json`，要服务器上下完才有值）
- 单卡 4B+vLLM 墙钟；4 卡/8 卡尚未接线

## 历史本机结果（2026-09-17，修复前）

```text
252 passed, 17 deselected
```

命令：`python -m pytest tests -k "not complete_final_expression and not u6_gpu"`。  
当时 `complete_final_expression` 在 Windows 上会踩 math-verify 的 WinError 6；该路径现已修复并纳入全量测试。`u6_gpu` 要 CUDA。有 stack 时 U6 会跑 FP64 对照。

当时在 master：P0/P1 之后补上审查接线（同步门、审计就绪、γ 的 IPW、稳定去均值、缺失 logprob、λ 换算、账本不重叠）。这是历史记录；后来的 `3c03ce9` 已有 [9月18日真实实验](MINIMAL_RESULTS_REVIEW_20260919.md)。9月19日的后续修复尚无新 GPU 结果，不能把修复前链的实测归到新代码。

## 如何跑

```bash
python -m pytest tests -q -o addopts=''
python scripts/train.py --config configs/experiments/minimal.yaml --run-dir runs/cpu-smoke
```

当前工作树的 GPU 比较：`bash scripts/run_minimal_gpu.sh`。论文规模仍按 README：先 smoke Full-PG，再 Pilot。不把 tiny / 2026-09-16 两次 smoke / 那次 16 题链写成现役实测。
