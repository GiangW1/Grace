# GRACE：ICLR 论文框架 v3（按《论文框架模板 v3》§-1 到 §13 与附录填写）

工作标题：Learning Can Finish Before Reasoning: Gradient Completion for Compute-Efficient RLVR
方法名：GRACE（Gradient-completion RLVR with Adaptive Continuation and unbiased Estimation）；代码与图表里的显示名用 GRACE-GC，以区别两篇同名工作（梯度对齐数据筛选的 GRACE，arXiv:2605.13130；对比策略优化的 GRACE，arXiv:2510.04506）。
现象名：LAG（Learning-Answer Gap，学习完成与答案完成之间的时间差）
目标会议：ICLR，正文 9 页，附录不限，需附 Reproducibility Statement、Ethics Statement 与 AI 使用声明。ICLR 2027 的摘要与全文截止日期据另一份框架为 2026-09-18 与 2026-09-25（AoE），需核实；按 §13 的预算，本轮赶不上，目标定为下一个可投的截止。

文中带 `【目标 …】` 的数字是预注册目标，实验跑完后换成真实数字；`【核实】` 的引用要在写正文前逐条查 arXiv 或 DBLP。研究的三个目标不变：提高真实有效探索；降低达到同等能力所需的真实训练总算力；提高训练后模型在相同测试预算下独立完整解题的 pass@1 与 pass@k。三者分别报数，不能用"生成 token 更少"或"预测器更准"代替任何一项。

---

## §-1 叙事原型与贡献类型

主贡献类型是 T2（新洞察）：一条 RLVR 轨迹对参数更新的贡献，常常在它的最终答案确定之前就已经可以预测。辅贡献类型是 T1（新方法）：GRACE 把这个现象变成可执行、无偏、同时省生成和反向传播的训练算法。不主打 T6，因为预测梯度加校正的组合已有前作（GPCV、Randomized Telescoping），题目级的 rollout 预算分配也已有 VIP 与 VIGOR。

叙事原型选 E（现象型）：发现现象，解释它，再利用它。9 页正文的分配：Intro 1.25 页，现象与归因 §4 约 1.75 页，形式化与理论 §5 0.75 页，方法 §6 1.75 页，实验 §8 2.75 页，相关工作 0.5 页，结论 0.25 页。全文只讲一句话："学习可以在推理结束前完成，所以训练的计费单位可以比一条完整推理更小"。GRACE 的每个模块都要能被这句话解释，多决策点、更大的预测器、教师模型、PRM 一律不进主配置。

"学习完成"是相对于冻结的策略快照、梯度目标、LoRA 参数化和误差容忍度的操作定义。它不表示模型已经知道答案，也不表示未来梯度为零。

---

## §0 Idea Card（单一事实源，Pilot 推翻归因时先改这里）

| 字段 | 内容 |
|---|---|
| Problem | GRPO、DAPO、RLOO 这一族 RLVR 方法把"写完整条推理并拿到可验证奖励"当作学习信号的原子计费单位。由此产生的浪费我们叫 Post-Learning Compute（PLC）：轨迹的更新贡献已经基本确定之后，仍然被继续生成，并被整条反向传播。 |
| Why | 策略梯度 \(G=(R-b)\nabla_\theta\log\pi(\tau\mid x)\) 在给定前缀 \(h\) 后拆成两部分：对已做出决策的记账 \((R-b)\,g_h\)，符号要等答案揭晓，噪声为 \(\mathrm{Var}(R\mid h)\|g_h\|^2\)；对尚未做出决策的更新 \((R-b)\,g_s\)，方向在结果出来之前已经确定（Lemma 1：校准基线下，成功率为 \(q\) 的决策 token 上梯度条件方差是 \((1-2q)^2\mathrm{Var}(R)\)，\(q=1/2\) 时为零）。只要前缀携带的未结算梯度能量占比不大，而隐藏状态已经决定了接下来要走哪条路，这条轨迹的更新就比它的答案先"完成"。这是归因假设，由 §4 的方差分解与残差分解检验。 |
| Insight | 对还没做出的决策，更新方向在结果揭晓前就已经确定；对已经做出的决策，更新的符号要等答案。一条轨迹待学的部分越多在前方、待判的部分越少在身后，它的学习就完成得越早。 |
| Method | GRACE 在固定决策点读前缀隐藏状态，(i) 用学到的低维梯度基底预测完整轨迹梯度 \(m(h)=Uf(h)\)；(ii) 按剩余更新误差与剩余成本之比 \(p(h)=\mathrm{clip}(\sqrt{\hat r/(\lambda\hat c)},p_{\min},1)\) 随机决定是否续写；(iii) 用 Horvitz-Thompson 校正 \(\widehat G=m+\frac{Z}{p}(G-m)\) 保持对完整轨迹梯度无偏；(iv) 被停止的轨迹省掉后缀生成和整条反向传播，省下的算力用于更多独立起步。 |
| Delta | ARRoL、ESPO、DPPO、Speculative Rejection 的决策变量是答案或奖励的可预测性，被剪掉的轨迹贡献为零或有偏。VIP 与 VIGOR 按题目成功概率或已完成组的奖励方差分配完整 rollout 的数量，粒度是题目或轮次。GRACE 的决策发生在本条轨迹的后缀和奖励都还不存在时，决策变量是更新的可预测性，被停止的轨迹仍以 \(m(h)\) 贡献且整体无偏：一条"结果五五开但前缀常规、后续走法已定"的轨迹会被停，一条"看起来会对但前缀做了大胆选择"的轨迹会被继续写。GPCV 对已完成样本预测梯度，只省反向传播；GRACE 同时省生成。Randomized Telescoping 给出无偏截断框架但不用前缀信息，Prop. 2 证明这种均匀截断不可能改善"方差乘成本"。"Beyond the 80/20 rule"回答哪些 token 携带梯度，我们回答梯度在什么时候变得可预测。 |
| Headline Result | 现象：Qwen3-4B-Base 在 DAPO-Math-17k 上，答案仍不确定且更新信号非零的前缀里，512 token 处剩余更新不确定性 \(\rho_L\le\)【目标 0.5】；中等难度题里 \(t_L<t_A\) 的轨迹占比 ELF ≥【目标 55%】；GRPO 的 rollout 算力中 PLC 占【目标 ≥30%】。训练：同 A100-hours 账本下 MATH-500、AIME24、AIME25、AMC23 平均 pass@1 +【目标 ≥2.0】；达到 GRPO 最终精度所需总算力 ≤【目标 0.65×】；每 A100-hour 上经真实验证的困难题首次成功数 +【目标 ≥30%】。方法开销 ≤【目标 8%】总算力。 |
| Who cares | 凡是"产生一个样本比对它做梯度更新贵"的学习系统，长 CoT、多步智能体、代码与工具调用、长回复 RLHF，都默认了"任务完成才能学习"。LAG 说明值得付费获取的对象不只是更确定的结果，也可以是尚未确定的更新信息；这给 RL 算力缩放多了一个可优化的轴：每单位算力的有效样本数。 |

---

## §1 Title

主选：Learning Can Finish Before Reasoning: Gradient Completion for Compute-Efficient RLVR

备选：
- When Learning Finishes Before the Answer: Gradient Completion for RLVR
- The Learning-Answer Gap: Trajectories Stop Teaching Before They Stop Reasoning

保留 "Can"：表达存在且可利用，不表示每条轨迹都如此。不用 "Unbiased GRPO"、"Zero-Cost Reasoning" 之类超出定义目标的说法。

---

## §2 Abstract（英文草稿，约 230 词，每句后标兑现章节）

1. Reinforcement learning with verifiable rewards (RLVR) trains reasoning models by generating complete reasoning trajectories and backpropagating through them. [§3 P1]
2. This treats finishing the reasoning as the atomic unit of learning. We show that this unit is systematically too large: a substantial fraction of rollout compute is spent after a trajectory's contribution to the update is already determined. [§4.2, Fig. 2]
3. We call this the learning-answer gap (LAG). Through repeated independent continuations of identical prefixes, we separate the irreducible suffix uncertainty of the full-parameter update from predictor error, and show that for a measurable set of prefixes the update is predictable while the answer is still uncertain and not yet emitted. The gap is not explained by problem difficulty, reward confidence, post-answer text, or near-zero gradients; it follows from the fact that the update to decisions not yet made is determined before the outcome is: under a calibrated baseline, the gradient on an unresolved binary decision token has conditional variance \((1-2q)^2\mathrm{Var}(R)\). [§4, §5 Lemma 1]
4. GRACE exploits this. It predicts the full-trajectory gradient from the prefix in a learned low-dimensional gradient subspace, continues trajectories stochastically in proportion to their remaining update error rather than their predicted success, and corrects the prediction with an unbiased Horvitz-Thompson estimator, so stopped trajectories still update the model while saving both generation and backpropagation. [§6]
5. On Qwen3-4B-Base across math and code RLVR under a strict end-to-end compute ledger, GRACE matches GRPO's final accuracy with 【0.65×】 the A100-hours, improves average pass@1 by 【2.0】 points at equal compute, and finds 【30%】 more verified solutions to hard problems per A100-hour. [§8.3, §8.7]
6. Remaining update uncertainty is thus an actionable signal for pricing training compute. Code, gradient-audit data, and complete compute ledgers are released. [§10]

三个数字与 Table 2、Fig. 5、Table 4 逐位一致；第 2 句的 "substantial fraction" 对应 Fig. 2c 的 PLC 数字。若结果只满足部分目标，摘要按"证实、无显著差异、未达预注册门槛"分别写，不把门槛填成结果。

---

## §3 Introduction

P1 背景，三句：RLVR 让 LLM 靠可验证奖励学推理；长 CoT 下生成占训练时间 60% 到 80%【引 DAPO 或 DeepScaleR 的系统数据，核实】；现有效率工作沿"少生成哪些题、少生成哪些轨迹"展开，包括动态采样、题目级 rollout 分配（VIP、VIGOR）、基于奖励预测的早停（ARRoL、ESPO）。落点句：The unit of training computation is usually an entire verified trajectory（英文句号保留在正文里）

P2 现象。这些方法都在判断题目或路径的成功率、奖励方差、是否值得完成；我们追问另一个问题：这段后缀还会改变多少完整更新？我们对真实 RLVR 训练做了梯度审计（Fig. 1 左、Fig. 2）：在同一前缀上独立续写多次并计算完整 LoRA 梯度。对答案仍不确定、更新信号非零的前缀，前缀 25% 处完整梯度的条件方差已降到初值的【0.5】，而答案的条件方差仍在【0.8】以上。我们把"更新基本确定之后仍花的算力"叫 PLC，它占 GRPO rollout 总算力的【≥30%】。用两类前缀建立直觉：一类最终可能对也可能错，但更新方向稳定；另一类成功概率高，但后缀里有尚未学会的重要选择，更新仍难预测。两类都要用真实样本展示。

P3 归因。给定前缀，梯度拆成两项：对已做出决策的记账 \((R-b)g_h\)，符号要等答案；对尚未做出决策的更新 \((R-b)g_s\)，方向在答案之前就定了。Lemma 1 给出后者的精确形式：对成功率为 \(q\) 的决策，校准基线下成功和失败两种结局在该 logit 上的梯度分别是 \((1-q)^2\) 和 \(q^2\)，条件方差 \((1-2q)^2\mathrm{Var}(R)\)。结果越接近五五开，这个决策的更新越确定。Lemma 2 进一步说明，只知道一个成功概率 \(q(h)\) 不能决定 \(\mathbb E[G\mid h]\) 中的奖励与后缀得分函数的交叉矩，所以答案置信度不是更新不确定性的代理。核心句：Knowing what a trajectory will teach is not the same as knowing whether it will succeed。 §4 用同前缀独立续写直接测量，而不是用预测器自己的置信度自证，并排除题目难度、奖励可预测性、答案后文本、近零梯度、子空间幻觉、特殊基线、LoRA 初始化退化七种替代解释。

P4 方法。GRACE 只讲一条链：前缀，完整更新预测及剩余误差，抽样购买后缀，完成样本真值纠偏。停止样本不是丢弃，也不是赋予伪失败奖励。训练后测试时移除所有辅助模块，模型独立完整作答。Theorem 1 用 Pilot 可测量给出可盈利条件，Prop. 2 说明没有前缀信息的截断不可能盈利。

P5 结果预告：三笔账分别报（同账本 pass@1、到达同精度的算力比、每 A100-hour 困难题首次成功数），再报代价：单次起步的估计方差上升，收益来自每次起步更便宜。最后一句机制回收：GRACE 把 PLC 从【30%】降到【<10%】，批梯度对全完成 oracle 梯度的余弦不降。

P6 贡献列表：

- C1（LAG 现象与审计协议）We identify and quantify the learning-answer gap in real RLVR. We define per-trajectory learning and answer completion times \((t_L, t_A)\), the population curves \(\rho_L(t), \rho_A(t)\), and post-learning compute; we decompose the residual into irreducible suffix uncertainty, subspace omission, and predictor error; and we show that LAG survives seven alternative explanations. [§4, Fig. 2 与 3, Table 1]
- C2（GRACE）We propose GRACE, an RLVR trainer whose continuation decisions are driven by remaining update error rather than predicted outcome. It combines a prefix gradient predictor in a learned gradient subspace, Neyman-optimal stochastic continuation, and a Horvitz-Thompson correction that keeps the raw policy gradient unbiased while saving both generation and backpropagation. [§6, Alg. 1, Table 3]
- C3（理论）We prove a decision-token lemma explaining why update uncertainty resolves before outcome uncertainty, show that uniform truncation without prefix information cannot improve variance per compute, and give a closed-form profitability condition in Pilot-measurable quantities. [§5, App. A, Fig. 6c]
- C4（实证）Under an identical end-to-end compute ledger on Qwen3-{1.7B, 4B, 8B}-Base across math and code RLVR, GRACE reaches GRPO's accuracy with 【0.65×】 compute, improves pass@1 by 【2.0】 points at equal compute, and increases verified hard-problem discoveries per A100-hour by 【30%】; we also characterize where the gain vanishes. [§8.3 到 8.7]

对齐：C1 对应 §4、Fig. 2、Table 1；C2 对应 §6.1 到 6.3、Table 3、Table 2；C3 对应 §5、App. A、Fig. 6c；C4 对应 §8.3、Fig. 5、Table 4、Fig. 7。

---

## §3.5 Figure Storyboard

图注是数字填入后的成稿模板。没有实验时只做空坐标和标为 schematic 的机制示意，不画虚构曲线。

| 编号 | 内容 | 章节 | Caption 草稿 |
|---|---|---|---|
| Fig. 1 | 左：一条真实 AIME 风格轨迹，x 轴为前缀 token 数（0 到 2048），红线 \(\rho_A(t)\)，蓝线 \(\rho_L(t)\)，标出 \(t_L\) 与 \(t_A\)，两线之间阴影即 LAG，文本框摘出 \(t_L\) 附近的推理片段，并标注两条真实续写的不同奖励。右：四个基准平均 pass@1 对累计 A100-hours，GRPO 灰、GRACE 蓝，虚线标 GRPO 最终精度，箭头标 0.65×。案例、示意、实测分区标注。 | §3 | A trajectory's update becomes predictable (blue) long before its answer does (red): here the gradient is 80% determined at token 480 while the answer is still a coin flip. Redirecting the compute after \(t_L\) to new rollouts lets GRACE reach GRPO's final accuracy with 【0.65×】 the A100-hours. |
| Fig. 2 | (a) 总体曲线 \(\rho_L(t)\)、\(\rho_A(t)\)，只统计答案未定且信号非零的前缀，均值加 95% 分层 bootstrap CI，x 轴双刻度（绝对 token 与相对长度），标出每个位置的存活分母与删失率。(b) \(t_A-t_L\) 直方图，按题目难度三组着色。(c) PLC 占比柱状图，GRPO 训练的不同 step。 | §4.2 | Among prefixes whose answer is still uncertain and whose update is non-trivial, update uncertainty falls below 0.5 at 25% of trajectory length while answer uncertainty stays above 0.8; 【57%】 of medium-difficulty trajectories finish learning before finishing answering, and 【33%】 of GRPO's rollout compute is spent after \(t_L\). |
| Fig. 3 | (a) 条件方差三项分解随 \(t\) 的堆叠面积图（前缀记账项、后缀项、交叉项）。(b) 逐轨迹 LAG 对前缀梯度能量占比 \(\|g_h\|^2/\mathbb E\|g_\tau\|^2\) 的散点。(c) Lemma 1 曲线 \((1-2q)^2\) 与实测决策 token 梯度条件方差比的散点。(d) 残差诊断分解随 \(t\)：不可约后缀项、子空间遗漏项、坐标预测误差项。 | §4.3 | LAG is largest when the prefix carries little unsettled credit while the hidden state already fixes the upcoming decisions; the empirical variance ratio on unresolved decision tokens follows the predicted \((1-2q)^2\) law, and the residual is dominated by irreducible suffix uncertainty rather than by the 8-dimensional basis. |
| Fig. 4 | 方法总览：决策点前缀，隐藏状态（detach），预测头输出 \(f,\hat r,\hat c\)，\(p(h)\)，Bernoulli \(Z\)，预测流（k 维累加，批末一次 \(U\cdot\)）与真实流（续写加 \(1/p\) 加权反向传播）。虚线是历史标签更新，红色边界标"当前未来信息不得穿越"。手绘或 TikZ。 | §6.0 | GRACE continues a trajectory in proportion to its remaining update error, not its predicted success; stopped trajectories update the model through the predicted stream, and the real stream corrects the prediction without bias. |
| Fig. 5 | (a) 数学与代码两张独立曲线：平均 pass@1 对累计端到端 A100-hours，机制轨道与实用轨道全部方法，3 seeds 逐 seed 加均值。(b) 每 A100-hour 累计的困难题首次验证成功数（HVD 覆盖曲线）。(c) 最终 pass@k，k 取 1、4、8。 | §8.3 | Under an identical compute ledger GRACE reaches GRPO's final accuracy at 【0.65×】 compute and discovers 【1.3×】 more verified hard-problem solutions per A100-hour. |
| Fig. 6 | (a) 训练中每步 PLC 占比，GRPO 对 GRACE，同 Fig. 2c 坐标。(b) \(p(h)\) 对真实剩余残差的校准图，按预测风险十分位。(c) Theorem 1 预报的可盈利比对实测"到达同精度算力比"，每点一个设置。 | §8.5 | GRACE removes the post-learning compute measured in Fig. 2 (【33%→7%】) without biasing the update (cosine to the oracle gradient 【0.97】), and the closed-form profitability condition predicts the measured savings across 【9】 settings. |
| Fig. 7 | (a) 增益对平均轨迹长度（Countdown、数学、代码）。(b) 增益对题目饱和度分箱。(c) 增益对模型规模 1.7B、4B、8B。(d) 预测器在已见题与未见题上的残差。(e) 二维相图：横轴全空间 NRUE，纵轴剩余成本占比，颜色为实测方差乘成本比。 | §8.5 | The gain grows with trajectory length and vanishes for short-reasoning tasks, saturated prompts, and unseen prompts in the first epoch; the high-residual, low-remaining-cost corner of the phase diagram is unprofitable, as Theorem 1 predicts. |
| Fig. 8 | 两条轨迹并排：一条结果五五开但被停止，一条看似会对但含未学会的决策被继续写；再加一条失败例：预测风险低但正交补残差大，\(1/p\) 权重放大校正。图中附 ID、checkpoint、seed。 | §8.6 | GRACE stops a coin-flip trajectory whose update is already determined and continues a likely-correct trajectory that still contains an unlearned decision. |

翻图顺序：问题与结果（1），现象（2），归因（3），方法（4），有效（5），为何有效与理论验证（6），边界（7），直觉（8）。9 页放不下时 Fig. 1 与 Fig. 4 合成首图，Fig. 8 进附录，Fig. 2 和 Fig. 6 不能删。配色固定：GRACE 蓝，GRPO 灰，答案不确定性红，更新不确定性蓝。

---

## §4 Key Observation：Learning Finishes Before Reasoning（约 1.75 页）

### 4.1 Setup：梯度审计协议

模型快照用 Qwen3-4B-Base 加 LoRA（q、v，rank 16）的三个点：共同格式 warmup 结束、GRPO step 50、GRPO step 150，后两个复用 §8 主实验的 checkpoint。不用纯初始化快照做主分析：LoRA 的 B 矩阵初始为零，此时对 A 的梯度恒为零，会制造假的"可预测"。Qwen3-1.7B-Base 做小规模复核。

题目分三份，按 problem_id 划分，互不重叠：审计集 240 题，从 DAPO-Math-17k 抽出并从训练流中移除，按 base 模型 16 样本估计的 pass rate 分三层各 80 道（hard [0.1, 0.4)、medium [0.4, 0.7)、easy [0.7, 0.9]），饱和题单独做边界分析；校准集 256 题，用来定阈值、决策点和超参；预测器训练用训练流的历史数据。审计集的续写不得用于训练预测器、选秩或选检测点。

采样分两级。每题采 16 条独立前缀；在决策点 \(t\in\{64,128,256,384,512,768,1024,1536\}\) 处截断，从该前缀独立续写 16 次到结束（最长 2048 token，温度 1.0，不用 top-p 截断，保证采样策略与求导策略一致），续写用前缀缓存。总量约 49 万条续写，含梯度计算约 6 A100-days；超预算时把 \(t\) 网格缩到 {128, 256, 512, 1024}，续写数降到 12。在一个预先冻结的随机子集上把续写加到 96 条，用来缩小完成时刻的置信区间。自然提前结束的轨迹保留在分母并记为删失，不强行补足不存在的前缀。

对每条完整轨迹算 per-sample LoRA 梯度 \(G=(R-b(x))\nabla_\theta\log\pi(\tau\mid x)\)，序列内 token log-prob 取和，\(b(x)\) 用同快照下另外独立的 16 个样本估 pass rate。存储用固定的稀疏随机投影 \(P\in\mathbb R^{4096\times d}\)（JL，保范数，种子固定）后的 4096 维向量，另外每题保留 32 条完整梯度用于基底与正交补分析。每条后缀记录：token 序列、终局奖励、\(b(x)\)、完整梯度范数与基底坐标、答案首次出现位置、结束原因、生成与训练成本、模型和基底 hash。奖励用 math-verify 规则验证器，二元，没有 LLM judge。

### 4.2 Phenomenon：两条曲线的分离（Fig. 2）

统计量定义如下，§8.5 用同一定义回收。

- 剩余答案不确定性 \(\rho_A(t)=\mathbb E_h[\mathrm{Var}(R\mid h_t)]/\mathrm{Var}(R\mid x)\)，每题归一化后跨题平均。
- 剩余更新不确定性（oracle 预测器）\(\rho_L(t)=\mathbb E_h\,\mathbb E[\|G-\mathbb E[G\mid h_t]\|^2\mid h_t]\,/\,\mathbb E\|G-\mathbb E[G\mid x]\|^2\)。分子用续写的样本方差（乘 \(M/(M-1)\) 修正），分母用同题所有轨迹。它是全参数空间的量，不能用 \(\|U^\top G-f\|^2\) 冒充。
- 非平凡前缀的两个门：答案未定，\(\mathrm{Var}(R\mid h_t)>0.15\)；更新信号非零，条件均值能量 \(\|\mathbb E[G\mid h_t]\|^2/\mathbb E[\|G\|^2\mid x]\ge\kappa=0.05\)。条件均值能量用 A/B 两半续写的均值内积估计，避免把均值估计噪声当成信号。主统计量 \(\rho_L\)、ELF、LAG 指数都只在过了两个门的前缀上算。答案已定的前缀两条曲线都低，近零梯度的前缀任何剪枝都能停，二者都不构成现象。
- 答案完成分两个时刻：\(\tau_{\rm emit}\) 是首次完整可解析答案出现的位置；\(t_A=\min\{t:\mathrm{Var}(R\mid h_t)\le\delta\}\)，\(\delta=0.05\)。主现象要求发生在 \(\tau_{\rm emit}\) 之前且 \(q(h)\) 的 Wilson 区间落在 (0.05, 0.95) 内。
- 学习完成时刻 \(t_L=\min\{t:\rho_L(h_t)\le\varepsilon\}\)，\(\varepsilon=0.2\)，用 \(\rho_L\) 的上置信界判定。阈值在校准集上先定，附录给 \(\varepsilon\in\{0.1,0.2,0.3\}\)、\(\delta\in\{0.02,0.05,0.1\}\) 的敏感性。
- ELF 是 \(\Pr[t_L<t_A]\)；LAG 指数是曲线间面积 \(\int_0^1(\rho_A-\rho_L)\,d(t/L)\)。
- PLC 是 GRPO 训练里 \(t>t_L\) 的生成 token 成本加这些轨迹整条反向传播成本，除以 rollout 总成本，用实测 token 时序和 backward 计时换算。

统计细节：标准误和 bootstrap 以题目为最高层，内部保留前缀、后缀嵌套；二项奖励用 Wilson 区间；报告每个 token 位置的存活分母。完成时刻是阈值定义，不保证沿每条路径单调，报告越界后再次升高的比例，不做单调拟合。

目标：门内前缀上 \(\rho_L(512)\le0.5\)，\(\rho_A(512)\ge0.8\)；medium 层 ELF ≥ 55%；LAG 指数 ≥ 0.12；PLC ≥ 30%。LAG 指数低于 0.05 触发 §13.2 的回改流程。

防伪划分：\(t_L\) 的判定用 16 条续写里的前 8 条，报告用后 8 条。

### 4.3 Attribution：两个分解与七种替代解释（Fig. 3，Table 1）

条件方差分解。写 \(G=(R-b)(g_h+g_s)\)，\(g_h=\nabla\log\pi(h\mid x)\) 给定前缀是常量，\(g_s=\nabla\log\pi(\text{suffix}\mid h)\) 是随机量：

\[
\mathrm{Var}(G\mid h)=\mathrm{Var}(R\mid h)\,\|g_h\|^2+\mathrm{tr}\,\mathrm{Var}\bigl((R-b)g_s\mid h\bigr)+2\,g_h^\top\mathrm{Cov}\bigl(R,(R-b)g_s\mid h\bigr).
\]

第一项是对已做决策的记账噪声，不被任何机制衰减；第二项是后缀项，Lemma 1 说决策 token 在这里被衰减，低熵执行 token 的贡献本来就接近零。LAG 大的条件是前缀梯度能量占比 \(\|g_h\|^2/\mathbb E\|g_\tau\|^2\) 小，同时隐藏状态已经决定了接下来的走法。三个测量对应它：Fig. 3a 三项随 \(t\) 的堆叠面积图；Fig. 3b 逐轨迹 LAG 对前缀能量占比的散点，预期 Spearman ≤ −0.4；决策在状态里先于在文本里：用与被测模型不同源的 LLM 给每条完整轨迹的解法打类别标签（附录报告与人工抽检 50 条的一致性），再用线性探针从 \(t\) 处隐藏状态预测该标签，探针准确率曲线应明显领先于解法在文本中出现的位置。

残差诊断分解。对正交基底 \(U\) 和条件均值 \(\mu(h)=\mathbb E[G\mid h]\)：

\[
\mathbb E[\|G-Uf(h)\|^2\mid h]=\mathrm{tr}\,\mathrm{Cov}(G\mid h)+\|(I-UU^\top)\mu(h)\|^2+\|U^\top\mu(h)-f(h)\|^2.
\]

第一项高说明该前缀确实还没学完；第二项高说明 8 维表示不够；第三项高说明预测器没学到规律。Fig. 3d 画三项随 \(t\) 的变化，分 LoRA A/B、q/v、层组各报一份，防止总均值被少数大范数层掩盖。

Lemma 1 的实证版放 Fig. 3c：找熵大于 1 bit 且后续 16 条续写成功率 \(q\in(0.2,0.8)\) 的 token，测该 logit 方向梯度的条件方差与奖励方差之比，对照 \((1-2q)^2\)。

替代解释排除放 Table 1，每行一个解释、一个对照、一个"若成立会看到什么"、一个实测：

1. 只是题目难度。题内跨前缀比较 \(t_L\)，报告题内方差占总方差比例（目标 ≥ 40%），三个难度层内 LAG 都成立。
2. 只是奖励可预测。训练三个预测器：只看题目的题级预测器、看 \(\hat q(h)\) 与题目的奖励条件预测器、看隐藏状态的前缀预测器，比较残差；再加一个离线 oracle \(q(h)\)（由 16 条续写直接估计）条件的对照。若 LAG 只是奖励可预测的副产品，后两者应持平；预期前缀预测器残差显著更低（paired bootstrap，p<0.01）。
3. 答案后的冗余文本。以 \(\tau_{\rm emit}\) 处截断重算所有统计量，LAG 指数下降不超过 20%。
4. 只是梯度接近零。非零信号门已排除；再与"按 \(\|G\|\) 或其预测值选样"的剪枝对照（消融 A12）。
5. 只是 8 维投影看起来稳定。Fig. 3d 的正交补能量；全空间 \(\rho_L\) 同样低，否则 8 维版本不能自证。
6. 只是选了特殊基线。\(b\) 取 0、0.5、历史题级基线三种重算。
7. 只是 LoRA 初始化退化。至少两个已学习快照，分别审计 A/B 与 q/v。

强 critic 对照分两层：先看给定其最佳奖励概率能否预测残差；再训练一个 critic 驱动的前缀 bootstrap 基线，用最终能力检验。

### 4.4 现象边界

记录 LAG 指数随下面四个量的变化，喂给 H5 和 Limitations：题目饱和度；轨迹长度（预期越长越大）；训练进程（策略熵下降后 \(\rho_A\) 也会提前下降，LAG 可能收窄或前移）；任务类型（Countdown 短推理预期 LAG 小）。

### 4.5 Design Implication 与训练前的盈利预报

两个分解直接给出方法的三个要求：要一个从前缀预测更新而不是预测结果的预测器（§6.1）；续写与否由剩余更新误差 \(r(h)\) 和剩余成本 \(c(h)\) 决定（§6.2）；若误差主要在 \(U\) 的正交补，真实纠偏不可删（§6.3）。

Pilot 数据还能在花主实验预算之前算两件事。一是 Oracle-GRACE：用续写的条件均值当理想预测器 \(m^*=\mathbb E[G\mid h]\)，代入 Theorem 1 算理想可盈利比，这是 headline 的上界。二是用实际训练出的预测器重算一遍，得到预报值。两者都进 §8.5 和 App. D。

---

## §5 Problem Formulation

### 5.1 符号

| 符号 | 含义 |
|---|---|
| \(x,\ \tau,\ h\) | 题目；完整轨迹（自然结束或共同最大长度）；决策点处的前缀（含题目） |
| \(R\in\{0,1\}\)，\(b(x)\) | 可验证奖励；题级基线，历史拟合、批内冻结并 detach |
| \(\theta\)，\(d\) | 全部 q/v LoRA 参数及其维数。Qwen3-4B-Base 为 36 层、隐藏维 2560、32 个 query head、8 个 KV head、head_dim 128，rank 16 时 \(d=36\times16\times[(2560+4096)+(2560+1024)]=5{,}898{,}240\)【核实 config】 |
| \(G=(R-b)\nabla_\theta\sum_t\log\pi_\theta(a_t\mid x, a_{<t})\) | 完整轨迹的原始策略梯度，token 和，含前缀与后缀；目标量 |
| \(m(h)=Uf(h)\) | 前缀梯度预测，\(U\in\mathbb R^{d\times k}\) 基底 |
| \(r(h)=\mathbb E[\|G-m(h)\|^2\mid h]\)，\(c(h)\) | 全空间剩余更新误差；剩余成本（后缀生成加整条反向传播） |
| \(p(h)\in[p_{\min},1]\)，\(Z\sim\mathrm{Bern}(p(h))\) | 续写概率与续写指示 |
| \(\widehat G\) | GRACE 单起步估计量 |

### 5.2 定义

Def. 1（答案完成时刻）\(t_A(\tau;\delta)=\min\{t:\mathrm{Var}(R\mid h_t)\le\delta\}\)；\(\tau_{\rm emit}\) 为首次可解析答案位置。
Def. 2（学习完成时刻）\(t_L(\tau;\varepsilon)=\min\{t:\mathbb E[\|G-\mathbb E[G\mid h_t]\|^2\mid h_t]\le\varepsilon\,\mathbb E\|G-\mathbb E[G\mid x]\|^2\}\)。
Def. 3（LAG）\(\Delta(\tau)=t_A-t_L\)；ELF、LAG 指数、PLC 见 §4.2。

优化对象：在总算力 \(B\) 下最小化批梯度估计量的"方差乘单起步成本" \(\mathcal V\cdot\mathcal C\)，等价于最大化每单位算力的有效样本数 \(B/(\mathcal V\mathcal C)\)，约束是估计量对 \(\mathbb E[G]\) 无偏。

### 5.3 假设

- A1（批内冻结）一批 \(N\) 个起步的生成、奖励、梯度计算结束前，不更新 actor、\(b\)、\(m\)、\(U\)、\(\hat r\)、\(\hat c\)；每批一次梯度更新。\(\lambda\) 可由本批已观察的全部前缀确定，但在所有 \(Z\) 抽样前固定。
- A2（预测器批前冻结）预测器只用历史数据训练；按抽样概率校正选择性获得的标签。
- A3（独立性与正概率）\(Z\) 在后缀生成前由独立 RNG 抽取，只依赖已观察信息；\(p\ge p_{\min}>0\)，记录实际 \(p\)。
- A4（固定起步分母）以预先确定的 \(N\) 个起步构成一次更新，不除以随机完成数或 \(\sum 1/p\)，不在墙钟截止时只保留先完成的轨迹。
- A5（基线与本批无关）\(b(x)\) 只依赖历史。
- A6（成本近似）平方根分配用剩余成本的可加近似；GPU 批处理、等待与验证成本以端到端实测裁决。
- A7（相同目标）核心理论只讨论原始轨迹策略梯度；GRPO 组内归一化、PPO clip、多 epoch、Adam 预条件、全局裁剪不自动继承期望等价，实用比较另设轨道（§8.1）。

### 5.4 理论结果（证明进 App. A）

Lemma 1（决策 token 梯度）。设最后一个二元决策成功概率 \(q=\sigma(z)\)，基线 \(b\)。该 logit 上的 REINFORCE 梯度在成功与失败时分别为 \((1-b)(1-q)\) 与 \(bq\)；\(b=q\) 时条件方差为 \(q(1-q)(1-2q)^2=(1-2q)^2\mathrm{Var}(R)\)，在 \(q=1/2\) 时为 0。\(b\) 由历史 pass rate 拟合，对中等难度题近似校准，Fig. 3c 实证。

Lemma 2（奖励概率不够）。后缀得分函数的条件均值为零，所以

\[
\mathbb E[G\mid h]=(q(h)-b(x))\,\nabla_\theta\log\pi_\theta(h\mid x)+\mathbb E\bigl[(R-q(h))\,\nabla_\theta\log\pi_\theta(\text{suffix}\mid h)\,\big|\,h\bigr].
\]

第二项是奖励与后缀得分函数的交叉矩，一个标量 \(q(h)\) 不决定它，更不决定条件方差。这解释了比较对象，不是"任何 critic 都做不到"的不可能性定理：若拥有精确且可微的策略条件成功函数及其参数敏感性，可以得到额外信息。

Prop. 1（无偏）。A1 到 A5 下，\(\mathbb E[\widehat G\mid h]=\mathbb E[G\mid h]\)，对任意 \(m\)、任意不满秩的 \(U\) 成立。

Prop. 2（方差恒等式与无免费午餐）。\(\mathrm{tr}\,\mathrm{Cov}(\widehat G)=\mathrm{tr}\,\mathrm{Cov}(G)+\mathbb E[(1/p(h)-1)\,r(h)]\)。每次起步的方差不会低于完整真梯度。若 \(m\) 与 \(p\) 都不依赖 \(h\)，则对任意 \(p\in(0,1]\) 有 \(\mathcal V_{\rm new}\mathcal C_{\rm new}\ge\mathcal V_{\rm full}\mathcal C_{\rm full}\)。

Prop. 3（Neyman 分配）。在期望续写成本约束下最小化附加方差的解是 \(p^*(h)=\min\{1,\sqrt{r(h)/(\lambda c(h))}\}\)；这是标准方差成本工具的实例，不作为首创。

Theorem 1（可盈利条件）。记 \(c_0\) 为前缀成本（含预测器前向），\(A=\mathcal V_{\rm full}-\mathbb E[r(h)]\)，\(S=\mathbb E[\sqrt{r(h)c(h)}]\)。若 \(A>0\) 且未截断的 Neyman 解满足 \(p^*\le1\)，则 \(\min_\lambda\mathcal V_{\rm new}\mathcal C_{\rm new}=(\sqrt{c_0A}+S)^2\)，GRACE 严格盈利当且仅当

\[
\bigl(\sqrt{c_0A}+\mathbb E[\sqrt{rc}]\bigr)^2<(c_0+\mathbb E[c])\,\mathcal V_{\rm full}.
\]

该闭式要求 \(A>0\)、\(c_0>0\) 且驻点对应的全部概率满足实际上下界。\(A\le0\) 不能推出不盈利：风险异质时，即使 \(m=0\)，自适应抽样也可能降低方差×成本，有限分布反例见 `.planning/research/PITFALLS.md`。增加 \(p\le1\) 与 \(p\ge p_{\min}\) 约束不会降低最小值；当上述正值条件成立但驻点不可行时，放松问题的闭式是受限最优值的下界，具体可行分配的目标值才给出上界。受限情形直接计算实际分配的方差×成本。同题多轨迹分块时用整块梯度协方差，不把方差机械除以轨迹数。这些是理论适用范围，不作为实验启动或继续条件；Fig. 6c 是待完成的实证验证。

Remark（Adam）。无偏性是对原始梯度说的，Adam 的归一化是非线性的，所以 §8.5 同时报方差和最终精度，并补一个小模型 SGD 对照。

---

## §6 Proposed Approach：GRACE

### 6.0 Overview（Fig. 4）

一个 rollout 批有 \(N_x\) 道题、每题 \(N_{\rm start}\) 次起步。每条轨迹先生成到决策点 \(t_d\)（默认 512，校准集从 {256, 512, 1024} 里选；提前 EOS 的直接完成，\(p=Z=1\)，\(f=0\)）。决策点上做一次 actor 前向取隐藏状态（detach），预测头输出 \((f(h),\hat r(h),\hat c(h))\)，按预算解出 \(\lambda\)，得到 \(p(h)\)，抽 \(Z\)。\(Z=1\) 的进入真实流：vLLM 在同一快照下续写，带权反向传播。\(Z=0\) 的进入预测流：只留 k 维坐标。批末合成 \(\widehat G_{\rm batch}\) 交给 AdamW，只 step 一次。预测器用本批完成轨迹的审计梯度更新，供下一批用。三个模块是一条依赖链：预测器的输出是分配的输入，分配的 \(p\) 是校正的权重，去掉任何一个都退化成已知方法（6.4）。

### 6.1 前缀梯度预测器

动机来自 4.3：要利用"更新已定"，必须从前缀读出更新的方向和剩余误差。用预测奖励 \(\hat q(h)\) 代替只能捕获 Lemma 2 的第一项，漏掉交叉矩（消融 A3）。

基底 \(U\in\mathbb R^{d\times k}\)：对历史 reservoir 里最近 512 条完成轨迹的完整梯度（减去题级均值）做随机 SVD 取前 k 个方向，每 32 个 actor update 刷新一次，刷新时记录 basis_id 并重投影 reservoir 中的真梯度，不把旧坐标当新基底的标签。默认 \(k=8\)，消融 {4, 8, 32, 128}。显存与内存：\(U\) 在 BF16 下 94.4 MB（k=8），FP32 主副本 188.7 MB；512 条 FP32 完整梯度的 reservoir 约 12.1 GB，放主机内存，内存不足时改 128 条并记录。这些数字不能合并宣传为"只需 95 MB"。

特征 \(\phi(h)\)：决策 token 处最后一层与第 \(L/2\) 层隐藏状态、前缀最后 64 个 token 的均值池化、前缀 token 熵统计（均值、最大值、高熵 token 数）、前缀长度、题级 \(b(x)\)。

预测头：坐标头 MLP 2560→256→k，回归 \(U^\top G\)；风险头 2560→64→1，输出 \(\hat r>0\)；成本头用前缀长度、预计长度桶和批处理状态做轻量回归，也可以用常数替代（消融 A7）。风险头与坐标头不共享参数。

风险监督针对全空间残差。对每条有真值的轨迹算

\[
e_i=\|G_i\|^2-2f(h_i)^\top U^\top G_i+f(h_i)^\top(U^\top U)f(h_i),
\]

BF16 基底不完全正交时保留 Gram 项。风险头损失用 \(e/\hat r+\log\hat r\)，其条件最优值是 \(\mathbb E[e\mid h]\)；不用 \(\log e\) 的 MSE 再取指数，那预测的是几何均值。

训练：真值标签来自完成轨迹的审计子样本（6.3），纳入概率为 \(p\cdot s\)，损失按 \(1/(p s)\) 逆概率加权；不加权的话训练分布偏向高 \(\hat r\) 的轨迹，预测器会系统性高估残差。历史数据按题目拆分：先训练 \(f\)，再在没有用于拟合该 \(f\) 的历史样本上重算 \(e\)、训练风险头；\(U\) 改变则用新 \(U\) 重算目标。AdamW，lr 1e-3，每批 2 个 epoch，前 20 步全部 \(p=1\) 作为 warm-up。

### 6.2 残差驱动的随机续写

动机来自 4.5 的第二个要求。按预测成功率停止（ARRoL、ESPO 的做法）会停掉五五开但更新已定的轨迹，保留看起来会对但没有新信息的轨迹，而且删除即有偏（消融 A2、A4）。只用范数、只用长度也各漏一部分信息（A12、Reward-CV）。

\(p(h)=\mathrm{clip}\bigl(\sqrt{\hat r(h)/(\lambda\hat c(h))},\,p_{\min},\,1\bigr)\)，\(p_{\min}=0.2\)，消融 {0.1, 0.2, 0.4}。\(\hat c\) 只含增量成本：后缀生成、整条 actor forward/backward、验证；所有起步都已支付的前缀与预测器成本计入总账，不重复放进 \(c\)。

预算旋钮 \(\beta\)：令期望续写成本占"全部续写"成本的比例 \(\mathbb E[p\,\hat c]/\mathbb E[\hat c]=\beta\)，默认 0.5，消融 {0.3, 0.5, 0.7}；每批在已观察前缀上对 \(\lambda\) 二分求解，20 次迭代。这是期望成本约束，不是每批硬预算：抽完 \(Z\) 后不删超额完成者，否则 \(p\) 不再是纳入概率。

算力回流：\(N_{\rm start}=\lceil N_{\rm GRPO}\cdot C_{\rm full}/\hat C_{\rm new}\rceil\)，用最近 10 批实测成本，使每批实测 A100 秒与 GRPO 对齐；主表以累计 A100-hours 对齐，不以步数对齐。

只设一个决策点，避免概率连乘。多决策点扩展放 App. E。

### 6.3 无偏梯度补全与双流更新

预测不完美，不校正就是有偏剪枝。只对完成轨迹按 \(1/p\) 重加权、丢掉 \(m\)，无偏但方差大，正是 Prop. 2 的场景（Uniform-HT）。保留 \(m\) 但不加 \(1/p\) 权重，方差低但有偏，偏差量是 \((1-p)(\mathbb E[G\mid h]-m)\)（消融 A4'）。

单起步估计量 \(\widehat G_i=m(h_i)+\frac{Z_i}{p_i}\bigl(G_i-m(h_i)\bigr)\)。对 \(N\) 个固定起步：

\[
\frac1N\sum_i\widehat G_i=\frac1N\sum_{i:Z_i=1}\frac{G_i}{p_i}+\frac{U}{N}\sum_i\Bigl(1-\frac{Z_i}{p_i}\Bigr)f(h_i).
\]

校正项包括完成者的负预测项，不是只给停止者加 \(Uf\)。

真实流：对 \(Z=1\) 的轨迹做 token 级损失 \(-\frac1N\sum_{i:Z_i=1}\frac{R_i-b(x_i)}{p_i}\sum_t\log\pi_\theta(a_{it}\mid\cdot)\)，一次 backward。分母固定为 \(N\)。

预测流：k 维累加器 \(a=\sum_i(1-Z_i/p_i)f(h_i)\)，批末一次 \(Ua/N\) 加到参数梯度上。顺序是：backward，混合精度 unscale，跨卡全局归一化与求和，加入同样全局归一化的校正，统一 clip 一次，optimizer.step 一次。单元测试确认 DDP 不会把校正乘以卡数。

审计子采样：主 actor 更新不需要所有完成轨迹的 per-sample 梯度。以 \(s=1/8\) 在完成者中再抽样，在 optimizer.step 之前、同一 \(\theta\) 上用 autograd.grad 取 per-sample LoRA 梯度，不覆盖 actor 的 `.grad`；审计 RNG 与奖励独立。这些梯度进 reservoir 和预测器训练，纳入概率 \(p\cdot s\)。

基线 \(b(x)\)：该题历史奖励的 EMA（\(\alpha=0.7\)），epoch 0 用 4 样本预扫初始化，预扫成本记入账本，GRPO 也给这一选项并报两种。不用 GRPO 组内均值的原因：组均值依赖同组其它样本的奖励，随机停止使其部分不可观测，Prop. 1 不再成立。std 归一化按 Dr. GRPO 去掉。

超长处理：到 2048 未给答案记 \(R=0\)，所有方法一致。停止轨迹的 reward 字段记 null，不填 0。

Algorithm 1（GRACE 一个训练步）

```
Input: policy π_θ frozen for this batch; predictor (U, f, r̂, ĉ) frozen; baseline b; budget β; p_min; N starts; audit prob s
1  for each prompt x and start i: generate prefix h_i up to t_d tokens (vLLM, T=1, no top-p)
2  early-EOS trajectories: Z_i = 1, p_i = 1, f_i = 0
3  actor prefill on prefixes -> detached φ(h_i); (f_i, r̂_i, ĉ_i) <- predictor(φ(h_i))
4  λ <- bisection on visible prefixes s.t. Σ_i min(1, sqrt(r̂_i/(λ ĉ_i))) ĉ_i = β Σ_i ĉ_i
5  p_i <- clip(sqrt(r̂_i/(λ ĉ_i)), p_min, 1);  Z_i ~ Bernoulli(p_i) from a separate RNG;  log (p_i, Z_i)
6  continue generation only for Z_i = 1 with the same snapshot (prefix-cached); R_i <- verifier
7  real stream: loss <- -(1/N) Σ_{Z_i=1} (R_i - b(x_i)) / p_i · Σ_t log π(a_t); backward
8  audit: for Z_i = 1 sampled with prob s, per-sample G_i via autograd.grad at the same θ -> reservoir (inclusion p_i·s)
9  predicted stream: a <- Σ_i (1 - Z_i/p_i) f_i;  grad_pred <- U a / N
10 unscale; global reduce; add grad_pred once; clip once; AdamW step once
11 update predictor on reservoir with weights 1/(p_i s); update b(x) EMA; every 32 steps refresh U and reproject
12 ledger: log generation, prefill, backward, audit, predictor, SVD, verifier, sync and idle time per component
```

### 6.4 减法与开销

去预测器等于 Uniform-HT（Randomized Telescoping），去校正等于 ARRoL 型剪枝，去分配等于 Uniform-CV（GPCV 型控制变量），Table 3 按这个结构组织。

开销正文一句、App. C 一张表：预测器约 1.1M 参数；\(U\) 94.4 MB；决策点 prefill 一次，约为生成 512 token 成本的 1/20；审计 per-sample 梯度只在 \(s=1/8\) 的完成者上算；随机 SVD 每 32 步一次，不到 1 A100 分钟。目标是总开销不超过 8% 总 A100-hours。

两个账本。部署账本从共同初始 checkpoint 起，计入 warmup、生成、actor forward/backward、审计、预测器训练与读取、基底、权重同步、KV 恢复、验证、CPU verifier 等待、调度空闲，按 GPU 数乘墙钟记，不累加重叠 kernel 时间。研究账本另报离线 pilot、超参搜索、失败运行、最终评价、复跑，合计不超过 64 A100-days。每次部署必须支付的预测器开销不能藏进研究账本。

设计选择表（App. E）：

| 选择 | 放弃的替代 | 理由 | 消融 |
|---|---|---|---|
| 决策变量用 \(\hat r\) | 预测成功率 \(\hat q\)、范数、长度 | Lemma 1、Lemma 2 | A2、A3、A12 |
| HT 校正并保留 \(m\) | 丢弃停止轨迹；保留 \(m\) 不加权 | 无偏；Prop. 2 | A1、A4、A4' |
| Neyman \(\sqrt{r/c}\) | 均匀 \(p\)、阈值硬停 | Prop. 3 | A5 |
| \(k=8\) | 全维回归、只用范数、随机基底 | 显存与方差平衡；真值覆盖遗漏 | A6 |
| 单决策点 512 | 多点、256、1024 | 避免概率连乘 | A8 |
| 历史 EMA 基线 | GRPO 组均值；\(b\)=0、0.5 | A5 | A9、A11 |
| IPW 训练预测器 | 不加权 | 完成轨迹是有偏抽样 | A10 |
| 审计 \(s=1/8\) | 全部完成者算 per-sample 梯度 | 控制标签成本 | A13 |
| \(\hat c\) 预测 | 常数 | 小增益 | A7 |
| 固定起步分母 \(N\) | 固定幸存数、自归一化权重 | 保住精确目标 | 单元测试 U4 |
| 算力回流为更多起步 | 不回流 | 收益来源 | A14 |

---

## §7 Hypotheses → Experiments

数值是预注册目标，与显著性阈值分开报告；每项保存 effect、CI、target_met、statistically_supported 四个字段。

| 假设 | 可证伪陈述 | 指标 | 实验 | 贡献 |
|---|---|---|---|---|
| H0 现象 | 门内前缀上 LAG 指数 ≥ 0.12，medium 层 ELF ≥ 55%，PLC ≥ 30%；Table 1 七项替代解释被排除；残差诊断分解中不可约项占主导 | \(\rho_L, \rho_A, t_L, t_A\)，ELF，LAG，PLC，三项分解 | §4，Fig. 2 与 3，Table 1 | C1 |
| H1 有效性 | 同账本下平均 pass@1 ≥ GRPO +2.0（3 seeds，paired bootstrap p<0.05）；到达 GRPO 最终精度的算力 ≤ 0.65× | pass@1，time-to-target | Table 2，Fig. 5a | C2、C4 |
| H1' 探索 | HVD ≥ GRPO 1.3×；最终 pass@8 ≥ GRPO +1.0 | HVD，pass@k | Fig. 5b 与 5c | C4 |
| H2 机制 | GRACE 训练中 PLC 降到 <10%；批梯度对全完成 oracle 梯度的偏差与 0 无显著差，余弦 ≥ 0.95；打乱 \(\hat r\) 分配后精度回落到 Uniform-CV 水平，打乱 \(m\) 坐标后方差升、精度降；风险十分位与真实残差单调 | PLC，偏差与余弦，干预后 pass@1，校准图 | Fig. 6a 与 6b，Table 5 | C2 |
| H2' 理论预报 | Theorem 1 用 Pilot 量预测的算力比与实测算力比 Spearman ≥ 0.8（≥ 9 个设置） | 预报对实测 | Fig. 6c | C3 |
| H3 泛化 | 代码 RLVR 与 1.7B、8B 上增益同号且显著；OOD 评测（AIME25、OlympiadBench）增益不小于 ID | 各基准 pass@1 | Table 2，Fig. 7c | C4 |
| H4 效率 | 总开销 ≤ 8%；Pareto 曲线全程占优；成本乘批方差比 <1 | A100-hours 分项，Pareto，\(\chi\) | Table 4，Fig. 5a | C2 |
| H5 边界 | 增益随平均轨迹长度上升；Countdown 上增益 ≤ 0.5 点且不显著；饱和题子集无增益；第一个 epoch 未见题上预测器残差接近题级预测器；相图右下角（高残差、可省成本少）不盈利；\(p\equiv1\) 时数值回归 Full-PG | 增益对长度、饱和度、epoch；相图 | Fig. 7 | §10 |

H2 必须同时有"PLC 消失"（同坐标回收）和干预实验，只有主表不够。

---

## §8 Experiments

### 8.1 Setup

任务与数据。数学是主任务：训练用 DAPO-Math-17k 固定 revision，去重后移除审计集与校准集，其余均匀采样；评测 MATH-500（avg@4，预注册主指标）、AIME24 与 AIME25（avg@32）、AMC23（avg@16）、OlympiadBench（avg@4，OOD）；评测统一 T=0.6、top-p 0.95、max_new_tokens 4096，所有方法同预算独立完整作答，训练时的预测器与停止器全部移除。代码是第二任务族：训练用 TACO 官方 train 中有有效测试、Python 可运行的题，按 EASY/MEDIUM 分层取至多 5000 题（或 Code-R1-12k，二选一后固定）；评测 LiveCodeBench 固定 release（时间窗在训练数据之后）、HumanEval+、MBPP+；全部测试通过才记 \(R=1\)；规模 Qwen3-4B、2 seeds。边界探针用 Countdown（TinyZero 配方），Qwen3-1.7B，1 seed 两个方法。去污染：用来源 URL、problem_id、规范化题干 hash 与近重复筛查排除训练与评测重合，保存移除清单。

模型与训练。骨干 Qwen3-4B-Base 为主，Qwen3-1.7B-Base 与 Qwen3-8B-Base 各 1 seed；LoRA q、v，rank 16，alpha 32，dropout 0；训练生成最长 2048 token，prompt 最长 1024，T=1.0 无 top-p。每次 update 固定 \(N\)=512 起步（64 题乘 8，GRACE 由算力对齐自动加起步）。AdamW，LoRA lr 1e-4，预注册小网格 {5e-5, 1e-4, 2e-4}，weight_decay 0，梯度裁剪上限统一并记录触发比例。若 base 模型不能稳定输出可解析解答，先在训练 split 上做所有方法相同的格式 warmup，费用计入部署账本，在校准阶段锁定；不为 GRACE 单独 SFT。Full-PG 与 GRPO 必须在校准集上产生有意义学习，否则先排查共同管线。

实现。代码基用 verl（vLLM rollout 加 FSDP actor），GRACE 自行实现 prefix pause/resume、RNG 隔离（题目流、轨迹 token、续写、选择、审计各自独立）、精确归一化和参数校正；所有方法在同一代码基、同一 vLLM 版本、同一硬件上跑。硬件 4 张 A100-80GB 为规划假设，拿到机器后填实际显存。seeds 固定 {17, 29, 43}。

两条比较轨道。机制轨道 M：Full-PG、Uniform-HT、Uniform-CV、Reward-CV、Prompt-CV、GRACE 共享同一原始梯度目标、\(b\)、\(N\)、优化器与输入分布，回答"收益是否来自更新补全与续写分配"。实用轨道 P：GRPO、GRPO-short、Dr. GRPO 加动态采样、ARRoL、VIP、VIGOR，以及能公平复现的 DPPO、ESPO、GPCV-style、G2RL，各保留自己的目标，但对齐 backbone、LoRA、数据、硬件预算、调参资源和测试协议，回答"是否值得实际使用"。

测试协议与统计。主表每题独立生成 \(n\) 个答案，无偏 pass@k 估计 \(1-\binom{n-c}{k}/\binom nk\)；报告可解析率、截断率。3 seeds 报均值与标准差和逐 seed 原值；题目配对 bootstrap 保留同题样本；题目层与种子层不确定性分开报告，三 seed 的 seed-bootstrap CI 标注不稳定；确认性多重比较用 Holm 校正。同一方法的调参预算相同（3 组超参各短跑 1 A100-day，只在校准集上选）。GRPO 同时报组均值优势和历史 EMA 基线两种，取更好者作对手。评测全是规则验证器；解析边界案例盲抽 100 条报争议率。

### 8.2 Baselines

| 轨道 | 方法 | 核心思路 | 复现方式 |
|---|---|---|---|
| M | Full-PG | 每个起步完整 rollout、真奖励、原始 \(G\) | 自跑 |
| M | Uniform-HT | \(m=0\)，均匀续写加 \(1/p\) 校正 | GRACE 去预测器 |
| M | Uniform-CV | 与 GRACE 同预测器，均匀 \(p\) | GRACE 去分配 |
| M | Reward-CV | 保留 \(m\)；风险只用奖励概率、其熵、难度与长度，校准到相同期望成本 | GRACE 换风险头 |
| M | Prompt-CV | \(m,\hat r\) 只读题目，同容量同标签 | GRACE 换特征 |
| P | GRPO（DeepSeekMath，2024）；Dr. GRPO 归一化 | 组相对优势，全轨迹 | verl 自复现 |
| P | GRPO-short | 每题 16 起步、最长 1024 token，超长记 0 | GRPO 改配置 |
| P | Dr. GRPO 加动态采样与超长处理 | DAPO 中与单次更新兼容的部分；完整 DAPO 单列参考行 | 自复现 |
| P | ARRoL（arXiv:2603.24840，核实） | 前缀成功率在线剪枝，有官方代码 | 官方代码起点 |
| P | VIP（arXiv:2602.01601，核实） | 题目级成功概率到梯度方差到 rollout 分配 | 自复现 |
| P | VIGOR（arXiv:2607.22002，核实） | 按已完成组奖励方差逐轮增加 rollout | 自复现 |
| P | DPPO、ESPO、GPCV-style、G2RL（核实） | 事后剪枝；失败早停；已完成样本梯度控制变量；梯度几何探索 | 公平复现后才进主表 |
| 上界 | Oracle-GRACE | 理想预测器，离线重放 | Pilot 数据 |

GRPO-short 是最容易被审稿人想到的对照，必须有。VIP、VIGOR 写作 strongest reproducible baselines，不写 SOTA，除非完成足以支持的公开比较。

### 8.3 Main Results（Table 2，Fig. 5）

Table 2 的行按轨道与谱系排，GRACE 放最后；数学与代码是独立训练 run，排版时可拆表。列按难度递增：MATH-500、AMC23、AIME24、AIME25、OlympiadBench、平均、代码（LCB、HE+、MBPP+ 平均），再加"实际消费 A100-hours"、"到达 GRPO 最终精度所用算力比"和"HVD 相对 GRPO"三列。每格均值加标准差，加粗最优，下划线次优，显著性符号。未运行的格填 not run，不填预期。

time-to-target：从校准阶段锁定的验证能力阈值计算，要求连续两个评测点达到阈值；checkpoint 按累计预算共同设置（2/4/8/12/16 A100-hours）；终点未达标记 >budget，不外推。

表后三句结论：难任务差距更大，与"长轨迹 LAG 更大"一致；Uniform-HT 与 GRPO-short 都不优于 Full-PG/GRPO，前者验证 Prop. 2，后者说明收益不来自"多跑短的"；Reward-CV 与 ARRoL 在困难题上掉点，因为它们停掉了五五开但有信息的轨迹；Uniform-CV 略优于 Full-PG 但远小于 GRACE，收益主要来自分配与回流。若 GRACE 优于 Full-PG 但落后于某个实用基线，分开写。

### 8.4 Ablation（Table 3，4B 数学，同账本，2 seeds）

| 编号 | 变体 | 检验什么 |
|---|---|---|
| A1 | \(m=0\)，按范数分配加 HT | 预测方向的价值 |
| A2 | 决策变量换成 \(\hat q\) | Lemma 1 与 Delta |
| A3 | 预测器输入只有 \((x,\hat q)\) | Lemma 2 |
| A4 | 去 HT 校正，停止即丢弃 | 无偏性的必要性 |
| A4' | 保留 \(m\)，不加 \(1/p\) 权重 | 偏差与方差的取舍 |
| A5 | 均匀 \(p=\beta\) 但保留 \(m\) | Prop. 3 |
| A6 | \(k\in\{4,8,32,128\}\)；随机基底 | 子空间容量 |
| A7 | \(\hat c\) 换常数 | 成本预测的价值 |
| A8 | \(t_d\in\{256,512,1024\}\)；\(\beta\in\{0.3,0.5,0.7\}\)；\(p_{\min}\in\{0.1,0.2,0.4\}\) | 敏感性 |
| A9 | 基线换 GRPO 组均值 | A5 假设 |
| A10 | 预测器不加 IPW | 抽样偏差 |
| A11 | \(b\) 取 0、0.5 | 特殊基线 |
| A12 | 只允许低更新范数前缀停 | 是否只是在跳过零梯度样本 |
| A13 | 审计 \(s\in\{1/16,1/8,1/4\}\)；基底刷新 16/32/64 步 | 标签成本与漂移 |
| A14 | 不回流算力，\(N_{\rm start}=8\) | 收益来源 |

干预实验的执行：从共同中期 checkpoint 开 3-seed 短分叉，冻结同一套 \(U, f, \hat r, \hat c\)。打乱 \(\hat r\) 只在同题、相同检测长度、相同成本桶内做，重新解 \(\lambda\) 匹配期望成本；打乱 \(f\) 在当批 \(p, Z\) 固定后做，置乱映射由独立 RNG 生成，不读 \(Z\)、后缀或奖励，也不在"仅幸存"或"仅停止"子集内分别置乱。终点看独立测试能力并报 CI。

### 8.5 Analysis

1. 机制回收（Fig. 6a）：每步 PLC 占比 GRPO 对 GRACE，同 Fig. 2c 坐标，叠加门内 \(\rho_L(t_d)\) 随训练的变化。给定同一 \(\theta\)，GRACE 不改变后缀的固有随机性，它改变的是花在低残差前缀上的算力；跨训练阶段的固有残差变化单独测量。
2. 无偏性与方差（Fig. 6b，Table 5）：固定 5 个 checkpoint，同一批前缀重复抽 \(Z\) 20 次，报偏差、方差乘成本 \(\chi\)，对 Full-PG、Uniform-HT、GRACE、A4'；按预测风险十分位画真实残差、成本与 \(p\) 的 reliability 图。
3. 干预（Table 5）：见 8.4。
4. Theorem 1 预报验证（Fig. 6c）：至少 9 个设置。
5. 敏感性（A8）。
6. 规模（Fig. 7c）：1.7B、4B、8B 的增益，附 Pilot 中 LAG 指数随规模的变化；跨规模训练收益只在完成匹配预算比较后才写。
7. 边界（Fig. 7a、b、d、e）：增益对轨迹长度、饱和度；预测器在已见题与未见题上的残差与分 epoch 增益；NRUE 对剩余成本占比的相图。
8. 非单调与漂移：学习完成阈值再次越界的比例、旧预测器的残差膨胀与基底年龄；误差大时 \(p\) 应回到 1。
9. 优化器：原始梯度、clip 后梯度与 Adam 实际位移三者的关系；小模型 SGD 对照。
10. 错误分析：统计"低 \(p\) 但真实残差高"的轨迹占比和题目特征，以及 HT 尖峰频率，喂 §10。

### 8.6 Case Study（Fig. 8）

从审计集自动选满足"答案未出现、至少有正负两类真实续写、门内、低全空间残差"的前缀作成功例：展示题目、前缀、两条真实后缀的奖励、\(p\)、完整梯度相似度与范数误差、实际省下的 backward。对照一条前缀已选定非常规路线、成功率 0.7 但残差高的轨迹被继续写。失败例选预测风险低但正交补残差大的前缀，展示一次大修正及其原因（罕见算法切换、代码边界条件）与后续 \(p\) 的回升；也保留"这一批没抽到真值、因此发现不了该错误"的情形，说明无偏不等于逐样本即时正确。文本可摘录，图中附 ID、checkpoint、seed。

### 8.7 Efficiency 与有效探索（Table 4，Fig. 5）

HVD（hard verified discoveries）。训练前在训练题的固定观察池上（1024 题，每题用共同初始模型采 16 次）确定低成功率组：初始成功 0 到 3 次的题，零成功题单独报。

\[
\mathrm{HVD}(B)=\frac{\#\{x\in\mathcal H:\ \text{预算 }B\text{ 内首次获得真实完成且验证成功的轨迹}\}}{B\ \text{A100-hours}}.
\]

同时画累计覆盖曲线避免有限池饱和误导。只统计完整、真实验证的成功；预测坐标、未完成前缀、重复解同一题不计。解法多样性只作次指标，规则预注册。

三条资源曲线：accuracy 对端到端 A100-hours；hard verified coverage 对 A100-hours；固定快照批梯度 MSE 对 A100-hours。token 节省、起步数、完成数、backward 数是解释变量，不是 headline。A100-hours 分项账本（前缀生成、后缀生成、决策 prefill、反向传播、审计、预测器训练、SVD、验证、同步与等待）另列 CPU-hours。Pareto 图加 A14（只省时间不回流）的点。vLLM 批处理不可加，报实测吞吐，FLOPs 只作参考。

---

## §9 Related Work

第一段对应 C1 和 C2：RLVR 的算力效率、题目级分配与路径剪枝。GRPO、DAPO、Dr. GRPO、RLOO 配方；动态采样与零方差题跳过（DAPO、GRESO）；题目级或逐轮 rollout 预算分配（VIP 按成功概率与梯度方差，VIGOR 按已完成组的奖励方差）；按前缀成功率或失败检测早停（ARRoL、ESPO，Speculative Rejection 在 best-of-N 上的部分奖励剪枝）；完成后按优势剪枝（DPPO）。这些工作已经说明样本不应均匀对待；本文测量的是同题、同成本、同成功率下完整更新的剩余不确定性，并让停止样本仍贡献可校正的更新。VIGOR 的标题与本文相近，主动说明其分配粒度是题目与轮次。

第二段对应 C2：梯度预测、合成梯度与控制变量。Synthetic Gradients 与 DNI（Jaderberg et al., 2017）、GPCV（arXiv:2511.05187 v2，核实）、合成梯度的信息分析（arXiv:2605.27946，核实）、G2RL 的梯度几何探索、策略梯度控制变量（Q-Prop 等）。它们对已获得的样本预测梯度以省反向传播，GPCV 的成本账本是模拟的；GRACE 在后缀和奖励都不存在时预测，同时省生成，账本是真实硬件。同名的两篇 GRACE 分别做梯度对齐数据筛选和对比策略优化，与本文任务不同。

第三段对应 C3 与 C1：无偏随机截断与 token 级 RLVR 分析。Randomized Telescoping（Beatson & Adams, 2019）、Russian roulette、Horvitz-Thompson、无偏截断 BPTT（ARTBP、UORO）；高熵少数 token 驱动 RLVR（Wang et al., 2025）、熵机制（Cui et al., 2025）。前者给出无偏框架但没有前缀信息，Prop. 2 说明这不够；后者回答"哪些 token"，我们回答"什么时候"。

对比表 Table 6 进正文，用文字限定，不机械全勾：行是 GPCV、ARRoL、DPPO、ESPO、VIP/VIGOR、GRACE；列是在线当前前缀、决策依据、停止轨迹如何处理、主张对应的成本口径。

明确排除推理时早退（DEER 等）、过程奖励模型、长度惩罚、教师蒸馏、搜索树分支，它们改变目标或推理行为，我们只改为同一目标付费的方式。

并发工作：GPCV v2（2026-09-02）与 VIGOR（2026-07-24）已公开，按相关性引用，按代码可得性与预算决定是否实证比较，并在检索日志里写理由；"四个月内可不比较"不是 ICLR 政策。投稿前两周用 `("early stop" OR prune OR truncat*) AND (RLVR OR GRPO) AND (gradient OR update)`、`"gradient prediction" rollout`、`"rollout allocation"` 再扫一遍。所有 2026 年引用逐条核实标题、作者、版本和设置。

---

## §10 Limitations、Broader Impact、Reproducibility、Conclusion

Limitations，每条带一句 future work：

1. 严格 on-policy 单次更新（A1、A7）。带 clip 的多 epoch 变体下无偏性要重推，future work 是把 HT 权重并入重要性比。
2. 低秩覆盖。8 维可能覆盖不了有用的条件均值，稀有解法切换可能落在正交补；真值纠偏保住期望但抬高方差、减少可省成本。future work 是可预测子空间的学习，不是删掉正交补审计。
3. LoRA 参数化与优化器。"更新信息"依赖 LoRA 参数化与选定的梯度目标；原始估计器无偏不等于 Adam 位移相同，不能事后换度量保主结果。
4. 策略漂移与冷启动。预测器失准、后缀高随机时需更多真值或 \(p\approx1\)，此时 GRACE 可能比完整采样慢；主文先报实际失效区间。
5. H5 边界：短推理任务、饱和题、第一个 epoch 的未见题上没有增益；Lemma 1 依赖校准基线。
6. 探索指标。HVD 衡量新解出的低成功率题，不等于解法多样性；pass@k 是至少一次成功的概率，不是能无代价挑出正确答案。
7. 范围。4B q/v LoRA、数学与代码不自动代表全参数、长时程智能体或不可验证任务。

Broader Impact 与 Ethics Statement：降低推理模型 RL 训练的算力与能耗；数据全是公开数学与代码集，没有人工标注；代码执行在禁网、无凭据、低权限、有资源限制的沙箱里；TACO 的汇集数据许可逐项核对；更高效的 RL 也可能降低滥用门槛。

Reproducibility Statement：代码、配置、梯度审计数据、Pilot 的 4096 维投影梯度、概率日志与两份账本开源；App. B 超参表、App. C 账本、Algorithm 1；随机种子与 vLLM 版本固定。

AI 使用声明：按 ICLR 当年政策如实描述 LLM 用于代码辅助、文字润色以及 4.3 解法标签的范围与人工核验。

Conclusion 三到四句：学习完成早于答案完成；GRACE 以剩余更新误差分配续写并无偏补全；同账本下更高精度和更多有效探索；训练算力应由尚未确定的更新信息定价，不必一律等待所有推理完成。

---

## §11 Rebuttal 预案

| 维度 | 审稿人找什么 | 弹药 |
|---|---|---|
| Soundness | 账本是否公平；无偏是否真成立 | 8.1 同代码基与账本、两条轨道、3 seeds、paired bootstrap；Table 5 重复抽 \(Z\) 的偏差检验；App. C 两份账本；A1 到 A7 与 token-sum、固定 \(N\) 的单元测试 |
| Novelty | 与 GPCV、RT、ARRoL、VIP、VIGOR 的差异 | Idea Card Delta；Table 6；Prop. 2；A2 与 A3 消融；同题不同前缀的干预 |
| Clarity | 一句话能否复述；名字像推理早停 | Fig. 1；Lemma 1 的 \((1-2q)^2\)；Fig. 4 标明同时省后缀生成与 backward，推理时完全独立 |
| Significance | 谁在乎 | Who cares；H3 代码任务；Theorem 1 作为通用判据；审计协议可复用 |

典型攻击与两句防御（设计一句、实验一句）：

1. "现象是平凡的，前缀越长当然解释的方差越多。" 设计：只统计答案未定且信号非零的前缀，前缀记账项 \(\mathrm{Var}(R\mid h)\|g_h\|^2\) 在答案未定时不会自动变小。实验：Table 1 第 2、4、5 行，Fig. 3b 与 3c。
2. "8 维子空间怎么可能捕获梯度。" 设计：无偏性不依赖 \(U\)（Prop. 1）。实验：Fig. 3d 的正交补能量、A6 到 k=128、A1（\(m=0\) 仍有增益）。
3. "一个更好的 reward critic 就够了。" 设计：Lemma 2 的交叉矩。实验：A3、同容量与更强 critic、离线 oracle-\(q\) 对照，终点看预算匹配的独立解题。
4. "与 VIP、VIGOR 没区别。" 设计：它们分配完整 rollout 的数量，本文在本条轨迹后缀未知时决定是否购买后缀并保留停止样本的预测更新。实验：同题不同前缀的干预（Table 5）。
5. "换了基线，不公平。" 实验：GRPO 报两种基线取更优；A9、A11。
6. "省下算力多跑几条短的也一样。" 实验：GRPO-short 不优于 GRPO；A14。
7. "开销被低估。" 实验：部署账本全部计入，开销 ≤【8%】，Fig. 5a 的 x 轴含开销，另列 CPU-hours。
8. "只在 LoRA 和 4B 上成立。" 实验：1.7B 与 8B 趋势；Theorem 1 给任何规模的判据；全参数列为 limitation。
9. "预测器只对见过的题有效。" 实验：Fig. 7d 与分 epoch 增益；RLVR 常规配方本来就多 epoch，单 epoch 场景列入 H5。
10. "无偏不代表训练更好；Adam 下无偏没意义。" 实验：Table 5 同时报方差与精度，SGD 对照，Uniform-CV 分离控制变量本身的贡献。
11. "多起步不是探索。" 实验：HVD 只统计真实完成并验证的困难题首次成功，另报 pass@8。

预留实验：换骨干（Qwen2.5-7B 或 Llama-3.1-8B，1 seed）；多决策点小规模；全参数 1.7B；训练长度 4096 一组；关键对比方差过大时优先加 seed。状态记 planned/running/complete。

---

## §12 Self-Check

分"设计已完成"与"证据待完成"两列勾，不把方案完成当研究完成。

- [ ] 全文只讲"学习完成早于答案完成"，方法每个模块都能被它解释
- [ ] Insight 一句话能被同行复述
- [ ] 多决策点、\(\hat c\)、高维 k 都在附录
- [ ] C1 到 C4 各对应一个方法小节、一个消融或分析、一个结果数字
- [ ] H0 到 H5 各有指标、实验、结果，没有孤儿实验
- [ ] \(\rho_L, \rho_A\)、PLC 在 Fig. 2 与 Fig. 6 同定义同坐标，且是全空间量
- [ ] 只看图与 caption 能还原故事，caption 都是结论式
- [ ] H5 有图，Limitations 引用它
- [ ] 摘要六句各有章节，三个数字与表逐位一致，占位符全部被真实结果替换
- [ ] 同代码基、同账本、两条轨道、3 seeds、题目层与种子层不确定性分开
- [ ] Alg. 1、App. B 超参、App. C 账本、概率日志、代码链接、Reproducibility Statement
- [ ] §5.1 的符号在 §6 全部用到；full-gradient 与 projected-gradient 从不混用
- [ ] 投稿前两周扫 arXiv，2026 引用逐条核实
- [ ] §11 的 11 条攻击都有设计与实验两句防御
- [ ] 正文 9 页；Ethics、Reproducibility、AI 使用声明齐备；匿名

---

## §13 生产顺序、预算与风险

### 13.1 顺序与 64 A100-days 预算

| 阶段 | 内容 | A100-days | 产出 |
|---|---|---|---|
| P0 | 冻结研究合同：claims.yaml（结果全为 null）、research_plan.yaml、experiments.csv、source_registry.json | 0 | 单一事实源 |
| P1 | 实现与正确性：verl 两阶段 rollout、RNG 隔离、LoRA per-sample 梯度 hook、预测器、账本；单元测试 U1 到 U7 | 8 | 可运行的 GRACE 与基线 |
| P2 | Pilot 梯度审计、Oracle-GRACE、Theorem 1 预报 | 8 | Fig. 2 与 3、Table 1；Go/No-Go |
| P3 | 画空表 Table 2、3、5 与图脚本，占位 not run | 0 | 决定要跑哪些格 |
| P4 | 校准：在校准集上锁定 \(t_d\)、lr、\(\beta\)、warmup；生成带 hash 的只读实验清单 | 含在 P1 | 锁定配置 |
| P5 | 主训练：数学 4B，机制轨道 6 个方法与实用轨道 6 到 8 个方法，3 seeds，每 run 约 1.2 A100-days | 26 | Table 2，Fig. 5 |
| P6 | 消融 A1 到 A14（2 seeds 或短分叉） | 8 | Table 3 |
| P7 | 代码任务 2 seeds、规模 1.7B 与 8B、Countdown 边界 | 8 | Fig. 7，Table 2 代码列 |
| P8 | 干预、重复抽 \(Z\)、Theorem 1 验证、最终评测、复跑 | 6 | Fig. 6，Table 5 |
| P9 | 写作：Method 与 Exp 先写，再 Intro、Abstract、Title；Rebuttal 预案；终检 | 0 | 成稿 |

预算是分配计划。先测 4B/2048 的实际吞吐，再锁定每 run 对应的最低有效训练步数；若每 run 预算不足以让 Full-PG 进入有意义学习区间，先修订资源计划，不用不学习的短跑当终局证据。

Go/No-Go（P2 结束时）：门内前缀上 LAG 指数 ≥ 0.08，残差诊断分解中不可约项占比 ≥ 50%（说明现象不是投影幻觉），Oracle-GRACE 预报算力比 ≤ 0.8，三条同时满足进入 P5；否则走 13.2。

### 13.2 风险与备用方案

| 风险 | 触发信号 | 备用方案 | 不允许的"修复" |
|---|---|---|---|
| LAG 太弱 | 门内 LAG 指数 <0.05 | 先查训练长度 2048 是否太短（换 4096）、是否只在 medium 成立。仍弱则故事转 D 型（重审）："RLVR 中更新可预测性与结果可预测性同步，早停必须以结果为变量"，保留 Theorem 1 解释负结果 | 放宽门或改标签定义 |
| 只有投影内现象 | 正交补能量占残差 >50% | 检查基底构建，比较 rank 与条件均值可预测能量；退化为只按范数分配（A1），Theorem 1 里 \(A\) 变小但 \(S\) 项仍可盈利 | 删正交补或把目标改成 8 维梯度 |
| 奖励 critic 完全解释 | Reward-CV 与 GRACE 残差持平 | 用更强同预算 critic 复核，检查同题异质性是否仍在 | 选弱 critic |
| Theorem 1 预报不盈利 | \((\sqrt{c_0A}+S)^2\ge(c_0+\mathbb Ec)V\) | 决策点前移到 256 降 \(c_0\)，生成长度加到 4096 提 \(c\)，或缩小 \(p_{\min}\)；仍不盈利则论文转纯 T2，主打 LAG 现象与可盈利判据 | 给预测器与审计成本打折 |
| 统计盈利但 GPU 不盈利 | \(\chi<1\) 但墙钟不降 | 优化批重排、坐标汇总、审计采样、隐藏态提取，优化费用计入 | 把开销移到"免费 GPU" |
| 预测器只对已见题有效 | 未见题残差与题级预测器持平 | 分 epoch 报增益；单 epoch 场景写进 H5；预测器加跨题泛化特征 | 隐藏第一个 epoch |
| HT 尖峰导致不稳 | 梯度范数尖峰频率 >5% 批 | \(p_{\min}\) 提到 0.4；残差分位数回归；梯度裁剪对所有方法一致 | 事后裁剪 \(1/p\) 仍称无偏 |
| 精度提高但 HVD 不增 | Fig. 5b 无差 | 检查是否只是更新利用效率的收益，如实分开写 | 把更多起步充作发现 |
| per-sample 梯度显存超限 | OOM | 只对 B 矩阵取 per-sample 梯度，A 用批梯度近似；或降 micro-batch | 悄悄丢标签 |
| 两阶段生成吞吐低 | 阶段 2 吞吐 <阶段 1 的 70% | prefix caching 加按 \(Z\) 排序批；或改单阶段生成加事后截断，只省 backward 和部分生成 | 不计等待时间 |
| 共同 baseline 不学习 | Full-PG 校准集无提升 | 排查验证器、采样分布、LoRA trainables、lr、格式 warmup | 降低 baseline 配置 |

任何实验数字与 Idea Card 矛盾时，先改卡再改文；所有设计变更记入 amendments.md，注明发生在测试解盲前还是后。

---

## 附录 A 理论证明提纲

A.1 Lemma 1：\(\partial\log\sigma(z)/\partial z=1-q\)，\(\partial\log(1-\sigma(z))/\partial z=-q\)；两种结局梯度 \((1-b)(1-q)\) 与 \(bq\)，差 \(1-b-q\)；\(b=q\) 时方差 \(q(1-q)(1-2q)^2\)。可选：softmax 多分支的类似上界。

A.2 Lemma 2：\(\mathbb E[\nabla\log\pi(\text{suffix}\mid h)\mid h]=0\)，代入 \(G=(R-b)(g_h+g_s)\) 并加减 \(q(h)\)。

A.3 Prop. 1：\(\mathbb E_Z[\widehat G\mid h,\tau]=m+\frac{p}{p}(G-m)=G\)，再对后缀取期望；用到 A2 与 A3。\(p\) 或 \(m\) 依赖未见后缀、分母随机、同批拟合泄漏都不在证明条件内。

A.4 Prop. 2：\(e=\widehat G-G=(Z/p-1)(G-m)\)，\(\mathbb E[e\mid G, h]=0\)，\(\mathbb E[\|e\|^2\mid G, h]=(1/p-1)\|G-m\|^2\)，交叉协方差为零。无免费午餐：\(m\) 为常数时 \(\mathbb E[r]\ge V\)，于是 \((V+(1/p-1)\mathbb E r)(c_0+p\mathbb Ec)\ge V(c_0/p+\mathbb Ec)\ge V(c_0+\mathbb Ec)\)。

A.5 残差诊断分解：\(G-Uf=(G-\mu)+(I-UU^\top)\mu+U(U^\top\mu-f)\)，第一项条件均值为零，后两项正交。有限样本估计 \(\|\mu\|^2\) 用 A/B 两半均值内积。

A.6 Prop. 3：拉格朗日 \(\min_p\ \mathbb E[r/p]+\lambda\mathbb E[pc]\) 得 \(p=\sqrt{r/(\lambda c)}\)，加 \(p\le1\) 的 KKT；\(c=0\) 且 \(r>0\) 时取 \(p=1\)。

A.7 Theorem 1：在未截断的 Neyman 族中，\(\mathbb E[pc]=S/\sqrt\lambda\)，\(\mathbb E[(1/p-1)r]=\sqrt\lambda S-\mathbb E r\)；令 \(s=\sqrt\lambda\)，\(f(s)=(c_0+S/s)(A+sS)\)。对 \(A>0,c_0>0,S>0\)，驻点 \(s^*=\sqrt{A/c_0}\)，\(f(s^*)=(\sqrt{c_0A}+S)^2\)；只有对应概率均满足实际上下界时才是受限问题的最优值。\(A\le0\) 时该未截断表达式的单调性不证明截断后的最优分配为 \(p\equiv1\)。异质风险反例：两类前缀等概率，条件梯度各为等概率的 \(\pm1\) 与 \(\pm0.1\)，\(m=0,c=1,c_0=0.1\)，则 \(A=0\)；\(p=(1,0.2)\) 的方差×成本为 \(0.525\times0.7=0.3675\)，低于全采样的 \(0.505\times1.1=0.5555\)。约束问题须用截断分配重新计算目标；同题分块另计协方差。

A.8 Adam 备注。A.9 多决策点扩展：停止时刻 \(T\) 服从离散分布，HT 权重 \(1/\Pr[T\ge t]\)，仍无偏。

## 附录 B 超参与实现细节

| 项 | 值 |
|---|---|
| 模型与 LoRA | Qwen3-{1.7B, 4B, 8B}-Base；q、v；r=16，α=32；dropout 0；\(d\)=5,898,240（4B） |
| 训练生成 | 最长 2048 token；prompt 最长 1024；T=1.0；无 top-p |
| 批 | 64 prompts；GRPO 8 starts（\(N\)=512）；GRACE 自动 \(N_{\rm start}\) |
| 优化 | AdamW；LoRA lr 1e-4，网格 {5e-5, 1e-4, 2e-4}；β=(0.9, 0.99)；weight_decay 0；grad-clip 1.0 并记录触发率；每批单次更新 |
| 基线 \(b(x)\) | EMA α=0.7；epoch 0 预扫 4 样本 |
| 决策点、预算、下界 | \(t_d\)=512（校准集选）；β=0.5；\(p_{\min}\)=0.2 |
| 预测器 | 坐标头 2560→256→8，风险头 2560→64→1；lr 1e-3；IPW；每批 2 epochs；k=8；\(U\) 每 32 步随机 SVD（oversample 16）；reservoir 512 条 FP32 |
| 审计 | \(s\)=1/8；autograd.grad 在 step 前同一 \(\theta\) 上 |
| Pilot | 审计集 240 题、16 前缀、8 决策点、16 续写（子集 96）；校准集 256 题；JL 投影 4096 维 |
| 评测 | T=0.6，top-p 0.95，最长 4096；MATH-500 avg@4；AIME avg@32；AMC avg@16；pass@k 无偏估计 |
| seeds | {17, 29, 43} |

## 附录 C 算力账本模板（部署账本；研究账本另表）

| 分项 | GRPO（A100-h） | GRACE（A100-h） | 说明 |
|---|---|---|---|
| 格式 warmup | | | 所有方法相同 |
| 前缀生成 | | | 到 \(t_d\) |
| 后缀生成 | | | GRACE 只有 \(Z=1\) |
| 决策 prefill 与预测器前向 | 0 | | |
| 反向传播 | | | GRACE 只有 \(Z=1\) |
| 审计 per-sample 梯度 | 0 | | \(s\)=1/8 |
| 随机 SVD 与重投影 | 0 | | |
| 预测器训练 | 0 | | |
| 验证器（GPU 等待） | | | 另列 CPU-hours |
| 权重同步、KV 恢复、调度空闲 | | | |
| 合计 | | | 主表 x 轴；按 GPU 数乘墙钟 |

## 附录 D 补充结果清单

Oracle-GRACE 与离线重放；\(\varepsilon,\delta,\kappa\) 阈值敏感性；三个快照的 LAG 曲线；各基准 pass@k 全表；子空间解释方差对 k；残差诊断分解分层报告；干预实验完整表；Countdown 完整结果；解法标签器与人工抽检一致性；非单调越界比例；换骨干实验。

## 附录 E 设计选择表与多决策点扩展

见 §6.4 的表；多决策点见 A.9。

## 附录 F 日志与数据合同（最小字段）

每个起步：run_id、domain、method、seed、step、problem_id、trajectory_id、snapshot_sha、basis_sha、predictor_sha、decision_t、prefix_tokens、natural_finish、p_continue、z_continue、audit_selected、prediction_coords、r_hat_full、c_hat_remaining、reward（停止者为 null）、finish_reason、answer_first_token、各 RNG counter、global_starts_denominator、gpu_reserved_seconds、cpu_verifier_seconds、true_grad_norm_sq、true_grad_coords、data_split。

每个 run：config_locked.yaml、environment.txt、trajectories.jsonl、audit.parquet、compute_ledger.jsonl、checkpoints.json、eval_per_problem.jsonl、summary.json（未运行前结果为 null，run_status 为 planned/complete/failed/censored）。

图表数据：Fig. 2 与 Fig. 6 共用 prefix_audit_summary.csv；Fig. 5a 用 learning_curves.csv；Fig. 5b 用 discovery_curves.csv。

## 附录 G 单元测试与执行顺序（给 Claude Code 或 Codex）

必须通过的测试：U1 固定 \(G, m, p\) 枚举 \(Z\) 验证精确均值；U2 枚举多条潜在轨迹验证全空间方差恒等式；U3 \(U\) 无法表示的正交分量仍被期望正确恢复；U4 向量化估计器与"加权 actor 梯度加一次 \(U\) 校正"逐位一致，\(p\equiv1\) 回归真梯度；U5 用后缀决定 \(p\) 的负例触发告警；U6 FP64 参考梯度与分布式实现相对误差达预设精度；U7 校正符号、AMP unscale、DDP 全局分母、LoRA 参数列表 hash 一致。

顺序：
1. 读 claims.yaml、research_plan.yaml、experiments.csv、source_registry.json，检查模型 revision 与数据 hash 不为 null。
2. 跑 U1 到 U4 的 NumPy 参考实现；实现 toy policy 的 autograd 梯度。
3. 在 verl 上实现两阶段 rollout 与 prefix caching；测 \(p\equiv1\) 时与单阶段逐 token 分布一致；测续写 RNG 与选择 RNG 隔离。
4. 实现 LoRA per-sample 梯度 hook，与 autograd 逐样本梯度误差小于 1e-4；实现双流估计器与账本；跑 U5 到 U7。
5. 跑 Full-PG 与 GRPO 4B 数学的可学习性 sanity check（token-sum、固定 \(N\)）；它既是基线也提供 Pilot 快照。
6. 跑 Pilot 审计脚本，产出 Fig. 2、Fig. 3、Table 1、Oracle-GRACE、Theorem 1 预报，写 Go/No-Go 报告。
7. Go 则在校准集锁定配置，生成只读实验清单，按 13.1 P5 到 P8 跑；每完成一组填 Table 2、3、5 的空格；所有修改追加 amendments.md。
8. 只从真实结果生成图表、正文、摘要；结论比证据强时输出警告。生成 LaTeX（ICLR 官方模板），按 §3 写 Intro，按 §2 写 Abstract，按 §3.5 画图（matplotlib，统一配色），按 §11 写 rebuttal 文档，检查页数与匿名。
