# GRACE 下一轮研究与实施方案：让可预测的更新收益覆盖真实成本

日期：2026-09-19。调研代码基准：本地 `e92dced`。性质：调研结论与分阶段实施方案；首批代码状态见§9.1，不是新增 GPU 实验结果。

**推荐路线：保留在线 RL，优先验证固定基底、紧凑梯度监督和小型在线预测器；先解决单卡上的预测有效性与完整成本，再扩展 rollout 数据并行。完全冻结的离线预测器作为对照。**

本文综合三路独立文献/系统调研、当前源码、原论文框架和 9 月 18 日实验记录。既有 30 项问题的状态没有因本方案而改变。本机无 GPU；没有承诺 1.5 倍加速、某个准确率或显著性。

## 1. 要回答什么

三个研究问题：

1. **监督与表示：** 如何用主训练已经产生的监督，持续预测当前策略的完整 LoRA 梯度，同时减少存储、拟合和特征费用？
2. **实际效率：** 哪些截断位置、预测精度和系统实现，可能使新增方差与完整成本的乘积降低，并转化为训练进展？
3. **证据：** 怎样区分 LAG 现象、预测器收益、风险分配收益、baseline 变化和多 GPU 系统收益？

算法目标继续保持 `G=(R-b)∇θΣlogπ`，覆盖全部 q/v LoRA A/B；`G` 为上升方向，优化器接收负号。主估计器是 `Ghat=m+Z/p·(G-m)`，`m=Uf(prefix)`，分母为事先确定的全局 starts 数 N。actor、baseline、U 和预测器在批内冻结，选择只读前缀，停止者 reward=null。保留 ChatML、repetition_penalty=1、无 min_tokens、原 parse_rate 和 ρ 平均口径。

原稿的主贡献是 LAG 现象，因此仅证明程序更快仍不够。固定 U、替换特征或改变监督协议要作为明确的方法变体记录；不能把实验目标改写成运行门槛。

## 2. 调研范围与证据边界

检索截至 2026-09-19，使用论文原文、会议出版页和 vLLM 官方版本源码。三路分别调查控制变量与非平稳预测、训练早停与相关工作、rollout 系统；另核对 baseline、统计评测和本地实现。检索从 policy-gradient control variates、gradient prediction、rollout pruning、gradient subspaces、prefix caching 等关键词，向直接引用与反证收敛。

只采用能够核对标题、作者/版本及相关内容的来源。预印本与同行评审论文分列；没有把论文内部的模拟成本、token 指标或其他任务结果转写为本项目的 GPU 收益。以下三个主题构成证据主线。

### 2.1 控制变量的可用性与失效条件

[Q-Prop](https://arxiv.org/pdf/1611.02247) 与 [LAX/RELAX](https://arxiv.org/pdf/1711.00123) 支持用辅助预测量减少梯度估计噪声，但后者强调优化估计器方差，不能仅以 reward 拟合好坏衡量控制变量。[The Mirage of Action-Dependent Baselines](https://proceedings.mlr.press/v80/tucker18a/tucker18a.pdf) 则说明，复杂 baseline 的可消除方差可能有限，部分收益还会受到归一化及实现差异影响。

对 GRACE 的含义：旧数据训练 predictor 与当前真值校正可以共存；但 Q-Prop 的 advantage 方差结论不能直接当作全 LoRA 梯度 trace 方差保证。应保留批末拟合，先检验当前前缀上的独立残差，再讨论预测器大小。

[Greensmith 等的 baseline 分析](https://www.jmlr.org/papers/v5/greensmith04a.html) 表明平均奖励 baseline 未必最小化梯度方差；[多保真 Monte Carlo](https://dspace.mit.edu/entities/publication/f104683d-44c4-40ec-bd06-47502c5c0bfd) 则把代理质量和代理成本一起优化。这共同提示：固定 b=.5、更多非零标签、较低预测损失，都需要通过最终的方差与费用比较检验。

补充检索发现 [Online Learning to Sample](https://arxiv.org/abs/1506.09016) 提出了在线学习采样分布，但 v2 元数据明确说明因错误删除了收敛定理和证明。本文只把它作为方法线索，不采用其早期收敛保证。

### 2.2 低秩不等于前缀可预测

[GaLore](https://arxiv.org/pdf/2403.03507) 研究已经算出的逐层梯度矩阵投影，并需要切换子空间；[GoLore](https://proceedings.mlr.press/v267/he25i.html) 给出噪声主导 SVD、遗漏真实梯度方向的反例。这些证据提醒我们区分梯度总能量和可预测条件均值。

它们不能证明跨题、拼接全部 LoRA A/B 的单轨迹梯度能被固定的 8 个向量预测。逐层矩阵秩、LoRA 参数化秩和本项目跨样本梯度向量的协方差秩是不同概念。GoLore 对投影优化的非收敛结论也不能直接套用到保留全空间 HT 校正的 GRACE。

### 2.3 最接近的早停与预测工作

| 机制 | 原始工作 | 可借鉴内容 | 不能据此声称什么 |
|---|---|---|---|
| 近似梯度加真实残差校正 | [GPCV，v2](https://arxiv.org/html/2511.05187v2) | 预测与精确梯度之间的方差、校准和成本折中 | 它用完整输入及已知目标的低精度 reverse-mode；模拟折价 fleet 账本不证明未知后缀可预测或真实分布式加速 |
| 生成中预测成功率并剪枝 | [ARRoL](https://arxiv.org/html/2603.24840v1) | 在线质量头、生成中删除请求、幸存者重新组批 | 停止者不贡献 GRACE 式补全，不能等同于保持原 Full-PG 目标 |
| 已完成样本的重要性剪枝 | [DPPO](https://arxiv.org/html/2603.04135v1) | 抽样与校正要成对设计 | completion pruning 发生在 rollout 后，不能当作前缀节省解码 |
| 给提前停止的前缀终止奖励 | [ESPO](https://arxiv.org/html/2605.29860v1) | 失败恢复、停止规则误判的分析 | absorbing failure 改了奖励与优化对象，不符合本项目停止者 reward=null 的定义 |
| 选择高熵 token 更新 | [Beyond the 80/20 Rule](https://arxiv.org/html/2506.01939v2) | 分析后缀中尚未解决的高熵分叉 | 仍生成完整 rollout，且归一化改变；少更新 token 不等于少生成，文中还观察到回答变长 |

GPCV 与 DPPO 提供校正机制参照，ARRoL 与 ESPO 提供真实前缀早停参照；它们各自覆盖 GRACE 的部分设计，但没有直接验证“答案未定而完整梯度已相对确定”的联合经验命题。原稿对此仍需要自己的实验。

## 3. 先分清三种统计失败，再决定改哪里

### 3.1 当前能确定的事实

9 月 18 日末态 reservoir 为 86 条，其中 61 条零 G；新鲜窗口内 22 条，fit 侧 10 条中仅 2 条非零。最终 γ=0。训练口径审计的方差×token 比为 1.1373；这尚不是物理成本收益。

旧 40 步 GRACE/Full-PG 的 starts 分别为 340/224，墙钟约 2392/1486 秒。因此它们不是同工作量性能基准。旧 GRACE 的 prefix 计时含 HF 特征和预测器，不能全部除以 rollout worker 数。当前修复后的代码尚无对应新 GPU 结果。

这些事实不足以证明只是“预测器不够大”或“GPU 不够多”。需要分开测量：

\[
\mathbb E\|G-Uf(h)\|^2
=\underbrace{\mathbb E\operatorname{tr}\operatorname{Cov}(G\mid h)}_{不可约后缀随机性}
+\underbrace{\mathbb E\|(I-P)\mu(h)\|^2}_{基底遗漏}
+\underbrace{\mathbb E\|P\mu(h)-Uf(h)\|^2}_{特征与拟合误差},
\]

其中 μ(h)=E[G|h]，P 是固定 U 列空间的正交投影；正交 U 时 P=UUᵀ。这是平方误差的正交分解，也是原稿已有的分析方向，不是新增成功标准。

| 主要误差在哪里 | 应优先尝试 | 不应先做 |
|---|---|---|
| 后缀本身随机性大 | 在开发集比较其他决策位置、训练阶段和任务条件；保留未成立结果 | 只扩大预测器或强制 γ>0 |
| 可预测均值明显，但 U 遗漏多 | 比较可预测信号基与原基，再小范围比较 k=8/16 | 用随机梯度总能量覆盖率宣称问题解决 |
| U 能覆盖，但留出预测差 | 增加主轨迹监督利用率，比较近期加权、特征与拟合方式 | 一开始训练大规模离线网络 |
| 残差变小，但真实成本不降 | 减少 HF 重放、传输、持久化和低利用率 | 把停止比例或 token 代理称作加速 |
| 原始估计更有效，但学习不更好 | 查 clip、Adam 历史、步长、更新方向和同预算质量 | 由无偏性推出优化器后的学习保证 |

### 3.2 用独立续写估计可预测信息，避免预测器自证

同一冻结 actor、baseline、前缀 h 的两条独立后缀产生 G_a、G_b，则：

\[
\mathbb E\langle G_a,G_b\rangle=\mathbb E\|\mu(h)\|^2,
\qquad
\tfrac12\mathbb E\|G_a-G_b\|^2
=\mathbb E\operatorname{tr}\operatorname{Cov}(G\mid h).
\]

对审计数据之外拟合的 U，投影梯度的交叉内积还能估计捕获的条件均值能量。不要直接用有限后缀样本均值的平方替代信号能量，它包含采样噪声。小样本交叉估计可能为负，原样保留及解释，不能截成有利结果。

开发诊断建议从一个有效训练快照、同一批问题、t∈{128,256,512,1024} 开始；例如 16–32 题、每题 2 个前缀、每前缀 4–8 条独立后缀。数量只是可调整的工作量起点，不是最低样本门槛，也不是 GPU 时长估计。先复用现有原始数据能支持的部分；缺失权重或完整梯度的部分不能补造。

检测/选择与最终 report 使用独立续写，按题统计相关性与区间。自然结束、答案已输出、梯度信号弱、分母无效均保留相应数量。总体与筛选子集并列；不改变原 ρ 的定义和平均方式。最终还应复核训练早、中、晚期，而不能只展示最好快照。

## 4. 真正需要优化的是残差与成本的组合

固定快照、共同目标和独立 Bernoulli 选择下，原始单轨迹估计满足：

\[
\operatorname{tr}\operatorname{Cov}(\widehat G)
=V_{\rm full}+\mathbb E[(1/p(h)-1)\|G-m(h)\|^2].
\]

新增项非负。GRACE 应减少相对于 m=0 的抽样代价，再通过较低成本获得更多有效样本；不能要求固定 N 的原始估计方差凭空低于全续写。该式也不覆盖 clip/Adam 的非线性更新。

一个仅用于说明的简化模型：设全续写单样本成本为 1，必付前缀成本为 a，GRACE 额外固定成本为 h，统一继续概率为 p，残差比 r=E||G-m||²/V_full。假设样本独立、成本线性且不改变目标，则效率代理为：

\[
F(p)=[1+(1/p-1)r]\,[a+(1-a)p+h].
\]

**以下是假设数字，不是 GPU 测量：** a=.25、h=.05、p=.5 时，成本比为 .675。

| r：预测残差 / 全续写方差 | F(p)：方差×成本比 |
|---|---:|
| .20 | .8100 |
| .50 | 1.0125 |
| 1.00 | 1.3500 |

这说明“不可约不确定性下降一半”也未必足以支付新增费用；基底遗漏和拟合误差还会进一步增加 r。此 r 的归一化与原稿按题平均的 ρ_L、以及 `residual_to_m0_ratio` 不相同，不能直接代入历史 ρ 计算收益。

真实 GPU 批处理不是线性成本。下一轮要在同一冻结前缀批上测不同保留率下的实际 suffix、特征、反向和同步时间；绘制质量/方差与真实费用的关系。固定工作量要求净节省，同时间预算则允许因处理更多样本而生成更多总 token，两张表不能混为一个要求。

## 5. 推荐的最小算法方案

### 5.1 先使用已有低成本候选，建立新版参照

当前已有 `minimal_gpu_cost_candidate.yaml`：主完成轨迹 audit_s=1、fresh=0、可预测交叉矩基、固定 b=.5 且 prescan=0；已有 `mechanism_fixed.yaml` 固定 N=16。应先测这些已经实现的改动，避免用旧日志估计新版瓶颈。

固定 baseline 与独立平滑 prescan baseline 保留单因素比较。b=.5 会让二元奖励的优势不再因 b=R 而变零，但不保证方差或学习改善；必须同时观察梯度二阶矩与质量。对共享 PG 目标的方法使用同一 baseline；GRPO 保留自己的定义。

若固定 baseline 明显损害质量，再研究仅使用历史监督的 prompt baseline。首版不引入新的大型 critic，也不把当前轨迹自己的 reward 塞进自己的 baseline。持续新题场景下，简单的题目历史表也不能凭空拥有新题历史。

### 5.2 固定 U，保存坐标和范数，保留在线更新

推荐增加一个与当前动态 U 并列的固定基底变体：

1. 复用本次 run 中正常 warmup 的完成轨迹建立 U，沿用实际 fit/hold/cal 隔离；费用仍计入 run。首版保留当前建基方法，避免同时更换多个统计设计。
2. U 建立后，在该段实验中固定。坐标头、风险头及 γ 在批末更新，下一批使用。
3. 每条已获得的监督保存 `UᵀG`、`||G||²`、features、原始 p/s、观察步数、题目与实际成本信息，保留合法零标签。停止者没有 G 或 reward 标签。
4. 继续使用完成者的真实全空间梯度作 HT 更新，不把 actor 更新投影到 U 中。
5. U 作为只读 artifact 保存一次；actor checkpoint 保存引用、版本身份及小型动态状态，保持可恢复性。

对任意 U，令 a_i=UᵀG_i、H=UᵀU，则恒有：

\[
\|G_i-Uf_i\|^2=\|G_i\|^2-2f_i^\top a_i+f_i^\top Hf_i.
\]

所以固定 U 时，无须存完整历史 G，也能计算完整空间的残差与现有 γ 校准目标。保留 Gram 项，覆盖非严格正交和补零列。监督存储从 O(MD) 降为 O(M(k+d_phi))，但 U 自身的 O(Dk)、真实梯度计算和特征费用仍存在。

首版建议保留普通的紧凑样本列表，复用现有拟合/划分代码。不要为了存储优化再发明通用 streaming 学习框架。固定输入变换、正交 U 和线性头下还可以维护 ridge 的充分统计，但动态归一化、滑动窗口和审计追踪会增加复杂性，因此后置。

先在已有 CPU 收集路径上做投影，验证数学与恢复行为；它减少持久化但不减少 GPU→CPU 的完整梯度拷贝。若新计时证明传输显著，再把坐标/范数归约移到 GPU，并验证精度与峰值显存。

**边界：** 换到新的基底 span 后，旧 UᵀG 无法精确恢复新坐标。不能用 `U_newᵀU_old·a_old` 冒充完整重投影。固定 U 若随训练失效，再复用现有周期刷新入口，保留有限的新鲜完整梯度建立新 U，并重建对应紧凑监督；不先添加复杂的自动 OOD/换基系统。

### 5.3 更充分利用监督，但不同时改完所有划分

先测 audit-all 与紧凑存储能否缓解稀疏问题，再决定是否改变 fit/hold/cal。不要同时取消划分、换 basis、换特征、换 baseline，再把收益全归因于补全。

若永久分题仍明显限制拟合，可增加 prequential 对照：先记录当前批实际使用的旧预测、p、U 和未收缩输出；真实 G 到达后计算时间顺序残差，批末再让这些标签参与下一批拟合。这样每个标签先用于预测核查，再用于学习，不需要新增多模型框架。

它评估的是旧预测器，不能保证更新后的风险相同；反复调参也会适应历史噪声。按纳入概率加权，保留时序和策略漂移，独立问题/续写的正式 audit 不变。此变体改变监督协议，需要单独验证，不能直接把滚动残差称为独立测试集。

### 5.4 风险分配按实际残差验证

首版保留现有风险头和 `p∝sqrt(r_hat/c_hat)`，不急于增加网络。以相同冻结 m 的 uniform allocation 作对照，检查实际额外方差项是否下降，Pearson/Spearman 相关只是辅助解释。

历史标签含旧策略、旧特征和旧导数；乘一个轨迹重要性比并不会把旧导数变成当前导数。近期加权是跟踪手段，不是纠正所有漂移的证明。label IPW 修正的是续写/审计纳入偏差，两类问题不能混同。

当前更新后的参数、γ、风险目标可能继续漂移。校准 γ 与拟合风险的先后顺序、使用哪个冻结 p 应记录清楚，不能用当前测试真值反复挑最有利 p 后称为在线结果。γ=0 或 risk 不优于统一分配时原样报告。

### 5.5 完全离线 predictor 的正确位置

以同一 U、同一初始化监督、同一输入特征比较冻结头与在线头，直接回答跟踪策略变化是否有价值。不能把“重新训练了大量离线数据”和“冻结”一起改变后比较。

离线标签生成、基底建立、拟合和校准费用都应记录。分别报告含全部准备费用与有依据的跨运行摊销费用；跨 seed 的 LoRA 初始化、参数布局和目标兼容性需要实证，不能默认一个 artifact 免费泛化。

若固定头更便宜且在早、中、晚期同预算质量都有效，再考虑采用它为主版本。它仍是使用离线辅助预测器的在线 RL。现阶段没有文献或本地实验保证这一点。

## 6. 单卡系统优化的顺序

历史运行 vLLM 为 0.18.0；下次实际环境要另行记录。当前代码已经启用 APC，prefix 和 suffix 使用同一 engine/snapshot；suffix 作为新 token prompt 请求提交，依靠完整块缓存复用。它既不是保留暂停请求，也不是必然重算全部前缀。[v0.18.0 KV manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.18.0/vllm/v1/core/kv_cache_manager.py)

| 优先级 | 工作 | 为什么与怎样验证 |
|---|---|---|
| 1 | 拆分 prefix_generate、HF features、predictor_forward、suffix、更新、同步与保存时间 | 现在 prefix 混合计时不能定位瓶颈；保留完整外层账本，阶段时间不重复相加 |
| 1 | 记录 cached tokens、实际请求 batch 数、逐条 fallback | 官方 RequestOutput 已有缓存 token 字段；现有包装未保留，不能靠 APC=true 推断命中率 |
| 1 | 缩小 TypeError 后串行 fallback 的适用范围 | 当前任意 TypeError 都可能串行重试；先复现兼容场景，避免掩盖真正错误；尚无历史触发证据 |
| 2 | GPU 上先提取需要的 hidden 位置/池化与 entropy 统计，再回传小结果 | 先保持特征定义；对比原始输出与新输出，记录临时内存和传输时间 |
| 2 | 实测 feature_batch_size=1/2/4 | 小范围吞吐/显存比较；不假设大 batch 必然更优，不把 OOM 重试费用漏掉 |
| 3 | 轻特征消融 | 采用 prefix 内已有长度、logprob 统计等，可消除 HF 特征前向；要重新训练头并比较独立残差，不能把 sampled surprisal 当 entropy |
| 4 | engine 内提取少量 hidden/entropy | 能避免重复 HF transformer，但需要固定版本的 worker/model runner 接线；不是当前标准输出的免费字段 |
| 5 | LoRA 内存传输、多 worker | 只有新计时说明同步或并行利用率值得改善时再扩展；当前没有每步复制整套 4B 基座 |

官方 [RequestOutput](https://raw.githubusercontent.com/vllm-project/vllm/v0.18.0/vllm/outputs.py) 提供 sampled logprob 和缓存相关信息，但不会自动提供任意中间层特征；[PagedAttention](https://arxiv.org/abs/2309.06180) 说明 KV 管理与批处理的重要性，不能据此保证小 batch 的具体加速。

三处容易产生新的算法错误：

- `−log p(sampled_token)` 不等于该位置完整词表 entropy；其条件期望关系也不能保证逐前缀统计等价。
- 生成第 t 个 token 所用的 logits 来自此前上下文；读取包含第 t 个 token 的前缀 hidden/下一 token entropy 时，要核对位置，不能偷看后缀。
- v0.18.0 中请求 prompt_logprobs 会绕过 prefix cache，不能为了补特征而悄悄抵消缓存收益。原地更新同名 LoRA 时，必须失效旧参数对应 KV。

**Full-PG 也要优化。** 当前 Full-PG 经统一两阶段路径，可另测 `decision_tokens=max_new_tokens` 的单次完整生成配置。它是系统对照，不能由不同采样批次产生的随机文本差异否定或宣称逐 token 完全一致。`p=1` 的 GRACE 是代数回归与开销对照，不能替代高效 Full-PG。

## 7. 多 GPU 方案后置，但设计现在可以明确

vLLM [v0.18.0 数据并行文档](https://docs.vllm.ai/en/v0.18.0/serving/data_parallel_deployment/) 支持多副本处理请求。未来可以比较单 actor GPU 加多个 rollout engine；当前本项目 `n_gpu>1` 仍拒绝，不应仅修改配置绕过。

设计要求：

1. 每步所有 rollout worker 使用同一个已发布 actor snapshot，prefix 与 suffix 不跨更新。
2. 保持逐 start 的身份和 RNG 流，全局合并前缀信息后按同一预算定义分配 p，最终除以全局 N。
3. prefix/suffix 尽量保持 worker 归属；重新均分 suffix 会损失本地 KV 命中，不能只追求请求数量平均。
4. 记录 actor 等待、worker 尾部等待、同步、KV 失效、模型驻留和所有预留 GPU-seconds。
5. 与相同硬件资源下同样优化的 Full-PG 比较；跨卡数结果单独说明系统扩展性。

全局 N=16 时，3 个 rollout worker 每个只有约 5–6 个 starts，筛选后更少。更多 GPU 可能缩短墙钟，也可能降低单位 GPU 吞吐。应先测一个 engine 的保留率—耗时曲线与更大固定 batch，所有方法同步采用相同 batch 配方。

附件的 `1 actor+3 rollout` 与 `T_rollout/4` 估算不一致。即便改成除以 3，线性模型也不包含小 batch、特征、通信和长尾影响；本文不采用 11–14 分钟作为预测结果。

## 8. 实验安排：先归因，再扩大

以下是工作顺序，不是审批、最低样本或自动 Go/No-Go 条件。结果不利时照常保存，用来决定下一项工作更值得投入哪里。

### 8.1 第一组：新版低成本参照与数值核查

- 使用已经实现的 fixed-N/cost candidate，从共同 SFT actor 开始，对照同 baseline 的 Full-PG。
- 同 checkpoint、同请求 seed 做独立进程重复评测，使用现有 repeatability 工具定位初始评测差异；同时记录 HF/vLLM logprob 差异。未解释的差异限制归因，不伪装成数值一致。
- 单独测冷/热 cache、同步后 cache 失效、不同保留率；保存真实 GPU 分项及完整费用。
- 固定 b=.5 与独立平滑 baseline 的对照共享题序；若在同轨迹重算 baseline，需要原始 score 梯度。原来乘优势后为零的 G 无法反推 score，缺失时补算并记账。

### 8.2 第二组：单快照的三项误差分解

使用 §3 的小规模独立续写诊断，比较 m=0、Prompt-CV、Reward-CV、现有前缀低秩预测。必要时用独立训练部分构建“续写均值基底”作为可预测信号诊断，不把它冒充免费线上可用的 U。

先做有限的 t 网格，辨认主要误差；基底遗漏明显再比较 k，拟合误差明显再比较特征/近期监督。避免 t×k×baseline×特征×网络宽度的完整笛卡尔搜索。开发诊断开销单列，不能隐藏在最终方案之外。

### 8.3 第三组：固定基底与预测头是否需要在线适应

| 版本 | U | predictor | 用途 |
|---|---|---|---|
| 当前修复候选 | 当前周期更新 | 在线 | 新参照，实际效果未知 |
| 推荐候选 | 固定 | 批末在线更新、紧凑监督 | 检验能否减少存储/拟合费用并保留跟踪能力 |
| 离线对照 | 同一个固定 U | 从同一监督状态冻结 | 隔离在线适应的价值 |

先在一个开发 seed 下比较，不用它建立统计显著性结论。若共享同一全续写 warmup checkpoint，必须包含 actor、optimizer、必要 RNG/状态；准备费用对每种独立使用场景计入，实际共享运行只记一次。不能只加载相同 actor 后把后续差异都归为预测器。

### 8.4 第四组：补全与分配的因果对照

复用现有四臂脚本：GRACE、m=0、uniform、p=1。冻结审计中保持其他量一致：

- GRACE vs **相同 p 的 m=0**：补全是否降低真实残差代价。
- adaptive vs **相同 m、同成本设计的 uniform**：风险分配是否有贡献。
- p=1：回到完整梯度，以及测量辅助开销。
- 独立训练的各臂：查看上述干预长期如何改变 actor；独立轨迹已分岔，不再宣称保持同一 p/同一 suffix。

效率主表另加高效 Full-PG。机制四臂保留相同辅助工作便于归因；性能基线应去除自身不需要的工作，不能让 Full-PG 陪付 predictor 的费用。

### 8.5 第五组：正式同成本与 LAG 证据

固定开发阶段选定的配方后，使用相同设备、明确墙钟预算和多个独立训练 seeds。3–5 seeds 可作为初始安排，区间宽度取决于实际变异；不能承诺这么多就一定显著。按 seed 配对比较，不把同一模型的回答样本当作独立训练重复。[统计评测依据](https://arxiv.org/abs/2108.13264)

主方法表保留高效 Full-PG、Uniform-CV、GRPO 和 GRACE；GRPO 使用自己的组优势定义，并给予各基线合理且可比的开发预算。另补直接竞争的前缀剪枝方法，优先考虑有公开实现的 ARRoL：使用相同模型、数据与硬件重新测量，明确它的目标/选择机制差异，不照搬论文中的倍数。若尚未完成复现，就将该比较列为缺失，不能只凭击败弱基线宣称优于已有早停方法。

最终输出四类表：

| 表 | 必需观测 | 结论范围 |
|---|---|---|
| 同算力质量 | 预算内已发布 checkpoint 的 avg@4/pass@4、steps、starts、seed 差及区间 | 同资源下是否学得更好；预算外终点与超额费用另列 |
| 完整成本 | 主/辅助生成、特征、拟合、反向、同步、保存、初始化、失败及 GPU-seconds | 同工作量是否更便宜，同质量是否更快 |
| 估计器 | 独立报告上的全空间方差、残差、同 p 的 m0、同成本 uniform、token 代理与时间测量 | 预测与分配各自是否有收益，代理不能替代真实训练成本 |
| LAG | 同前缀上的两项 ρ、联合事件、信号能量、自然结束/已答/无效分母及全体数量 | 答案未定且更新相对确定是否同时成立，及成立的范围 |

开发选择使用独立开发题目；不反复用最终 MATH/AIME 等评测挑配方。已经查看过的结果如用于调参须披露，不能事后改称未触碰的 holdout。正式结论需要未用于挑选配方的证据，并复核不同训练阶段和至少一个额外任务/模型条件。

“γ 在某区间”“非零补全占多数”“rank 接近 8”“realized-G 覆盖 35%”都不是尺度不变的必要条件，不能替代上述表格。即使 token 方差代理改善，也仍需实际优化器与质量结果。

## 9. 代码落点与验证

本节列分阶段实施任务；已有能力不会重复搭框架。首批实施状态见下节，每项都可以单独对照。

| 顺序 | 修改范围 | 验证 | 对应问题 |
|---|---|---|---|
| 1 | algorithm/gpu_engine/vllm_two_phase：细分计时、cached tokens、fallback 原因 | 包装层 CPU 输入输出测试；GPU 冷/热请求与批量验证，完整费用不重复计数 | P16/P19/P28/P29 |
| 2 | hf_actor：保持定义的小结果归约与回传 | 旧新特征数值对照、EOS/padding/决策位置测试；GPU 峰值与时间 | P05/P16 |
| 3 | reservoir/update/actor_update：固定 U 紧凑标签 | 直接全 G 与坐标+范数残差/γ一致，零 G、p/s 权重、Gram 项、固定 N、符号和 p=1 回归 | P01–P08/P13/P15 |
| 4 | state_io/basis：只读 U 引用与恢复 | 相同 artifact 恢复；布局/维度不兼容报错；缺失文件不伪造状态；旧格式读取与旧 run 保留 | P15/P27 |
| 5 | audit：复用已有独立续写，补三项分解与交叉能量 | 合成已知条件均值/噪声的 CPU 检验；检测/report 不串用，不改旧 headline/ρ | P07/P08/P18/P23 |
| 6 | 可选 prequential/轻特征变体 | 标签只影响下一批；独立 report 不变；schema 改动不复用不兼容 predictor 权重 | P02–P06/P18 |
| 7 | 多 GPU engine/同步/汇总 | 逐 start 唯一性、全局 p/N、同快照、worker 归属、真实 GPU 账本；所有基线同资源 | P16/P19/P29 |

既有候选配置、四臂脚本、成本—质量汇总和重复评测工具继续复用。调研阶段只读；用户随后授权实现。完整梯度历史可在新固定 U 变体中取消，但不能删除旧实验原件或让动态 U 恢复失真。

### 9.1 首批实施状态（2026-09-19）

顺序 1–4 已落到代码，实际 GPU 收益待测；顺序 5–7 尚未在本批实现。

| 改动 | 实际行为与边界 |
|---|---|
| 特征归约 | 保持 legacy/response/decision 定义、EOS/pad 与 NumPy FP64 特征布局；hidden 位置选择、last64 池化、熵统计在 actor 设备完成，回传最终小特征。完整 HF prefix forward 和模型 hidden-state 输出仍存在。 |
| 生成与同步观测 | `steps.jsonl` 保存 prefix generation、HF feature、predictor forward、suffix generation 的嵌套计时、cached tokens、成功/失败提交次数与回退原因，以及 adapter 保存/加载/清缓存计时。仅明确的 SamplingParams 单实例/list 类型错误允许串行兼容退回；其他 TypeError 原样抛出。子时间不重复计入账本。 |
| 固定 U 紧凑监督 | 新增单一 overlay `minimal_gpu_fixed_basis.yaml`，仅开启 `predictor.fixed_basis`。warmup 期间沿现有配置学基，warmup 结束有真实 U 则冻结；尚无真实 U 则继续现有建基尝试，不把初始化占位基当成果。冻结后历史及新增审计标签存坐标与完整梯度范数，继续原按题划分、IPW/年龄权重、在线拟合和 γ 校准。 |
| 完整梯度与数值 | 当前完成者的主反向、完整 G、HT 修正、固定 N 和停止者 null 保持原定义；只释放历史 reservoir 的 G 引用。紧凑残差保留一般 Gram 项，非正交/秩亏 U 可计算。FP64 展开在近乎完美预测且大数相消时有有限精度损失，仅夹掉舍入量级负数；不能把它解释为任意尺度逐位无损。 |
| 持久化 | 固定 U 按内容 hash 单独原子保存为 `basis-<sha>.npy`，NPZ 使用相对引用。加载验证 hash/维度，旧内嵌 NPZ 可读；发布 latest 时先复制所需 U。相同目录复用 U 文件；保存/复制时校验仍有读取和 hash 成本，不能声称存储开销为零。 |

生成/同步异常会写 `failed_execution.json`，保留失败调用边界、执行尝试、缓存和同步观测，原异常继续抛出；观测写盘失败也不覆盖原异常。已恢复成功的兼容回退不误归因为后续 actor/feature 的失败。固定基底 hash 改用连续数组 buffer，保持原字节协议，避免 `tobytes()` 再创建一整份 U 的临时副本；仍需读取并校验数组。

紧凑历史不能换基或恢复为完整 G。续训必须继续使用固定基底配置；actor-only 的新实验初始化与恢复完整学习状态是两种不同操作，不混称。固定 U 可能随策略漂移变差，仍应观察新题残差和独立审计，不能预设它胜过动态 U。当前实现没有完全离线预测器、多卡训练、额外监督或强制非零 γ。

服务器单因素对照：在同一套 cost candidate 上分别不加/加入固定 U overlay，使用同 seed 共享初始化；先检查固定步数下的机制与全成本，再用现有 wall 模式/多 seed 汇总评估同物理预算质量。四臂机制脚本可用 `COMMON_CONFIG=configs/experiments/minimal_gpu_fixed_basis.yaml` 比较其既有 recipe；该脚本自行设置各臂 ABLATION_CONFIG，不能用环境 ABLATION_CONFIG 叠加。若要组合 cost candidate 与固定 U 的四臂，只在一份普通 YAML 中合并这两个明确配置，再交给 COMMON_CONFIG；不新增包装框架。

完整迁移使用 `archive_run.py --include-checkpoints`，或复制包含 NPZ 和 U sidecar 的整个目录。默认精简归档仍可能排除大 U，排除清单可见，不能把这样的包当可恢复检查点。验证详情及最新结果见 [TEST_STATUS.md](TEST_STATUS.md)。本批没有新 GPU 训练、质量、方差收益或加速数据。

## 10. 结论、未决问题与调研阶段验证

**RQ1：** 固定 U 下，坐标、范数与 Gram 项足以保留残差及校准计算；主梯度监督与小型在线头是目前改动较小且能保留策略适应能力的候选。是否需要换基、是否能完全冻结头，仍需跨训练阶段的独立证据。

**RQ2：** 优先消除额外生成、持久化和重复 HF 特征处理，并在单卡测量真实保留率—成本关系。APC 已有；多 GPU 并行不创造梯度可预测性，不能弥补统计上没有可利用的信息。

**RQ3：** 用 oracle/基底/拟合三项分解回答为何失败，再用冻结干预与独立训练四臂区分补全和分配，最后以多 seed 同物理成本质量和同前缀 LAG 闭合论证。文献支持这条验证路线，尚不支持承诺最终必然成功。

最强反对仍是：答案未定的前缀可能保留大量不可约完整梯度噪声，或者可预测均值分散在远高于 rank-8 的空间。若诊断支持这个反对，应诚实收窄成立条件或修改研究主张；不能通过删困难样本、改 reward、改 ρ 平均、强迫 γ 或排除费用来制造达标。

调研阶段只读检查了源码与历史原始字段；用 CPU 随机矩阵核对一般 Gram 残差恒等式和加权目标，最大绝对误差分别约 1.14e-13 与 1.82e-12；简化成本表按公式计算。它们是代数核验，不是预测器泛化、CUDA 正确性或训练提速实验。三个调研子任务当时均只读；后续授权实现及验证另见§9.1，没有新增 GPU 数字。

## 参考来源

- Gu, Lillicrap, Ghahramani, Turner, Levine. **Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic.** ICLR 2017. [原文](https://arxiv.org/pdf/1611.02247)
- Grathwohl, Choi, Wu, Roeder, Duvenaud. **Backpropagation through the Void: Optimizing control variates for black-box gradient estimation.** ICLR 2018. [原文](https://arxiv.org/pdf/1711.00123)
- Tucker, Bhupatiraju, Gu, Turner, Ghahramani, Levine. **The Mirage of Action-Dependent Baselines in Reinforcement Learning.** ICML 2018. [原文](https://proceedings.mlr.press/v80/tucker18a/tucker18a.pdf)
- Greensmith, Bartlett, Baxter. **Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning.** JMLR 2004. [出版页](https://www.jmlr.org/papers/v5/greensmith04a.html)
- Peherstorfer, Willcox, Gunzburger. **Optimal Model Management for Multifidelity Monte Carlo Estimation.** SISC 2016. [MIT 原件与元数据](https://dspace.mit.edu/entities/publication/f104683d-44c4-40ec-bd06-47502c5c0bfd)
- Bouchard, Trouillon, Perez, Gaidon. **Online Learning to Sample.** 2015，v2 2016；收敛证明已撤回。 [版本记录](https://arxiv.org/abs/1506.09016)
- Zhao 等. **GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.** ICML 2024. [原文](https://arxiv.org/pdf/2403.03507)
- He, Li, Hu, Chen, Yuan. **Subspace Optimization for Large Language Models with Convergence Guarantees.** ICML 2025. [出版页](https://proceedings.mlr.press/v267/he25i.html)
- Ciosek, Felicioni, Elenter, Imani. **Gradient Prediction with Control Variates in the Cheap-Forward Regime.** 2025，v2 2026-09-02，预印本。 [原文](https://arxiv.org/html/2511.05187v2)
- Xu 等. **Prune as You Generate: Online Rollout Pruning for Faster and Better RLVR.** 2026，预印本。 [原文](https://arxiv.org/html/2603.24840v1)
- Zhu 等. **Unbiased Dynamic Pruning for Efficient Group-Based Policy Optimization.** 2026，预印本。 [原文](https://arxiv.org/html/2603.04135v1)
- Li 等. **ESPO: Early-Stopping Proximal Policy Optimization.** 2026，预印本。 [原文](https://arxiv.org/html/2605.29860v1)
- Wang 等. **Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning.** NeurIPS 2025. [原文](https://arxiv.org/html/2506.01939v2)
- Kwon 等. **Efficient Memory Management for Large Language Model Serving with PagedAttention.** SOSP 2023. [原文](https://arxiv.org/abs/2309.06180)
- Agarwal, Schwarzer, Castro, Courville, Bellemare. **Deep Reinforcement Learning at the Edge of the Statistical Precipice.** NeurIPS 2021. [原文与版本](https://arxiv.org/abs/2108.13264)
- vLLM 官方 v0.18.0：[KV 管理源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.18.0/vllm/v1/core/kv_cache_manager.py)、[输出结构](https://raw.githubusercontent.com/vllm-project/vllm/v0.18.0/vllm/outputs.py)、[数据并行文档](https://docs.vllm.ai/en/v0.18.0/serving/data_parallel_deployment/)。

本地依据：仓库根目录 `GRACE_ICLR论文框架_v3.md`，`docs/GRACE_ISSUE_CHECKLIST_20260919.md`，`docs/GRACE_EFFICIENCY_REMEDIATION_20260919.md`，`docs/MINIMAL_RESULTS_REVIEW_20260919.md`，以及 `_minimal_review_20260918_063159/minimal-chain-20260918-063159` 原始记录。9 月 16/17 日链未作为本轮现役效果依据。
