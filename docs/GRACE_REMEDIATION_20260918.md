# GRACE 9月18日修复与验证记录

对应 [50项审查发现及13份原报告](GRACE_REVIEW_CONSOLIDATED_20260918.md)。修复基于 `a1ca77b` 的工作树；以下是代码、实验脚本和 CPU 验证结果，**不是新 GPU 实验结果**。9月17日归档、PR #4 和9月16日历史数据均未改写。

本轮已处理审查发现中的确定代码缺陷，并把共同起点、四方法、多 seed、可追溯评测、独立诊断和完整成本记录接进运行链。涉及预测器泛化、停止预算和优化器效果的发现只能提出明确变体并实测，不能因代码测试通过就宣布 idea 成功。

**第二轮继续修复已并入本文件**：新增当前策略独立采样、监督年龄窗口/衰减、按题交叉拟合建基、真实墙钟反馈与共同预算、固定批全空间/优化器审计，以及全体已观测轨迹的行为比较。下面50项表已更新；这些是可执行实现，收益仍未经GPU验证。

**附件评审择优吸收**：新增GRPO无用prescan消除、固定/平滑baseline对照、同预算uniform收缩、独立关闭补全及p=1对照，并补全误差与clip/Adam方向分解。具体取舍与使用方法见第6节。

## 1. 实际改动

### 数学、预测器和数值路径

- 保留全空间 HT/CV、R−b、当批固定 N、G 为上升方向、停止者 `reward=null`、批内冻结 actor/baseline/U/heads，以及全部 q/v LoRA A/B 的目标范围。
- weighted ridge 改用适合小样本的 dual 求解，含不受罚截距；sum/mean 权重归一化显式区分。小型测试验证相同目标下与 primal 解一致，不能把求解更快当作泛化改善。
- basis 支持精确 Gram/SVD、IPW、三种中心化的旧 PCA 对照；第二轮默认覆盖为 `predictable_crossfit`：整题留一 ridge 预测条件均值，再求预测均值的加权低秩方向，不做题内去均值。仅用fit侧标签，单题/零信号返回无新基，实际秩不足k按真实秩补零。
- 固定历史特征尺度，避免每批改变输入坐标而保留旧网络和 Adam 状态。风险输出用 `softplus + eps`，移除硬 clamp 造成的额外梯度死区，并记录 logit 与接近下限的情况；极端负 logit 的浮点下溢仍需观察。
- 校准题从坐标拟合和建基中隔离。γ 使用当前设计下的校准权重，Uniform-CV 使用自己的固定 p；Reward-CV 使用对应风险模式。无可用校准观测时保留历史值并报告缺失，不增加最低样本量门槛。
- 在拟合新标签之前诊断整个冻结预测器，单列历史未见题。年龄窗口/衰减同时作用于建基、f/r、成本和γ；新配置窗口12步、半衰期6步，缺时间记录明确排除。每4步为每个当前题额外采1条独立完整轨迹，使用当前冻结actor、独立RNG、R−b及完整G，计入全部成本；仅用于本批之后的预测器，不影响当批actor更新或停止者null。不会把旧轨迹重算梯度冒充新策略样本。
- HF 使用可配置 BF16 autocast，保留 FP32 LoRA 参数与优化器状态；SFT 交叉熵使用 FP32 logits。vLLM LoRA dtype 显式记录。第二轮默认在更新前检查所有完成响应和停止者已观测前缀的全部token；缺失原行为分数保持不可用，不用新生成冒充原行为。汇总按token加权，保留每条失败和覆盖情况。
- 非有限真实梯度在 optimizer 更新前报错；合法零优势跳过无贡献求导，但不跳过必要的 CV、合法零标签和 Adam 动量更新。

### 运行、恢复和开销

- 每份 checkpoint 只压缩一次，原子发布 step 文件，再复制已压缩字节到 latest；新配置每5步保存，同时保留 step0、20、40及会话终点。未保存步明确记录最后已保存位置，不伪造逐步恢复能力。
- 全局 Python/NumPy/torch/CUDA RNG 随快照保存恢复；从旧快照回退会生成独立运行目录，保留原目录及未来日志。加载一次 payload 供恢复链共用。
- 输出请求配置、恢复后的有效配置及差异。允许有意变更，但记录数据和模型身份，不能把改变配置的续训称为精确重放。
- prescan、eval 支持保持逐请求 seed 的批处理；HF entropy 分块计算；短 probe 不再前向整条响应。特征批量可配置，新配置为控制峰值显存仍使用1。
- 去掉未消费的 HF KV cache；按方法移除不用的预测头和训练步骤，保留初始化 RNG 顺序。只在刷新时堆叠必要的完整梯度，残差按块计算。
- 增加 state/压缩/复制/hash/persistence 计时，区分 session 与累计成本；失败也计时。CPU 秒表示当前进程，GPU reserved wall 表示预留占用，均不能冒充所有 worker CPU 或 kernel 时间。资源快照记录共卡进程、功率/频率及可获得的显存事实。
- 新配置由完整batch实测秒数/N的EMA调整下一批N，包含采样、HF、fresh监督、predictor、sync和保存；不改变当批N。没有显式目标时采用首个完整batch实测时长；最大N=64为该单卡配置的显式边界，不是普适最优值。旧token建议另记 `token_proxy_next_n`，实际计划取保存后的 `cost_control.jsonl`。
- `run_wall_seconds` 为跨续训累计预算，`session_wall_seconds` 为本次预算；在batch边界停止并保存真实终点，报告超额和步数上限提前结束。时钟包含运行入口后的setup/SFT/保存，排除进程启动/退出和最终元数据自身写入。快照内是保存前状态，保存后的控制状态从同一步成本日志恢复，不再次压缩。warmup内结束也正常保存，不设最低训练步数门槛。

### 实验与测量

- 每个训练 seed 先生成一份共享 SFT actor。四方法加载相同初始权重，各自建立优化器、baseline 和 predictor；保存 actor hash 以供核对。`--init-checkpoint` 是 actor 初始化，`--resume` 是完整续训，两者不能混用。
- 默认顺序运行 Full-PG、GRACE、Uniform-CV、GRPO，seed 为17/23/41。评测固定可复现地选128题，每题4次；审计16题。样本增加不构成显著性保证。
- 保存实际 stage 路径、checkpoint 来源、训练 seed、评测题目/金标/顺序 hash、生成协议和 actor hash。来源不匹配或缺失时保留每方法原始事实，配对统计标明不可用；归档移动后仍能定位相对路径。
- 两个审计并列：原多 t、4096 horizon 审计；额外 t=512、2048 horizon、4样本 baseline 审计。后者匹配部分训练条件，仍是冻结 actor 的 pooled 诊断，不是固定多题训练 batch 的精确方差。
- 新增第三个 `batch-audit`：固定题集与N，冻结actor、U/heads、baseline与optimizer，重复独立完整采样；用全空间在线矩统计批梯度方差，复制checkpoint优化器状态在CPU上重放clip/Adam更新。所有后缀都实际生成，停止仅为诊断模拟，全部成本计入。GRPO目标不同，明确记为不适用，不混作R−b机制审计。
- 保持原 headline ρ 和 parse 平均；另加 detect-only 筛选、report-only 统计。单题上置信界不再冒充点估计。CountSketch 修正绝对尺度；`PU` 用正确 Gram 伪逆投影，并另记全空间 norm/coords。
- 保存原始判分字段、完整续写、baseline、请求 seed、token 分数、行为时刻和更新后身份。大小写题面、错误人名包含匹配、将 `5x` 当作 `5` 的判分问题均有反例回归；Windows math-verify 使用有超时的隔离进程。判分协议版本变更需要重判两侧，不能与旧分数直接混算。
- 运行时源码内容与对应 hash 一并保存。归档工具记录包含与排除项，包含文件的 hash 对应实际写入 tar 的字节；`--include-checkpoints` 保留大 NPZ。旧包已经缺失的权重不能由新工具恢复。

## 2. 50项逐条处理

“已修复”针对表中明确缺陷；“新变体”表示实现已可运行而收益未知；“已补测量”表示能观察并归因，未承诺该因素已经消失。所有 GPU 成本和质量仍需服务器验证。

| ID | 本轮处理 | 剩余边界 |
|---|---|---|
| 01 | 新变体：warmup末刷新，记录 basis rank、γ与同步状态 | 标签不足或低秩仍可能出现；不能强造高秩信号。 |
| 02 | 新增整题交叉拟合的条件均值建基，并作为第二轮默认变体 | 优化历史标签上的预测均值能量代理，非未来策略收益保证；旧PCA保留对照。 |
| 03 | 新变体：IPW 加权建基与中心化 | 不保证选出的 U 比原 U 更好。 |
| 04 | dual ridge；显式归一化、独立残差诊断 | 高维、小监督、分布漂移不会由求解器自动消除。 |
| 05 | 当前设计权重与独立校准题接线 | γ 仍为有限历史样本估计；后续风险训练可能受历史 γ 间接影响。 |
| 06 | 固定特征尺度，保存/恢复尺度状态 | 固定尺度对长期分布漂移的适应性需观察。 |
| 07 | 移除额外硬 clamp 死区，增加数值诊断 | 极端 softplus 浮点下溢仍可能发生。 |
| 08 | 年龄窗口/衰减实际进入各训练目标；增加当前策略独立新采监督 | 限制旧标签使用，不能删除神经头已有参数记忆，也不保证新标签足够。 |
| 09 | 校准排除建基；新题上冻结全流程再诊断 | 校准集不是最终测试集，不能把调参后的校准表现当泛化。 |
| 10 | 同成本 uniform/m=0/事后 r,c 诊断；新增实际训练uniform收缩、独立关闭补全和p=1对照 | 收缩保持预计成本预算，非真实GPU成本保证；m=0对照保留原风险模型。 |
| 11 | 显式 legacy/response/decision 特征语义 | 默认 legacy；特征变更需作为单独方法对照。 |
| 12 | 每步题数4，日志独立报告题数与 N | next_n 仍主要增加同题起步，题间噪声不会按 N 比例下降。 |
| 13 | prescan完整记录与批处理；新增16样本、固定baseline、先验平滑对照；GRPO移除无用预扫 | 默认EMA无平滑；平滑只使用独立prescan，不让当条reward进入自身baseline。 |
| 14 | 同checkpoint优化器状态下配对批更新分布，新增raw/clip/Adam三阶段方向与误差 | CPU重放不是CUDA逐位一致证明，一步更新改善也不保证长程质量。 |
| 15 | BF16 运算、FP32主参数、逐 token 行为比较 | HF/vLLM kernel 与有效策略的一致性待 GPU；配置相同不等于数值相同。 |
| 16 | 实测完整batch成本反馈已替代新配置的Z-proxy调度；新增共同墙钟预算链 | EMA不是精确性能模型；batch边界存在超额，保留真实耗时，不宣称严格相等。 |
| 17 | 修订论文 §5.4/A.7：条件、上下界方向、A≤0反例 | 没有把理论阈值转成运行门槛。 |
| 18 | 小样本 weighted dual ridge，保留不罚截距 | CPU 等价已验，GPU/整链提速未知。 |
| 19 | 小 n 精确 Gram/SVD，randomized 可显式选 | 数值秩与病态性仍要看实际谱。 |
| 20 | 避免每次更新重复完整 G stack，分块残差 | reservoir 和 U 本身仍有不可忽略内存；不是零开销预测器。 |
| 21 | 一次压缩、间隔保存、发布与计时分开 | GPU 实际压缩/I/O耗时待测。 |
| 22 | prescan/eval 批处理、entropy 分块、短 probe 截长 | HF feature 默认batch1；GPU批大小及vLLM调度收益待测。 |
| 23 | 零系数跳过无贡献前后向与审计求导 | 保留零标签、补偿流、Adam 动量语义。 |
| 24 | 不保留/训练当前方法未使用的头 | 为保持 RNG 顺序仍消耗相同初始化随机数。 |
| 25 | HF forward/SFT/logprob 显式不用 KV cache | 峰值显存净减少量待 GPU。 |
| 26 | 每seed共享一次 SFT actor，核对 hash | 旧包缺失 step0 权重的原因不能追补证明。 |
| 27 | 四方法脚本和 CPU 链已接通 | Uniform-CV/GRPO 新 GPU 完成结果仍为空。 |
| 28 | 3seed、随机固定128评测题、16审计题 | 覆盖面仍有限；不许用回答数量替代独立题/seed数量。 |
| 29 | 额外固定题集、固定N、独立重复全空间批审计 | 是配置指定的冻结批分布，不是还原训练最后一个动态batch，也不外推整个题库。 |
| 30 | 保留原曲线，追加 detect/report 分离报告 | 旧 headline 的选择依赖仍须注明，不能悄悄替换原口径。 |
| 31 | 单题 UCB 不可用、点估计另列，CI对象注明 | 更多问题/seed之前，不承诺小样本区间覆盖。 |
| 32 | 修正 CountSketch 多除 √m 的尺度错误 | 比值共同尺度抵消；不能据此改写旧2.3218结论。 |
| 33 | 正确投影到 span(PU)，另存全空间摘要 | JL 空间正交补仍不等于原空间正交补。 |
| 34 | 记录 rollout 前冻结身份与更新后身份 | 旧日志错时身份不由新代码反向补全。 |
| 35 | 新默认检查所有完成响应与停止者前缀；缺原行为分数不替代 | 只核对已观测token，不证明所有可能序列的策略分布相同；开销全部计入。 |
| 36 | session去嵌套、失败/续训累计、CPU与保存计时 | worker CPU、kernel级耗时及外部账单未完整覆盖。 |
| 37 | 实际生成 token、完成/parse、SFT使用量、方法来源分清 | 保留旧兼容字段，解释时使用明确新字段。 |
| 38 | 审计输出人群、样本单位、成本范围；反事实标hindsight | PLC/oracle/selected/token成本均不能替代训练加速。 |
| 39 | 源码、raw字段、manifest、资源记录与完整归档选项 | 历史缺件、跨语料近重复和外部进程精细影响尚不能完全追溯。 |
| 40 | 在线与resume共用起步分组定义 | 连续与恢复 CPU 回归通过。 |
| 41 | 历史 baseline 用全 starts 固定分母 HT 均值再EMA | baseline可超出[0,1]且噪声不消失；保持R−b与历史独立性。 |
| 42 | 非空根目录重跑另开目录；检查experiment/seed/train身份和真实终点评测 | 终点评测缺失不再用旧step40顶替；来源不明不输出配对收益。 |
| 43 | 回退恢复分新目录，旧未来日志留在原目录 | 不删除原始实验记录。 |
| 44 | 临时快照原子替换，保存成功后发布关联日志 | 不是跨全部文件的数据库事务；异常恢复路径已回归。 |
| 45 | 失败也记录本session envelope | 进程被kill/断电前尚未落盘部分仍可能缺失。 |
| 46 | 汇总累计session成本，恢复空闲不计作训练 | 同时保留各session事实。 |
| 47 | 全局RNG/有效配置/恢复差异/数据身份记录 | 有意改变配置不宣称精确重放；真实GPU重放待测。 |
| 48 | 题面去重保留大小写数学语义 | 语义近重复仍需语料层分析。 |
| 49 | 文本人名要求明确答案匹配，排除仅包含错误姓名的句子 | 不能承诺覆盖所有自然语言判分边界。 |
| 50 | 数字单位白名单，代数后缀不当单位 | 判分协议已版本化，旧新结果必须注明协议。 |

## 3. 新配置的可解释性

新入口依次叠加 `minimal_gpu.yaml`、`minimal_gpu_repaired.yaml`、`minimal_gpu_deeper.yaml`，不覆盖原始配置文件。等步数模式默认40步；t=512、训练horizon=2048、R−b、ChatML、`repetition_penalty=1`、无 `min_tokens`、单卡限制继续保留。

以下改动会改变实际方法或实验对象：每步题数2→4、reservoir64→128、IPW/fit-only建基、warmup末刷新、固定scaler、独立γ校准、ridge权重mean归一化、β=.5→.75、HF BF16计算、评测题与判分协议。新结果必须称为修订配置的结果，不能当作旧版本重跑的唯一变量因果解释。共同SFT与更多seed解决比较设计问题，不自动提高GRACE的学习能力。

第二轮再加入crossfit建基、年龄权重、额外当前策略监督及实测成本调度。这些改变需明确标记。crossfit每个fit题解一次小样本ridge，按高维块遍历两遍；调用方仍堆叠G，额外空间为O(n²+n×65536+Dk)，不是免费预测器。窗口/衰减也不能将历史监督变成当前策略监督；独立新采流在日志中标为 `fresh_policy`。

原始配置文件保留也不等于旧源码行为完全复现：求解器、判分和测量修正属于本次代码变化。严格历史重放需原提交、原环境和原权重；这次归档缺少的权重仍是缺证。

如需归因，复用同一共享actor和数据manifest，每次只覆盖一组选择：`basis_center`、IPW、γ校准、β、scaler/特征模式、ridge尺度。保留每次实际成本、全体结果和负结果；不从多个试验中只报告最好的seed。当前默认是可检验候选，不是已证明最优配置。

## 4. 如何运行与取回证据

在已安装依赖、准备 MODEL/TRAIN_DATA/EVAL_DATA 的单卡服务器：

```bash
export CUDA_VISIBLE_DEVICES=1
bash scripts/run_minimal_gpu.sh runs/minimal-repaired
python scripts/archive_run.py runs/minimal-repaired runs/minimal-repaired.tar.gz --include-checkpoints
```

共同实测预算比较使用：

```bash
bash scripts/run_matched_cost_gpu.sh runs/minimal-matched-wall
```

每个seed先让Full-PG完成参考40步，从其实际运行时钟导出另外三方法的共同预算与目标batch时长；其余方法在预算边界结束，`MAX_STEPS` 默认10000仅为步数上限。也可用 `RUN_WALL_SECONDS` 明确给共同秒数。实际训练步数、超额、是否先撞到步数上限全部保存，不能只根据“requested budget相同”宣布耗时严格相等。若预算结束时还没离开warmup，正常报告分配未启用。

baseline精度对照可用 `ABLATION_CONFIG=configs/experiments/baseline_prescan16.yaml`。跨配置比较时设置 `SHARED_INIT_CHAIN=原链根目录`，复用该链每个seed的共享actor；同时显式指定相同 `RUN_WALL_SECONDS`。只改prescan，不从多次运行中挑最好的seed。该对照尚未在GPU执行。

若该根目录已运行过，脚本创建带时间戳的新根并在控制台打印；归档使用该实际路径。`SEEDS=17` 可运行一份较短链；这不会被标作完成3seed。`EXPERIMENT_CONFIG` 可指定显式覆盖配置，命令尾部可指定方法列表。

结果位置：

- 每个 `seed-N/shared-init`：共同 SFT 权重及成本。
- 每个 `seed-N/<method>/train`：`initial_actor.json`、有效配置、checkpoint、逐步/逐轨迹日志、prescan、logprob probe、predictor诊断、账本及资源事实。
- `eval-0/20/40`（存在的快照）与 `eval-final`：真实终点评测可指向已有同一步评测，避免重复；缺终点评测不冒用step40。评测发生在训练后的快照上，不能算在线time-to-target测量。
- `audit`、`audit-training` 与 `batch-audit`：协议单列；批审计保存raw、RNG、冻结状态hash、全空间均值/方差及优化器更新，不用JL替代全空间。
- `seed-N/comparison.json`：本seed全部方法和成本；`seeds_comparison.json`：以训练seed为单位的描述性汇总。3seed的标准差不等于显著性结论。

等步数与共同预算模式在汇总中明确区分。后者已能根据服务器实测安排运行，但尚无真实结果；边界超额、共卡干扰和步数上限须一起阅读。time-to-target还需合适的在线评测设计，不能从停止比例、token减少或事后插值补出结论。共享SFT成本单列，比较总成本时两侧采用相同分摊约定。

## 5. 验证与当前结论

本机运行全套 `python -m pytest tests -q -o addopts=''`，结果为 **421 passed, 1 skipped in 47.24s**，详见 [测试状态](TEST_STATUS.md)。覆盖新预测器数学、按题交叉拟合与监督年龄、当前策略监督隔离、四方法共享起点CPU链、停止者null、HT/梯度符号、固定批全空间与Adam回放、墙钟反馈与预算恢复、行为分数缺失、实际终点评测、故障注入、判分反例、投影尺度、协议错配与归档清单；附件评审吸收另增加50项测试，覆盖分配/补全/baseline对照、非正交基底误差分解和实际优化器输入精度。Git Bash 对两个运行脚本的语法检查、训练与审计CLI帮助检查和 `git diff --check` 通过。墙钟控制测试使用合成时钟，不能当作GPU性能数字。

9月17日原包只读回归：原方差×token成本比仍为 **2.321800041853085**，原ρ曲线保持不变。新增同成本反事实也与前轮复算一致，属于旧观测的离线诊断，不是新GPU结果。

本机没有GPU。尚未获得修订后四方法的质量、实际速度/峰值显存、HF/vLLM全响应差异或多seed效果，因此当前科学结论仍是：**9月17日的idea未获正向验证；本轮已修正确定缺陷并建立更可比较、可追溯的实验，但尚不能宣布idea通过或效果很好。**

下一次结果最关键的是：新题上冻结预测器的残差是否下降；adaptive是否优于同成本uniform；这种改善是否经clip/Adam转化为质量；计入全部实际成本后是否还有收益。若未出现，应按独立诊断定位U/f、风险分配、监督陈旧或固定开销，而不是再用停止率替代收益。

## 6. GRACE_review.md 的取舍与实现

评估来源为用户提供的 `GRACE_review.md`（2026-09-18），SHA256：`696846d455212506fed13c902640f2a9a113c5ab6d7a4e0ee4f76ccef38dcd77`。本轮对照的是当前修复工作树；附件主要审阅的是旧提交 `a1ca77b`。附件中的运行指令、目标和时间表作为建议，没有自动执行；引用的文献及其新颖性判断本轮未独立核实，也未写成已验证事实。

| 附件建议 | 取舍及实际实现 |
|---|---|
| §3、4.3、4.5：共同起点、dual ridge、保存开销、CountSketch、真坐标、训练目标审计 | 前两轮已覆盖，保留并验证现有实现，避免重复建设。 |
| §4.4：GRPO不应为未使用的历史baseline支付prescan | 已移除GRPO及GRPO-short的prescan和无用历史更新；组均值优势不变，主采样流不变。 |
| §4.4：固定baseline或平滑prescan对照 | 新增显式`baseline.mode=fixed`，以及EMA模式的独立prescan先验平滑。默认仍EMA/无平滑。fixed跳过预扫与历史更新；平滑只发生在未见题的独立预扫初始化，后续HT历史估计不裁剪、不平滑。 |
| §6.2：保守分配 | 新增`allocation.uniform_shrink`，默认0。在未完成前缀上，令q=Σp_i c_i/Σc_i，再用(1-s)p_i+s q；保持原solver实际达到的预计成本及预算偏差，自然结束仍p=1。成本全零时q=1。不能把这个预算代理称为真实秒数。 |
| §8：分离补全收益、全组件p=1对照 | `predictor.control_variate=false`只关闭actor估计器中的m；继续训练原U/heads/risk并支付辅助开销，是固定风险模型的补全贡献对照，不是重新训练最优m=0风险策略。`allocation.beta=1`在训练和审计决策中全部续写，包括零风险前缀。 |
| §4.5B/C、§8：分开基底遗漏与坐标预测误差 | 在已有全空间||G||²和UᵀG之外保存UᵀU，分解真实子空间遗漏与子空间内预测误差，报告逐观测及按t汇总。对非正交/降秩U使用Gram伪逆；旧包缺字段明确缺失，不从JL反推全空间数值。 |
| §5：clip/Adam可能无法兑现原始梯度改善 | 固定批审计补充raw梯度、真实clip路径返回向量和优化器参数步三阶段配对方向、平方误差；零向量cosine或零参考相对误差为null。 |

baseline平滑对照采用`(sumR + strength × prior_mean)/(n + strength)`，示例配置的strength=2、prior_mean=.5只是待测候选。独立样本成本仍完整计入；fixed并不保证方差低于历史baseline。

新增配置只按需叠加，不自动改变默认训练配方：

| 覆盖文件（configs/experiments/下） | 要回答的问题 |
|---|---|
| `grace_full_completion.yaml` | 保留辅助组件、p=1时是否回归Full-PG原始梯度，以及这些组件花费多少成本？ |
| `grace_no_cv.yaml` | 保留风险分配时，预测补全是否提供额外收益？ |
| `allocation_uniform_shrink.yaml` | 向同预计成本的uniform收缩一半，是否缓解错误排序造成的损失？ |
| `baseline_fixed.yaml` | 固定b=.5、去掉prescan后，质量与实际成本如何变化？ |
| `baseline_smoothed.yaml` | 相同prescan样本量下，先验平滑是否减少零baseline问题？ |

例如只运行补全对照（服务器上执行，本机未启动）：

```bash
SEEDS=17 ABLATION_CONFIG=configs/experiments/grace_no_cv.yaml \
  bash scripts/run_minimal_gpu.sh runs/minimal-no-cv full_pg grace
```

跨配置比较应设置`SHARED_INIT_CHAIN`复用同seed初始actor；墙钟比较使用已有`run_matched_cost_gpu.sh`，并显式复用相同`RUN_WALL_SECONDS`。`comparison.json`保存变体名称与baseline/allocation/CV配置，避免多个“grace”结果混淆。Uniform-HT、Uniform-CV已可作为脚本尾部方法参数，无需新增方法框架。

暂未采纳成对续写交叉协方差建基、冻结Adam预条件器作为学习度量、大型prompt baseline或vLLM引擎内剪枝。它们可能有研究价值，但目前缺少证明该额外复杂度和标签成本值得支付的观测；当前交叉拟合建基也不等于成对续写交叉协方差方案。先用上述分解和机制对照定位瓶颈，再决定下一次方法变更。未把附件中的样本数、两周排序、论文目标或自动停止建议变成运行门槛。
