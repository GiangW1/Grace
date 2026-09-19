# 9月18日实验的根因、9月19日实现与待验证项

这份记录对应用户提出的六个问题。依据是 [9月18日实验逐文件审查](MINIMAL_RESULTS_REVIEW_20260919.md)，代码基于新仓库 `GiangW1/Grace` 的 `1bb75a8`（训练代码为 `3c03ce9`）。本轮修改在 `codex/grace-efficiency-20260919`；历史实验结果没有改写。

**结论：已经修复可确认的代码问题、减少重复计算，并实现两个明确的研究候选和配对实验链；尚不能宣称梯度补全带来真实训练效率提升。** 本机没有 GPU，本文件中的 CPU 验证不是新训练成绩。9月16日和9月17日结果不作为本轮修复的效果证据。

## 1. 六个问题的处理状态

| 问题 | 本轮具体处理 | 尚未解决的实证问题 |
|---|---|---|
| 1. 后20步仅7步有非零补全，最终 γ=0 | 新增 γ 的设计加权分子、分母、未裁剪值、有效观测数；加入同 G、同 p 的 m=0 干预；实现检查预测与真梯度对齐的候选建基目标 | γ=0 是否能被更好的预测器改变，以及改变后是否减少残差/成本，必须新跑；不强制 γ>0 |
| 2. reservoir86但新鲜22、fit10中仅2条非零G | 审计标签复用主训练反向，候选收集所有已完成梯度并取消额外 fresh 续写；使用独立预扫的先验平滑；监督日志区分已/未自然完成前缀 | 高维、小样本、按题隔离、策略漂移仍存在；标签更多不等于有效信号更多，零标签不删除 |
| 3. 方差×token成本未赢，风险分配未优于uniform | 修复全零早期残差把风险尺度永久锁为1的问题；冻结多臂审计输出同成本uniform、m0、p1及两种原始梯度方差×token估计 | 风险排序、补全残差和真实成本能否同时改善未知；token代理不等于GPU节省 |
| 4. 保存后N16→12→8→4 | 加入真正的 `cost_control.mode=fixed`，同时绕开wall反馈和旧token回收；wall模式分离主轨迹近似边际成本与摊销固定成本；快照无损紧凑保存 | 真实固定开销仍会压缩wall模式的可用预算，修控制模型不能消灭开销 |
| 5. 停止未转化为净节省 | 每条非零完成轨迹一次前向/求导同时服务actor和监督；降低压缩级别、无损压缩G存储类型；候选取消fresh额外生成，另提供无prescan的固定baseline对照 | HF前缀特征、行为核对、CPU拟合/拷贝、同步、I/O、预扫仍收费；端到端速度须实测 |
| 6. 弱基线和LAG证据不足 | 增加同方法四臂机制链，共享actor和输入设计、固定N；独立训练与冻结干预分开；记录eval引擎seed和逐请求轨迹 | 不借Uniform-CV/GRPO末分归因；LAG两个条件在同一适用群体是否同时成立仍待验证 |

## 2. 为什么会形成这条失败链

### 2.1 稀疏监督、独立划分与非平稳目标叠加

本次训练不断遇到新题。独立4条prescan很容易给出 b=0；当主轨迹也失败时，R−b=0，合法的策略梯度标签就是零。原始完成轨迹316条中222条优势为零。这个事实不能改记成有梯度，也不能只保留成功样本。

之后还叠加了审计抽样、reservoir容量、12步年龄窗口、按题fit/hold/calibration划分。在末次拟合侧10条中只有2条非零梯度，难以同时支撑高维跨题预测、建基、风险学习和独立γ校准。旧标签还来自过去的actor；放宽窗口会增加样本并增加策略漂移，缩短窗口则相反。两者都不是无代价的修复。

本轮候选把审计概率设为1，用训练必须计算的完成轨迹G补监督，不再另买fresh续写；保留所有零梯度和按题隔离。独立4样本prescan加2个先验伪计数、先验均值0.5，使新题初始化 b=(成功数+1)/6。它仍在主采样之前确定，仍计算R−b。它可能缓解离散baseline造成的零优势，但也会给失败样本引入非零梯度噪声，不能以“非零标签变多”宣布改善。

### 2.2 预测幅度不等于可预测信号

已有 `predictable_crossfit` 用整题留出预测 M=A G 的能量选低秩方向。虽然每条预测没有使用同题标签，但较大的预测幅度仍可能是噪声或反向预测。CPU反例中，预测能量目标会保留一个与留出真值反向的方向。

新候选 `predictable_crossfit_signal` 使用

```text
C = (Gᵀ W M + Mᵀ W G) / 2
```

的正特征方向，其中M仍是按题留一预测，W包含既有IPW与年龄权重；建基只用fit侧。利用 `G Gᵀ` 的小矩阵等价求解，按维度分块，不创建D×D矩阵。原能量目标保留，候选通过配置显式启用。

这只是更有针对性的代理目标：有限样本的正谱也可能由噪声产生；筛选出的方向没有经过独立效果验证；历史G不等于当前策略的条件均值。若没有正方向，则报告无新基并保留旧U，不伪造秩，也不强行清零/开启补全。

### 2.3 γ=0可能是正确的保护，而非程序失效

固定前缀可测的m和p，选择噪声为

```text
ΔV = (1/N²) Σᵢ (1/pᵢ − 1) ||Gᵢ − mᵢ||² .
```

γ校准应检查能否降低这个残差，而不是奖励非零预测。新日志给出设计权重下的内积和预测平方项：没有有效权重、分母为零、内积为负，是不同原因。缺信息时保留历史值并报告；负方向可以合理收缩为0。校准数据仍不进入坐标拟合和建基。

固定N时，原始HT/CV梯度相对全续写的方差通常是 `V_full + E[ΔV]`，不是无条件小于 `V_full`。补全应当比同p的m0减少ΔV，再靠成本降低赢得效率。这个恒等式不意味着clip/Adam更新无偏，也不把一步方差直接等同于长期训练质量。

### 2.4 调度器形成了开销放大效应

旧wall反馈把包含prescan、fresh、拟合、同步、保存的完整batch秒数除以N，当作每条轨迹成本。保存峰值因此影响下一批N；N缩小后，同样的预扫/拟合费用分摊到更少主轨迹，主监督进一步变少。

新wall反馈用主轨迹阶段秒数/N估计近似边际成本，用其余完整墙钟的历史均值摊销固定费用；所有费用仍计入预算。这不是精确GPU吞吐模型。**用旧观测N和耗时做算术回放，新模型依然可在10/15/20步建议12/8/4，最终仍为4**：旧末段固定费用均值约40.82秒，原目标约48.54秒，实际留给主训练的空间已经很小。该回放不能预测修改后真实运行会用多少时间。

所以机制实验采用显式fixed模式固定N=16；真实等成本实验另外运行。仅关闭 `cost_control.enabled` 不够，旧token回收仍可能改变N；此问题已在CPU测试中先复现再修复，恢复同一fixed配置也验证保持N。

### 2.5 辅助工作超过省下的续写

9月18日后20步主生成token仅少2.02%，包含prescan/fresh后反而多29.94%；整个训练墙钟GRACE比Full-PG约慢61%。这不能由停止人数解释为加速。

本轮直接消除一次重复计算：以往非零完成轨迹先为actor前后向，被审计后再前向求导取G；现在一次 `autograd.grad((R−b)logπ)` 得到G，给优化器累积 `−Z G/(Np)`，并按独立审计掩码保存未加HT权重的G。审计掩码不改变actor梯度路径；停止者没有完整G；零优势保存精确零标签。全方法使用同一路径，补偿流、clip、Adam与固定N均保留。

这不表示监督免费：GPU→CPU拷贝、存储和拟合仍付费。拷贝计入backward阶段，因此单看audit阶段变短会高估优化。固定容量下更高审计率也会更快替换旧标签。

另一个可选配置使用固定b=.5、prescan=0，同时取消fresh。它省掉预扫生成，但改变梯度方差和学习行为；四臂都必须使用相同baseline。它是成本/基线消融，不是证明LAG不依赖baseline。

## 3. 确定代码修改和验证范围

| 修改 | 位置 | CPU验证要点 |
|---|---|---|
| 主梯度与审计G复用 | [actor_update.py](../grace_gc/trainer/actor_update.py)、[algorithm.py](../grace_gc/trainer/algorithm.py) | 全空间HT与符号；p=1；审计开关不改actor梯度；停止/零优势/unused参数；FP32与原累积在误差范围内一致 |
| 真正固定N及恢复 | [cost_control.py](../grace_gc/trainer/cost_control.py)、[loop.py](../grace_gc/trainer/loop.py)、[verl_trainer.py](../grace_gc/backends/verl_trainer.py) | 模拟旧token建议仍固定N；同配置续训；四臂tiny训练实际同actor/同输入/N |
| 固定成本与主成本分离 | [cost_control.py](../grace_gc/trainer/cost_control.py) | 合成时间、保存尖峰、预算包含固定费用、配置切换和旧状态恢复 |
| checkpoint无损紧凑存储、低级别压缩 | [checkpoint.py](../grace_gc/trainer/checkpoint.py) | 原子替换/故障、旧NPZ读取、dtype/数值/符号位往返、不能由float32精确表示的G保持float64、输入payload不变 |
| 风险尺度零残差冷启动 | [scale.py](../grace_gc/predictor/scale.py)、[update.py](../grace_gc/predictor/update.py) | 全零残差不锁定fixed尺度，后来有信号时可拟合；全零风险训练仍执行 |
| Reward-CV风险特征与当前q一致 | [update.py](../grace_gc/predictor/update.py) | 更新后的成功头q用于风险拟合，历史长度/b保持原标签上下文；这不是9月18日四方法失败原因 |
| γ原因与监督组成可见 | [scale.py](../grace_gc/predictor/scale.py)、[update.py](../grace_gc/predictor/update.py) | 无信息/负内积/正内积区分；未额外修改γ公式或ρ平均 |
| 预测交叉矩基候选 | [basis.py](../grace_gc/predictor/basis.py) | 反向预测反例、加权显式D×D等价、秩亏、零权重、无正谱 |
| 训练批形状与冻结多臂审计 | [batch_audit.py](../grace_gc/audit/batch_audit.py)、[audit_batch.py](../scripts/audit_batch.py) | 精确checkpoint.step日志优先；不得用next_n/过期last_n；同G/同选择uniform随机数；p1、m0、uniform预算；原始方差token分母排除prescan |
| 引擎seed与评测执行轨迹 | [generate.py](../grace_gc/evaluation/generate.py)、[run.py](../grace_gc/audit/run.py) | eval及audit继承顶层seed、尊重显式vllm.seed；batch回退仍保留逐请求seed；请求ID和输入输出token hash |
| 四臂脚本及证据汇总 | [run_mechanism_gpu.sh](../scripts/run_mechanism_gpu.sh)、[summarize_mechanism.py](../scripts/summarize_mechanism.py) | Git Bash替身训练/审计实际走通四臂、共享初始化、配置覆盖、继承数据路径、只做一次终点多臂审计 |

checkpoint使用标准NPZ，DEFLATE level1；只在float64 G可以按值及符号位精确往返float32时缩小存储，加载恢复原dtype。actor、优化器、U没有进行有损压缩。降低压缩级别可能增大某些文件。

合成CPU内存序列化微基准：24条G，其中17条全零，每条262144元素，G合计48MiB；每种方式3次，原默认压缩中位1.490秒，level1加无损紧凑存储0.387秒，输出约11.99MB→10.91MB。只测BytesIO，包含紧凑转换，不包含磁盘/fsync/GPU；不是本次被排除checkpoint的重放，也不能外推端到端加速倍数。

## 4. 已实现的服务器实验

### 4.1 四臂机制比较

```bash
SEEDS=17 bash scripts/run_mechanism_gpu.sh runs/mechanism-fixed
```

默认仍叠加 `minimal_gpu_deeper.yaml`，再加 `mechanism_fixed.yaml`。依次跑完整GRACE、p1、m0、uniform；都以 `method=grace` 加标量配置实现。每个seed共享一份格式SFT actor，固定16 starts/4题，40步，四臂保留预测器工作，用于隔离机制。p1设β=1；m0关闭actor补全；uniform设同预计成本预算的统一收缩。**p1带预测器开销，因此不是高效Full-PG的替代基线。**

各臂学习后actor、优化器、标签和风险模型会自然分叉；共同初始actor/seed不保证未来回答相同。汇总检查完整问题及prompt-token顺序、实际N及初始hash；缺证据报告缺失，原始观测仍保留，不设置启动或继续门槛。

四次训练/评测后，只从GRACE终点做一次多臂冻结批审计。这里m0保留该checkpoint风险与p，估计“关掉当前补全项”的一步作用；独立m0训练会产生自己的后续标签与模型。两者不能混为同一个结果。冻结审计全部后缀真实生成并收费，不是运行中实际省算。

### 4.2 信号候选

```bash
SEEDS=17 \
COMMON_CONFIG=configs/experiments/minimal_gpu_signal_candidate.yaml \
SHARED_INIT_CHAIN=runs/mechanism-fixed/grace \
bash scripts/run_mechanism_gpu.sh runs/mechanism-signal
```

`SHARED_INIT_CHAIN` 应指向第一条链实际打印的GRACE子链路径；若同名目录存在，脚本会另建带后缀的路径，按 `mechanism.json` 中的 `arms.grace` 取值。也可省略该变量重新创建共享actor。它只共享actor初始化，不恢复旧预测器/旧优化器；候选需要重新训练，不能把旧checkpoint的预测器状态直接当成修复后的学习过程。

候选改动为 `audit_s=1`、`fresh_samples_per_problem=0`、`basis_variant=predictable_crossfit_signal`、4样本prescan的独立先验平滑。其他deeper设置保留，四臂一致。用 `COMMON_CONFIG` 追加，勿误用会替换deeper层的 `EXPERIMENT_CONFIG`。

若需要区分“多标签”“平滑baseline”“交叉矩目标”分别贡献多少，可把这三个改动分别写成单项覆盖配置，用同一脚本比较。先跑组合候选只能检验整体，不能归因于其中一项；不预先承诺哪项必然有效。

### 4.3 无预扫成本消融

```bash
SEEDS=17 \
COMMON_CONFIG=configs/experiments/minimal_gpu_cost_candidate.yaml \
SHARED_INIT_CHAIN=runs/mechanism-fixed/grace \
bash scripts/run_mechanism_gpu.sh runs/mechanism-cost
```

它将所有四臂baseline都设为固定.5、prescan=0。批审计现有代码识别固定训练baseline并跳过prescan，不因deeper里原有的 `baseline_mode=prescan` 额外采样。若整片任务奖励恒零，非零baseline只会制造有限样本梯度噪声，不能制造正确方向；必须同时看质量、留出残差与方差，不能只看标签密度。

### 4.4 真正的效率比较

固定N四臂用于机制识别，不能代替与高效Full-PG的共同墙钟比较。已有入口也支持同一候选：

```bash
SEEDS=17 \
COMMON_CONFIG=configs/experiments/minimal_gpu_signal_candidate.yaml \
SHARED_INIT_CHAIN=runs/mechanism-fixed/grace \
bash scripts/run_matched_cost_gpu.sh runs/signal-matched-wall full_pg grace
```

Full-PG先提供实际预算，GRACE按共同预算跑；包括训练入口后的加载、同步、拟合、辅助生成和保存，报告batch边界超额及步数上限。独立共享SFT、评测和审计成本也保留在各自账本，另报整链总成本，不把它们藏进“免费准备”。不用停止比例、理论节省或主采样token代替墙钟。质量应比较相同实测成本下的表现，或达到同质量的实际耗时，并保留中间点和失败尝试。

初次单seed只做定位，随后沿相同设计扩展17/23/41；独立题bootstrap区间不能替代训练seed差异。没有最低样本量门槛，小样本正常输出但结论相应保守。上述命令尚未在GPU执行。

## 5. 如何读新结果，什么仍可能失败

1. 先看新题、更新前的冻结残差相对m0是否降低，再看γ分子/分母与补全覆盖。只看到非零预测而残差变大，属于无效预测。
2. 同checkpoint、同p下m0与GRACE的条件额外方差差别，检验当前补全项。uniform比较检验风险排序是否值得。p1回归检查估计器；其本身不产生节省。独立m0训练保留GRACE风险学习管线，是补全开关消融，并不是重新优化风险目标后的最佳adaptive-HT基线。
3. 批审计同时给实际抽样raw协方差×预计主token比，以及 `(V_full + 平均条件额外方差)/V_full × 预计主token比`。默认8个重复批，估计会很噪；V_full为零/缺失时为null。均不包括clip/Adam的效率解释，更不是GPU速度。
4. 长程独立四臂再看质量、主/辅助token和完整墙钟。固定N的p1仍支付预测器开销；论文性能结论需与优化过的Full-PG比较。GRPO/Uniform-CV初期不稳定应单独诊断，不因末分低就算GRACE机制成功。
5. LAG需要答案不确定与梯度相对确定在同一指定群体/位置成立。保持原ρ平均和全体/筛选/独立report分列；不能从不同子集分别挑一个达标值拼成机制证据。新信号基不是LAG实证的替代品。

此外仍有下列边界：

- 当前实测有效秩3、监督少不等于“真实条件均值必然不存在”；也不能由零标签变多/变少反推idea正确。高维前缀特征是否含足够跨题可泛化信息仍未知。
- 风险尺度修复不会追溯修复旧checkpoint中已标记fitted且scale=1的状态：无法区分合法尺度1与早期零残差回退。新实验从共享actor重新训练。
- fixed模式续训应继续传入同一有效配置。当前请求配置可以有意改变调度模式；不同配置续训不叫精确重放。
- 顶层seed传给eval/audit引擎修复了非17 seed可能仍用17的问题；**它不能解释9月18日seed17下同初始actor仍有不同eval输出**。新增请求chunk、fallback、seed、ID、token hash及导出adapter hash帮助下次定位，但没有验证vLLM内部权重hash或GPU逐token确定性。
- HF/vLLM行为策略一致性、混合精度误差、显存峰值、主机拷贝/拟合代价仍需真实单卡检查。只允许n_gpu=1；未接多卡。
- 旧包排除的NPZ/adapter无法从hash还原。新运行保留原始轨迹、所有账本、audit原始记录及checkpoint；归档 `--include-checkpoints` 可保留NPZ，大JSONL仍受 `--max-file-mib` 限制，应检查归档清单并保留服务器原件。

不调整R−b、停止者null、ChatML、repetition_penalty=1、无min_tokens、全局N与G上升方向，不改parse/ρ平均，不增加Go/No-Go。最小实验仍是最小实验，不是Pilot或论文已完成实验。

## 6. 验证记录

最终合并树的CPU回归和脚本检查记录在 [TEST_STATUS.md](TEST_STATUS.md)。新测试包含四臂tiny训练及Git Bash替身编排；后者替换昂贵的训练/审计命令，只验证编排，不代表真实GPU链已完成。实际质量、净成本、LAG和多seed结果仍待服务器实测。
