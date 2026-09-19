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
- 旧包排除的NPZ/adapter无法从hash还原。9月19日追加修复后，归档默认保留所有JSONL和 `batch_audit_means.npz`，不受大小上限影响；`--include-checkpoints` 包含NPZ及LoRA adapter，或用重复 `--include PATH` 精确选择初始/最终状态。超限基础模型权重仍默认排除；完整规则、纳入原因和排除hash写入manifest，服务器原件继续保留。

不调整R−b、停止者null、ChatML、repetition_penalty=1、无min_tokens、全局N与G上升方向，不改parse/ρ平均，不增加Go/No-Go。最小实验仍是最小实验，不是Pilot或论文已完成实验。

## 6. 验证记录

最终合并树的CPU回归和脚本检查记录在 [TEST_STATUS.md](TEST_STATUS.md)。新测试包含四臂tiny训练及Git Bash替身编排；后者替换昂贵的训练/审计命令，只验证编排，不代表真实GPU链已完成。实际质量、净成本、LAG和多seed结果仍待服务器实测。

## 7. 追加修复：数据隔离、全程计时、归档与多seed统计

对应完整清单P26/P27/P28/P22，改动建立在 `a5d58e2` 之上，不修改9月18日原始实验文件或分数。

- **数据隔离（P26）**：CPU/GPU共用 `load_training_data`，先按原seed分割，再从train/calib/audit池排除外部评测题。比较题面而非题号/金标；规范化空白和精确DAPO包装，保留数学变量大小写，不声称检测语义近重复。`--eval-data-path` 传入完整评测语料，`run_minimal_gpu.sh` 自动把EVAL_DATA传给共享SFT和所有训练方法。原文件不改写，排除记录/评测文件hash存入 `data_exclusions.json`，实际池写 `data_splits.json`。未传入时标明not_provided，不阻断普通训练。新过滤不能净化已经训练过的旧共享actor；新实验应重新创建共享SFT，或确认旧来源没有相关曝光。
- **内部计时（P28）**：eval、prefix audit、batch audit从函数入口计时，覆盖环境采集、准备、结果/失败记录；finally只记一条envelope，避免结果保存失败时completed/failed双计。batch audit记录hardware.name。最终账本自身发布、CLI前置数据加载和进程退出由外层命令计时覆盖。
- **命令计时（P28）**：主链的SFT/train/eval/audit子进程通过 `measure_command.py` 测量启动至退出，失败也留记录。`command_timing.jsonl` 与 `.summary.json` 报告完整命令时长、第一条命令启动至最后一条退出的观测区间，以及未归因阶段空隙。空隙不冒充GPU计算；内外账本不能相加。该区间不包含首条被测命令之前的测试/准备、末条之后的工作，硬杀/不可写磁盘可能无法持久化。旧14.38分钟间隙无法事后自动归因。
- **未来归档（P27）**：保留上述关键原始证据，可按路径精确保留初始/最终checkpoint；不会凭hash恢复旧包漏掉的12.48GB。
- **跨seed区间（P22）**：增加配对训练seed差的Student-t均值95%区间，自由度n−1；保留原题目bootstrap并区分统计单位。区间假设配对seed差近似正态且独立同分布，少量seed下不稳，未经多重比较校正。n<2或缺SciPy时正常输出null及原因，均值/SD仍保留；安装 `pip install -e '.[analysis]'` 提供SciPy。新增pass@4均值/seed SD和完整主starts及来源；缺失、不完整、错配数据不补零。实现区间计算不等于已有多seed实验。

## 8. 用户四张目标表应如何验证

76.8%、+3.6pp、1.50倍、0.7875/0.7344、γ在[.4,.8]、35%覆盖和60%联合率均为用户提出的目标示例，**不是已测结果或理论保证**。不把它们写入运行门槛，不按评测结果挑选性地隐藏失败seed，不为了数值达标改奖励或ρ定义。

| 表 | 真正需要比较的量 | 不能混用的口径 |
|---|---|---|
| 同算力质量 | 相同设备/外部负载与请求预算，各seed实际完整成本、步数、starts、终点评测和配对seed区间 | 请求40分钟不等于物理恰好40分钟；当前按完整batch停止，会有overshoot，CLI/退出另有成本。§10已增加预算内已发布checkpoint选择和实际完整费用并列；这是事后可用模型比较，不能宣称进程恰好在截止时停止。 |
| 端到端成本 | 增量RL、共同SFT费用加回后的单方法成本，以及实际整链成本分列；所有主/辅助生成、拟合、同步、保存、失败均记账 | 同墙钟可以因更多更新而生成更多token；同40步也可能N不同。不能同时把固定工作量净节省要求强加到固定时间吞吐表，更不能仅看停止比例。 |
| 估计器收益 | 相同冻结状态、N、t、baseline、采样概率下GRACE对m0；同成本uniform；完整空间方差及token成本代理 | 在现有HT/CV设计下，完整梯度方差之外还有非负抽样项；补全应降低相对m0的额外项。表中Var=1.05等不是低于Full-PG单样本方差，而是方差×成本折中。token代理优于1也不等于GPU提速。 |
| LAG与补全机制 | 相同checkpoint、t、horizon、baseline、题/路径集合和独立report协议下成对的ρ与联合事件；同时报告全体、筛选和有效分母 | 1.3408与.5333来自不同集合，不能拼成一对。联合率不能由两条总体均值推出；“未观测到联合成立”不等于真实概率为0。非零比例、γ或realized-G覆盖不能替代留出残差及效率。 |

γ随原始预测尺度变化：原始预测乘c时，等效收缩可除c，而实际补全量不变，因此[.4,.8]不是尺度不变的成功定义。有效秩接近8、覆盖已实现G的35%也非必要；覆盖可能包含不可约后缀噪声。应保留合法零G和独立校准，不强制γ非零。

机制实验共用baseline/prescan配置；成本候选同样应用到Full-PG等适用方法。应用性能比较还应给各基线合理调参资源，不能以昂贵或已退化的Full-PG/Uniform-CV作为唯一参照。单次40步仍不能建立长期训练结论。

明确请求40分钟预算的已有用法（尚未在GPU执行）：

```bash
pip install -e '.[analysis]'
SEEDS="17 23 41" RUN_WALL_SECONDS=2400 \
COMMON_CONFIG=configs/experiments/minimal_gpu_signal_candidate.yaml \
bash scripts/run_matched_cost_gpu.sh runs/signal-wall-2400 full_pg grace uniform_cv grpo
```

这条命令从新共享SFT起点比较增量RL预算；SFT费用另列且加回单方法端到端成本，实际共享作业只计一次。基模预训练是共同既有资产，明确排除。未设置RUN_WALL_SECONDS时仍沿用Full-PG固定步数产生的实测预算，不能标成固定40分钟。应先使用§4的冻结/独立四臂识别补全效果，再扩大到多seed；数据不足或结果不利时继续正常记录。

当前仍需GPU定位P20重复评测差异、P29后端数值差异，检验P01–P08补全/风险有效性、P11/P14–P16净成本、P17基线稳定性、P18/P19/P22多seed公平效率，以及P23同组LAG。本轮修复不构成这些研究问题已经解决的证据。

## 9. 历史6144条评测回答全量重判（P30）

已在本机CPU对9月18日四方法eval-0/20/40重新执行当前 `rule_reward`，没有把保存reward当作重判输入。6144条回答对应5536个精确唯一 `(text, gold, truncated)` 输入；608条重复输入复用相同输入的重判结果。全部成功，错误0、未执行0、奖励差异0、提取差异0，1536条题级指标及12组汇总全部一致。原件和使用源码的前后SHA256一致。

Full-PG终点avg@4仍为0.712890625，GRACE仍为0.705078125，其余方法/阶段也未改变。因此这次核对没有为“历史分数算错导致GRACE未赢”提供支持，也没有产生任何新训练收益。

复判环境为math-verify 0.9.0、SymPy 1.14.0、NumPy 1.26.4，耗时601.0秒（本机评分复核时间，不是GPU训练成本）。当前reward源码与包内版本逐行一致；旧环境只记录present-no-version，不能独立证明依赖版本完全相同。这是同协议当前实现的完整一致性复核，不替代人工独立数学真值，也不能排除评分器的共同偏差；未扩展到训练/audit续写。

本机完整报告和逐条证据保存在 `_minimal_review_20260918_063159/reward_rejudge_20260919*`，未加入Git；其中 `.md` 含12组对照表，`.json` 记录输入hash/环境，`_answers.jsonl` 为逐答案旧新结果，`_issues.json` 的差异/错误列表均为空。

## 10. 继续落实问题清单：预算模型、定位工具和同组机制统计

本节建立在 `29aaefd` 上。没有新的GPU训练，没有强制开启γ，也没有调整奖励、优势公式、停止者null、parse rate或既有ρ曲线来追求目标数字。

### 10.1 同真实成本的模型选择（P19/P22/P28）

旧checkpoint保存的是写盘前的内部时钟，不能据此判断模型何时可用于评测。现在 `checkpoints.json` 为每步保存记录增加写盘、最新副本复制及hash完成后的可用时刻；命令包装器把自身单调时钟起点传给训练子进程，包含启动和准备。内部训练入口时钟另存，不能无声替代外层命令时钟。索引自身发布和后续费用仍在完整命令账本中；续训缺少累计外层命令历史时该字段留null。

`scripts/summarize_cost_quality.py` 输出 `cost_quality.json`：

- 每个已评测checkpoint的分数、hash、可用费用、步数和完整步日志可核验的主starts。
- 每个预算内最近的有效checkpoint；按时间选，不能择优分数、插值或默认采用超时终点。缺来源/hash/时钟、协议变化保持原值并列出问题。
- 预算点跨seed的avg/pass/步数/starts/费用均值与SD，以及GRACE对各对照的配对seed差和Student-t区间。缺失seed/方法保留，核对实际seed；不同题集、硬件或已存训练配方不混入一个均值/区间。跨seed允许训练与引擎seed及起始checkpoint路径变化，不要求各seed SFT actor相同。单seed不能产生有效训练seed区间。
- 同一checkpoint多个不同目录的重复评测不自动挑一次或择高值；原始点保留并标注，先用重复评测工具解释它们。最终别名指向同一目录不重复计入。
- 指定目标精度时给出**已观察到的最早越线模型**，不是精确越线时间、持续达到目标或验证集调参许可。应预先确定目标；探索后选择须如实标明。
- 整次作业实际命令费用、内部训练费用、本链共同初始化费用及本链初始化+训练之和分列。若通过SHARED_INIT_CHAIN导入既有SFT，本链只支付导入费用，原SFT费用需从源链另列，不能把该和当完整部署费用。不能把预算后工作免费抹掉；eval/audit成本另有命令账本，不与内部envelope重复相加。

主链默认保留0/20/40与实际终点评测，并在wall模式加评预算内最近的已保存模型。`EVAL_STEPS=all` 可评全部保存点；全部评测收费。`HARDWARE_CONFIG` 可选择项目已有单卡硬件配置，多卡限制不变。

```bash
SEEDS="17 23 41" RUN_WALL_SECONDS=2400 EVAL_STEPS=all \
bash scripts/run_matched_cost_gpu.sh runs/deeper-wall-2400 full_pg grace uniform_cv grpo

# 对已完成的新链可查询多个预算；稀疏保存点的实际用时一并输出。
python scripts/summarize_cost_quality.py runs/deeper-wall-2400 \
  --budgets 1200 1800 2400 --target-avg 0.75
```

启动/退出、保存间距、超时完整批仍使实际费用不严格相等。该实现让偏差可见，并提供物理截止前确实已有的模型；它不把旧9/18链补造为40分钟公平实验。旧链缺可用时钟/hash时曲线只能保留未核实原分数。

### 10.2 重复评测与HF/vLLM定位（P20/P29/P16）

新增 `scripts/check_eval_repeatability.py`，既能只读比较现有目录，也能对同一checkpoint启动多个全新evaluate子进程。核对已记录actor身份、seed、题集、评测协议、模型/tokenizer/引擎配置、依赖及运行环境；逐样本报告token hash、首处分歧、长度、chunk与串行回退差异。batch size变化标为单因素对照，不能冒充同协议重复。子进程参数、日志、完整费用和失败结果保留。

```bash
python scripts/check_eval_repeatability.py compare EVAL_A EVAL_B --run-dir runs/eval-compare
python scripts/check_eval_repeatability.py repeat --config TRAIN_RUN/config.yaml \
  --backend gpu_verl --checkpoint TRAIN_RUN/checkpoint.npz --data-path "$EVAL_DATA" \
  --model-path "$MODEL" --seed 17 --repeats 2 --batch-sizes 1 4 \
  --run-dir runs/eval-repeatability
```

新评测记录checkpoint文件hash、checkpoint actor内容身份及GPU加载HF LoRA的数值配置。已有probe复用已经计算的逐token分数，补充有符号差、RMS、分位数、最差token/序列位置、prefix/continuation分段，以及HF精度/attention/LoRA和vLLM主机元数据；不额外生成或用诊断数值改变p、奖励和梯度。

`collect_versions` 优先读取安装包元数据，可记录math-verify具体版本，避免只为查版本导入vLLM等包；没有distribution元数据时才回退到模块属性。来源一并记录。安装版本信息不是GPU依赖实际可执行的证明，CUDA环境检查仍保留。

本机真实CPU两次独立评测子进程已跑通，合成tiny模型逐token相同；历史9/18初始评测再次只读比较仍是512对中374对token序列不同。两者都**不是GPU重复性已经解决**的证据。完整base权重和vLLM worker内张量未独立hash，kernel或批处理影响仍需服务器单因素定位。

### 10.3 同组LAG与审计不确定性（P23/P08/P25）

`independent_report.joint_lag` 新增逐前缀成对ρ及联合事件：同一checkpoint、位置、detect/report划分和report行索引，保留梯度/奖励分子分母、选择状态、缺失原因。按位置报告全部、可拆分、selected、有效配对、联合成立的数量，以及前缀联合比例、题均联合比例和同一有效集合的两项ρ均值。空集/无效分母留null；默认分析目标ρL≤.5、ρA≥.8可通过 `analysis.joint_lag` 改统计阈值，不影响训练运行。

原有headline及ρ平均保持原样，汇总同时透传新的独立统计。CPU反例确认“两条总体均值各自达标，但没有任何一个前缀同时达标”确实可能，故不能再由边际均值推联合率。前缀共享题目，不给它们套独立Bernoulli置信区间。

`variance_cost_uncertainty` 增加冻结审计数据上的整问题簇bootstrap，整个问题的路径、位置和续写一起重采样，保留可复算充分统计量、区间及未定义抽样次数。默认1000次、seed17，可用 `analysis.variance_cost_bootstrap` 调整；0仅关闭重采样。它假设问题可交换，不重拟合预测器/分配器，不是训练seed区间、固定题批方差、完整算法不确定性或GPU效率区间。问题少或无方差的重采样会不稳定，原样报告。

本轮还实际复算了9/18 GRACE训练口径原始bundle：原path split完全一致，方差×token比 **1.1372756571** 精确重现。60条report行来自11个问题簇；1000次整问题重采样中965次比值有效、35次未定义，有限抽样条件下的percentile95区间为 **[1.09251,1.16293]**。同组LAG在t=512、horizon=2048时仅选中1个前缀，其4条report续写得到ρL=.45560、ρA=.53333，联合为0/1。没有新GPU运行，没有重建U或加载缺失权重，也没有把缺失多位置审计混入；完整来源及限制保存在 `_minimal_review_20260918_063159/lag_vc_supplement_20260919.json`。它继续支持“这份冻结审计尚未获益”，不能推广为完整训练显著失败或总体联合概率为0。

### 10.4 基线退化及标签供应的逐轨迹复核（P17/P02/P03）

新增 `scripts/summarize_training_dynamics.py`，按step×problem和warmup/正式阶段汇总，不重判答案。PG检查R−b，GRPO检查同组均值；停止者奖励、缺失G、单完成者组方差、截断、clip/参数更新及辅助生成均按真实可观测字段处理，缺失不补0。示例：

```bash
python scripts/summarize_training_dynamics.py FULL_PG_TRAIN GRACE_TRAIN UNIFORM_CV_TRAIN GRPO_TRAIN \
  --output runs/training_dynamics.json
```

已对9/18日志实际执行，详细JSON及中文分析保存在本机 `_minimal_review_20260918_063159/training_dynamics_20260919.*`，未提交原始实验目录：

- 160步、1308 starts、1257完成者的优势关系一致；51停止者reward均null。四方法题组顺序相同、每法160题无跨批重访，但前20步starts分别144/260/168/252，采样暴露不同。
- GRACE的46条主审计标签仅10条非零，40条fresh仅15条非零；与86条中61条零G一致。最终fit10条仅2条非零仍是实际限制。
- UCV在第11–20步已有27/40零优势及9/40截断；正式阶段29/53完成者截断，均长1512.75。前20步尚无停止/CV，因此其早期退化不能归因于补全。
- GRPO每题实际3–4 starts，111/160组全零优势；没有group=1接线bug，后程也不是完全无信号。
- 23个当前批梯度精确0的步中22个仍有参数更新，符合Adam历史状态可能继续产生更新的机制；没有optimizer原件，不能把它直接判成bug或确定动量的贡献。

另修复 `summarize_mechanism.training_costs` 的可复现边界错误：辅助JSONL仅存部分条目时，原代码可能把可读token之和当完整费用。现在按逐步prescan题数×已存配置采样数、fresh条数/token事实核对，缺损留null并说明。历史9/18辅助日志在本次检查中完整；这个修复不能解释历史质量退化。

### 10.5 把组合候选拆成可归因的实验（P02/P03/P04/P06/P14/P18）

已有signal/cost组合候选继续保留；新增五份单因素配置，均叠在deeper之后：

| 配置文件 | 改动 | 要回答的问题 |
|---|---|---|
| `minimal_gpu_audit_all.yaml` | 主完成轨迹audit_s=1 | 多标签收益能否覆盖拷贝、存储和拟合费用？ |
| `minimal_gpu_no_fresh.yaml` | fresh_samples_per_problem=0 | 取消额外生成后是否仍有足够当前策略监督？ |
| `minimal_gpu_signal_basis.yaml` | 交叉矩正谱建基 | 可预测方向是否改善独立残差，而不只是预测幅度？ |
| `minimal_gpu_smoothed_baseline.yaml` | 相同4次prescan加入先验 | 增加非零优势是否带来有效信号而非噪声？ |
| `minimal_gpu_fixed_baseline.yaml` | b=.5且无prescan | 辅助成本下降能否补偿baseline精度损失？ |

例如 `SEEDS=17 COMMON_CONFIG=configs/experiments/minimal_gpu_audit_all.yaml bash scripts/run_mechanism_gpu.sh runs/audit-all`。四机制臂使用同一候选、固定N和共享actor。跨配置比较时用 `SHARED_INIT_CHAIN=已有链路径` 引用经过数据隔离的同一共享起点，并记录初始化导入费用；不要把不同SFT起点的差当配置效应。先保留单因素及原设置，组合优选依据开发证据并明确记录；最终论文评测须独立，不能反复按测试集挑配方。

这些是可运行、可审计的候选，未获得GPU改善结果。若独立留出残差、同p的m0消融、净成本和同时间质量不能共同改善，论文核心主张仍然缺乏支持；代码完备不能保证自然数据中存在足够可预测的梯度信号。
