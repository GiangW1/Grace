# Requirements: GRACE-GC

**Defined:** 2026-09-15
**Core Value:** 保持原始梯度目标和真实账本，以最小可信实验判断主假设成立、否定或证据不足。

范围已经用户确认：完整主方法、核心对照和最小证伪流水线。本里程碑完成代码和本机可执行验证；GPU 实验执行单独安排。

## v1 Requirements

### 研究合同与运行基础

- [ ] **CNTR-01**: 研究者能从 claims.yaml、research_plan.yaml、experiments.csv、source_registry.json 读取主张、计划、实验与来源；所有未运行观测值为 null，并与预注册目标分开。
- [ ] **CNTR-02**: 研究者能校验模型/数据 revision、规范化题干 hash、split 身份与来源；允许明确的草稿配置，正式实验入口在缺少必需身份时拒绝启动。
- [ ] **CNTR-03**: 研究者能在 amendments.md 追踪 SPEC-01 至 SPEC-05 及统计口径澄清：负损失校正符号、含 p_min 的预算、理论适用边界、特征维度、PLC 分母；保留原论文。
- [ ] **CNTR-04**: 研究者能以不同配置运行 CPU 与 GPU 入口；核心包在没有 verl/vLLM/CUDA 时可导入，未知配置项和不支持的方法明确报错。
- [ ] **CNTR-05**: 研究者能用统一 run/snapshot/basis/predictor/parameter-layout ID、RNG 身份、固定 N 与 trajectory schema 追溯每个决策；停止样本 reward=null。
- [ ] **CNTR-06**: 研究者能分别查看 implementation、CPU 验证、GPU 验证和研究证据状态；synthetic/mock 不进入真实结果汇总。

### CPU 数学与小模型闭环

- [ ] **CORE-01**: NumPy FP64 参考实现通过有限枚举验证全空间 HT 均值、方差恒等式及正交补期望恢复（U1–U3），覆盖非零预测和异质概率。
- [ ] **CORE-02**: 研究者能比较逐样本向量估计与双流估计逐位一致，p=1 回归完整原始梯度，并检查一次 SGD 位移的符号（U4）。
- [ ] **CORE-03**: 续写分配器直接在 [p_min,1] 下匹配期望成本，覆盖零风险/零成本、不可行 beta、空可决策集合与极端数值，返回实际预算误差。
- [ ] **CORE-04**: tiny 自回归 LoRA 策略可在 CPU 从生成、奖励到参数更新完整运行：q/v A/B 均入目标、token-sum、固定 N、批内冻结、每批一次更新。
- [ ] **CORE-05**: 独立问题/轨迹 token/续写/选择/审计 RNG 流支持重放；改变选择抽样不改变同前缀的潜在后缀；后缀/奖励进入决策器的负例被拒绝（U5）。
- [ ] **CORE-06**: 参考 rollout 正确处理自然提前 EOS、共同长度上限、零幸存者和不等长 microbatch；停止轨迹不运行后缀/奖励/整轨反向传播。
- [ ] **CORE-07**: 成本方差参考实现按实际受限 p 计算盈利比，保留 A=0 可盈利反例和无前缀信息均匀截断反例；闭式仅在其可行条件成立时输出。

### 历史预测器与自适应训练

- [ ] **PRED-01**: 决策特征只含已观察前缀，detach 后按实际维度进入互不共享参数的坐标头和风险头；成本预测可配置为学习头或显式常数对照。
- [ ] **PRED-02**: 完整 q/v A/B per-sample 梯度在 actor step 前获取，不修改 actor.grad；完成者以独立概率 s 审计并记录纳入概率 p*s。
- [ ] **PRED-03**: 完整历史梯度 reservoir 支持容量限制、主机内存与持久化；按题级历史中心化构建低秩 basis，每次刷新重投影并更新 basis_id。
- [ ] **PRED-04**: 全空间风险标签保留 norm²、UᵀG 和实际 UᵀU Gram 项，与显式全向量残差一致；风险损失使用 e/rhat+log(rhat)。
- [ ] **PRED-05**: 坐标/风险训练只用历史且按 problem_id 分离的数据，损失使用 1/(p*s)；基底或坐标预测器改变后按当前版本重算风险标签。
- [ ] **PRED-06**: 基线、预测器、basis、风险与成本头在整批冻结，完成 actor 更新后才更新历史状态；warmup p=1 与 checkpoint 恢复保持这一时序。
- [ ] **PRED-07**: 研究者能在 CPU 自适应闭环中按历史实测成本预先确定下一批固定起步数，匹配目标预算并限制资源占用；抽 Z 后不删超额完成者。

### verl/vLLM/FSDP 训练后端

- [ ] **BACK-01**: 提供固定源码版本的 verl/vLLM/FSDP 适配实现与依赖清单；CPU/GPU 后端使用相同核心合同，不能仅留下未实现的训练占位接口。
- [ ] **BACK-02**: vLLM 后端以原始 token IDs 在同一 actor/LoRA 快照下生成前缀并仅续写被选者；明确 KV 缓存和 RNG 的能力及失效行为。
- [ ] **BACK-03**: 训练采样与求导均使用完整策略：T=1、不做 top-p/top-k 截断；提供 p=1 单阶段/两阶段分布与确定性测试入口，并记录硬件待验证状态。
- [ ] **BACK-04**: actor 以固定全局 N 和实际 p 计算 token-sum 原始 PG，合成负预测校正；不等长 microbatch 与各 rank 幸存数不改变目标。
- [ ] **BACK-05**: 参数布局包含全部 LoRA A/B，跨 DDP/FSDP 的分片、归约和校正只执行一次；零幸存 rank 正常参与通信并验证布局 hash。
- [ ] **BACK-06**: 提供 FP64 参考与真实分布式相对误差检验（U6）、符号/AMP unscale/全局分母/hash 检验（U7），区分 CPU 契约通过与 GPU 未运行。
- [ ] **BACK-07**: 训练 checkpoint 保存 actor、optimizer、历史 baseline/predictor/basis/reservoir/RNG/调度与账本游标，恢复后继续保持同策略批次边界。

### 数据、核心对照、账本与评测

- [ ] **PIPE-01**: 数学数据入口能固定 DAPO-Math-17k 版本，按 problem_id/题干 hash 去重并隔离训练、校准、审计、评测，输出被移除记录。
- [ ] **PIPE-02**: 数学奖励使用固定规则验证器，保留解析/截断/自然结束状态和答案首次完整可解析位置；相同轨迹在各方法下奖励规则一致。
- [ ] **PIPE-03**: 机制配置实际支持 Full-PG、Uniform-HT、Uniform-CV、Reward-CV、Prompt-CV、GRACE 和按范数自适应 HT 诊断；同目标、baseline、固定 N 与输入分布。
- [ ] **PIPE-04**: 实用轨道实际支持 GRPO 与 GRPO-short，保留各自目标；GRPO 的组均值/历史基线选项与长度/预算差异明确标注，不称其与机制轨道梯度等价。
- [ ] **PIPE-05**: 部署/研究两份账本保存预留 GPU 数乘墙钟、互斥分项或重叠归属及 CPU verifier 时间；硬件类型分开，不以 kernel 相加或 token 数代替总成本。
- [ ] **PIPE-06**: 独立评测完整生成答案，移除训练停止器；支持 MATH-500/AIME24/AIME25/AMC23/OlympiadBench 配置、avg/pass@k、题目配对不确定性及到目标预算删失。
- [ ] **PIPE-07**: HVD 只计固定困难题观察池中的首次真实完成且验证成功；输出累计覆盖与每 GPU-hour 发现率，停止/预测/重复成功不计。
- [ ] **PIPE-08**: 研究者能生成可运行的 Full-PG/GRPO 学习性 sanity 配置、共同格式 warmup 与审计快照清单；所有方法共同开销有据可查，输出未运行空表。

### 最小证伪与部署移交

- [ ] **PILOT-01**: 同前缀审计支持可配置题目/前缀/决策点/独立续写/快照规模；保留自然结束分母与删失，并能扩展到论文完整 Pilot 配置。
- [ ] **PILOT-02**: 审计存储固定 4096 维 JL 投影、完整梯度子集、全范数及坐标；用完整子集核对投影误差，并输出 LoRA A/B、q/v 与正交补诊断。
- [ ] **PILOT-03**: 统计管线实现 rho_L/rho_A、非平凡门、Wilson 区间、A/B 均值能量、t_L/t_A、ELF/LAG/PLC、问题层分层 bootstrap 与非单调越界率。
- [ ] **PILOT-04**: 完成时刻判定与报告使用预冻结的独立续写拆分；Oracle-GRACE 使用交叉拟合均值，未见题/历史预测器审计不能泄漏到训练或选参。
- [ ] **PILOT-05**: 盈利报告同时展示实际受限 p 的方差成本比、oracle/实际预测器、固定成本/可省成本/开销及闭式适用性；禁止用 A<=0 单独判 No-Go。
- [ ] **PILOT-06**: 证伪报告输出 go/no_go/inconclusive、effect/CI/target_met/statistically_supported、有效样本量和证据来源；小规模筛查与完整 Pilot 的门槛分别预注册。
- [ ] **PILOT-07**: 提供 4×A100 与单组 8×RTX5090 配置、两组任务分配说明及硬件/驱动/显存/拓扑/依赖预检；跨节点 16 卡须显式配置，成本不可自动换算。
- [ ] **PILOT-08**: 用户可依 README/运行手册完成 CPU 验证、部署预检、sanity、最小审计、报告生成；本机阶段将所有未运行 GPU 项逐项列出，合成数据报告强制标记。

## v2 Requirements

- **EXT-01**: ARRoL、VIP、VIGOR 及其他能公平复现的外部实用基线。
- **EXT-02**: A1–A14 完整实验矩阵、关键干预的多 seed 确认实验。
- **EXT-03**: 代码任务数据/低权限禁网执行沙箱及 LiveCodeBench/HumanEval+/MBPP+。
- **EXT-04**: Countdown、1.7B/8B、多决策点和全参数研究扩展。
- **EXT-05**: 大规模正式训练、完整论文图表与投稿材料。

## Out of Scope

| Feature | Reason |
|---------|--------|
| 本机真实模型训练/性能实验 | 用户明确暂不具备条件 |
| 自动使用服务器或消费 64 A100-days | 硬件告知不等于本轮实验执行授权 |
| 把目标值补成结果、合成曲线当实测 | 违反研究证据合同 |
| 当前多节点异种 GPU 混训或时间统一折算 | 不属于主方法且未经兼容/吞吐验证 |
| 推理时早退、教师/PRM/蒸馏 | 主方法改变训练计算分配，不改变最终完整作答协议 |

## Definition of Done

1. 需求代码实现完整，核心路径没有占位分支；CPU 数学、toy/autograd、数据与统计测试通过。
2. 真实 GPU 适配代码、锁版本策略、测试入口、部署与恢复文档齐备。未在硬件运行的检查标为 gpu_pending，不得计为通过。
3. 数学正确性、集成验证和科学结论分别记录。达到代码里程碑不表示 GRACE 有效或研究假设成立。
4. 研究合同和最小证伪配置可供下一阶段在服务器冻结并执行，未运行指标仍为 null。

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CNTR-01 | Phase 1 | Pending |
| CNTR-02 | Phase 1 | Pending |
| CNTR-03 | Phase 1 | Pending |
| CNTR-04 | Phase 1 | Pending |
| CNTR-05 | Phase 1 | Pending |
| CNTR-06 | Phase 1 | Pending |
| CORE-01 | Phase 2 | Pending |
| CORE-02 | Phase 2 | Pending |
| CORE-03 | Phase 2 | Pending |
| CORE-04 | Phase 2 | Pending |
| CORE-05 | Phase 2 | Pending |
| CORE-06 | Phase 2 | Pending |
| CORE-07 | Phase 2 | Pending |
| PRED-01 | Phase 3 | Pending |
| PRED-02 | Phase 3 | Pending |
| PRED-03 | Phase 3 | Pending |
| PRED-04 | Phase 3 | Pending |
| PRED-05 | Phase 3 | Pending |
| PRED-06 | Phase 3 | Pending |
| PRED-07 | Phase 3 | Pending |
| BACK-01 | Phase 4 | Pending |
| BACK-02 | Phase 4 | Pending |
| BACK-03 | Phase 4 | Pending |
| BACK-04 | Phase 4 | Pending |
| BACK-05 | Phase 4 | Pending |
| BACK-06 | Phase 4 | Pending |
| BACK-07 | Phase 4 | Pending |
| PIPE-01 | Phase 5 | Pending |
| PIPE-02 | Phase 5 | Pending |
| PIPE-03 | Phase 5 | Pending |
| PIPE-04 | Phase 5 | Pending |
| PIPE-05 | Phase 5 | Pending |
| PIPE-06 | Phase 5 | Pending |
| PIPE-07 | Phase 5 | Pending |
| PIPE-08 | Phase 5 | Pending |
| PILOT-01 | Phase 6 | Pending |
| PILOT-02 | Phase 6 | Pending |
| PILOT-03 | Phase 6 | Pending |
| PILOT-04 | Phase 6 | Pending |
| PILOT-05 | Phase 6 | Pending |
| PILOT-06 | Phase 6 | Pending |
| PILOT-07 | Phase 6 | Pending |
| PILOT-08 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 43
- Mapped to phases: 43
- Unmapped: 0

---
*Last updated: 2026-09-15 after scope confirmation and research*

