# Requirements: GRACE-GC

**Updated:** 2026-09-15

按用户最新要求：先完成全部核心代码，再运行最小证伪。以下是功能清单，不是实验准入规则；替代此前43项形式化需求。

## v1 Requirements

### Phase 1 — 完整代码

- [x] **CODE-01**: 提供简单配置、CLI 与 CPU/GPU 可选依赖；支持本地模型/数据路径，随运行自动保存可获得的版本信息。
- [x] **CODE-02**: 实现全空间 HT 估计器、双流等价形式、p=1 回归和负损失符号；用 NumPy FP64/小模型检查 U1–U4。
- [x] **CODE-03**: 实现包含 p_min 的续写成本分配与边界处理，返回实际 p 和预算偏差；非法概率或无法计算的输入明确报错。
- [x] **CODE-04**: 实现 CPU tiny LoRA 自回归训练，涵盖 token-sum、固定 N、RNG 隔离、自然 EOS、零幸存和信息泄漏负例 U5。
- [x] **CODE-05**: 实现 step 前的完整 q/v LoRA per-sample 梯度审计、独立抽样 s 和完整历史梯度 reservoir。
- [x] **CODE-06**: 实现低秩 basis/刷新重投影、前缀特征、坐标头、全空间风险头、Gram 残差与成本头/常数成本选项。
- [x] **CODE-07**: 实现历史数据和问题级拆分上的 IPW 预测器训练，使用 1/(p*s) 并按当前 basis/预测器重算风险标签。
- [x] **CODE-08**: 实现批内冻结、历史 baseline、warmup p=1、批后学习与基于历史成本决定下一批固定起步数。
- [x] **CODE-09**: 实现真实 verl/vLLM/FSDP 两阶段 rollout：相同快照、原始 token IDs、只续写被选者、明确 RNG/缓存处理；避免占位后端。
- [x] **CODE-10**: 实现分布式固定全局 N、完整参数布局、负预测校正、AMP/归约/clip 顺序，以及 FP64 对照和 U6/U7 测试入口。
- [x] **CODE-11**: 保存和恢复 actor、optimizer、预测器、baseline、basis/reservoir 与 RNG 等实际续训状态。
- [x] **CODE-12**: 实现 Full-PG、Uniform-HT、Uniform-CV、Reward-CV、Prompt-CV、GRACE；提供独立实用轨道 GRPO、GRPO-short 配置。
- [x] **CODE-13**: 实现数学数据去重与训练/校准/审计/评测分割、规则奖励、答案首次可解析位置与自然结束/截断处理。
- [x] **CODE-14**: 实现部署/研究成本记录，按 GPU 预留墙钟记时、处理重叠并另列 CPU 时间，不把不同 GPU 时间直接折算。
- [x] **CODE-15**: 实现独立完整作答评测：论文数学基准配置、avg/pass@k、解析/截断率、time-to-target 和真实验证的困难题首次成功 HVD。
- [x] **CODE-16**: 实现可调规模的同前缀独立续写审计、全空间/JL 核对、正交补诊断、rho_L/rho_A、t_L/t_A、ELF/LAG/PLC 与区间统计。
- [x] **CODE-17**: 实现审计的独立拟合/评估划分，以及 Oracle/实际预测器在实际受限 p 下的方差成本计算；不把理论闭式写成实验开关。
- [x] **CODE-18**: 提供最小实验、完整 Pilot 参数、4×A100/8×5090启动配置与简短说明；记录已跑测试、未跑项目和必要实现差异，结果不足照常输出。

### Phase 2 — 最小证伪实验

- [ ] **RUN-01**: 代码完成后，在4×A100目标环境安装依赖并短跑完整训练链路，实际遇到设备/数值错误就定位修复。
- [ ] **RUN-02**: 用可调的小规模配置运行同前缀审计及核心方法比较，保存所有观测；允许直接使用已有 checkpoint。
- [ ] **RUN-03**: 汇总梯度误差、解题结果、实际成本及不确定性，判断后续实验方向；无需满足额外自动 Go/No-Go 条件。

## 不建设的额外机制

独立研究合同阶段、强制来源登记/审批、实验准入状态机、最低有效样本/区间宽度门槛、自动 Go/No-Go 决策器，以及与当前代码无关的覆盖/安全认证流程均移除。论文原有定义用于结果分析，效果目标用于对照展示，不阻止试跑。

未知的可选元数据记录为缺失。只有实际不可执行或计算错误的配置报错，如路径不可读、依赖不可用、概率非法或张量维度不符。

## 完成代码的含义

主方法和上述脚本有实际实现，本机能跑的针对性测试通过，README 给出服务器运行方式。GPU 测试待服务器执行时注明即可；不要求先产生 GPU 性能证据才算完成本机代码工作。

## Later

外部实用基线全复现、完整 A1–A14 矩阵、代码任务、多模型扩展和大规模正式实验放到后续。最小实现不删主方法，只减少不必要的工程层次和流程。

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CODE-01 | Phase 1 | Code complete |
| CODE-02 | Phase 1 | Code complete |
| CODE-03 | Phase 1 | Code complete |
| CODE-04 | Phase 1 | Code complete |
| CODE-05 | Phase 1 | Code complete |
| CODE-06 | Phase 1 | Code complete |
| CODE-07 | Phase 1 | Code complete |
| CODE-08 | Phase 1 | Code complete |
| CODE-09 | Phase 1 | Wired: vLLM two-phase + HF logprob/backward/LoRA sync; GPU unverified |
| CODE-10 | Phase 1 | CPU reduce/clip done; U6 GPU pending |
| CODE-11 | Phase 1 | Code complete |
| CODE-12 | Phase 1 | Code complete |
| CODE-13 | Phase 1 | Code complete |
| CODE-14 | Phase 1 | Code complete |
| CODE-15 | Phase 1 | Code complete |
| CODE-16 | Phase 1 | Code complete |
| CODE-17 | Phase 1 | Code complete |
| CODE-18 | Phase 1 | Code complete |
| RUN-01 | Phase 2 | Pending |
| RUN-02 | Phase 2 | Pending |
| RUN-03 | Phase 2 | Pending |

21项需求均已映射。Phase 1 的 18 项代码已落地；Phase 2 的 3 项实验未跑。GPU 集成测试待服务器。
