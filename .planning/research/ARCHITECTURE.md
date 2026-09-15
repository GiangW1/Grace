# Architecture Research — GRACE-GC

**Status:** 建议结构，尚未实现。

## Components

| 拟定模块 | 职责 | 禁止依赖 |
|---|---|---|
| grace_gc/contracts | 配置、来源、run/trajectory/snapshot/basis 身份、状态机 | CUDA/verl |
| grace_gc/core | 估计器、受限分配器、RNG 分流、参数布局 | 数据集/网络/GPU 后端 |
| grace_gc/reference | NumPy oracle、tiny policy/LoRA、CPU 完整步骤 | 真实数据下载 |
| grace_gc/predictors | reservoir、基底、特征、坐标/风险/成本头、IPW | 当前未见后缀 |
| grace_gc/backends | rollout/actor 协议；CPU 与 verl/vLLM 适配器 | 绕开核心估计器合同 |
| grace_gc/data、rewards | 数学数据分割、解析、规则验证 | 测试/审计数据参与训练 |
| grace_gc/training | 冻结批快照、双流更新、核心对照、检查点 | 同批反馈提前更新预测器 |
| grace_gc/ledger、evaluation | 预留资源墙钟、分项归属、独立完整作答、HVD/pass@k | 伪造 GPU 时间 |
| grace_gc/audit、reporting | 独立续写审计、JL/完整梯度诊断、证伪报告 | 用训练预测误差自证现象 |
| configs、scripts、docs | CPU/4×A100/8×5090 配置、部署、预检、重放与移交 | 未验证的默认跨节点连接假设 |

## Data Flow

1. 读取冻结配置与来源；验证数据 split、参数列表 hash、snapshot、成本口径。
2. 在历史成本基础上先确定 N；冻结 actor、b、U、f、风险/成本头。
3. 生成前缀；自然结束直接 p=Z=1、f=0。可决策轨迹提取 detach 特征。
4. 用所有可见前缀解含 p_min 的预算方程；先记录 p 再用独立选择 RNG 抽 Z。
5. 仅被选者续写并取得真实 reward；继续使用冻结快照，保留原始前缀 token IDs。
6. 真实流形成负 PG 损失梯度；审计在 step 前取独立 per-sample 梯度，不污染 actor.grad。
7. 按唯一的全局归一化约定合成负 Ghat，统一 unscale/reduce/clip/step。FSDP 分片映射与 DDP 均值规则分别处理。
8. step 之后更新历史标签、基线、预测器和基底；记录旧/新身份，供下一批使用。
9. 两份账本、轨迹、checkpoint、审计与独立评测数据通过 run_id/step/problem_id 关联。

## Contracts Requiring Explicit Types

- PrefixObservation 不含后缀 token、reward、true gradient；决策器仅接受这种类型。
- SnapshotIdentity 同时覆盖 actor、baseline、basis、predictor 及参数布局。
- FlatParameterLayout 保存有序 name/shape/dtype/offset 与分片映射；G 与 loss_gradient 的符号语义明确。
- ContinuationDecision 保存 actual p、Z、决策时刻、各 RNG stream/counter。
- ResultProvenance 区分 synthetic、cpu_reference、gpu_measured；成本单位包含 device_type。
- BackendCapabilities 显式描述 batch invariance、prefix continuation、LoRA sync、per-sample gradient 支持。

## Build Order

研究合同 → CPU 真值闭环 → 自适应预测闭环 → 同数学合同的 GPU 适配 → 数据/对照/账本闭环 → 最小证伪与部署移交。每阶段交付可单独运行和检查的能力。

vLLM 的自动 prefix caching 复用已计算的 KV，但这本身不能证明续写随机状态可恢复。官方复现文档还限制同硬件/同版本且要求明确调度或 batch invariance 设置。需要分别检查缓存命中、语义分布、随机数隔离与确定性路径。

## Sources

- [vLLM prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- [vLLM reproducibility](https://docs.vllm.ai/en/v0.22.0/usage/reproducibility/)
- 论文 §5.3 A1–A7 与 §6 Algorithm 1（本地原始文件）。
