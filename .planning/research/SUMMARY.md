# Project Research Summary

**Project:** GRACE-GC
**Date:** 2026-09-15
**Method:** 主任务内完成论文到实现的研究与官方资料核查；未使用研究子代理。

## Executive Summary

第一版应沿着“可信数学目标 → CPU 可运行闭环 → 历史自适应预测 → GPU 双流实现 → 公平对照与真实账本 → 最小证伪”交付。用户已经明确主方法、核心对照和最小证伪范围，当前本机不运行真实实验。首个部署目标为 4×A100，未来两组各 8×RTX5090。

## Key Findings

- **Stack:** Python/NumPy/PyTorch 核心独立于 CUDA；verl v0.9.0 为服务器适配候选，GPU 依赖组合尚未验证；模型官方 config 与论文的 q/v LoRA 维数计算一致。
- **Features:** 完整机制轨道六方法与实用 GRPO/GRPO-short；全空间审计、历史预测器、真实两阶段生成与成本账本缺一不可。
- **Architecture:** 所有后端共用显式的前缀信息类型、快照身份、参数布局、概率日志和固定全局 N 合同。核心算法先有 NumPy/toy autograd 对照。
- **Critical issues:** 负 loss 梯度校正符号、含 p_min 的预算方程、拼接特征维度、PLC 成本分母需明确。A≤0 普遍不盈利存在精确反例，不能写入自动 No-Go 规则。
- **Verification:** 代码实现、CPU 验证、GPU 实测和科学结论独立标记。mock/synthetic 结果不能解锁真实性能声明。

## Implications for Roadmap

1. **研究合同与运行基础**：把歧义和理论边界固化成规格、数据状态和测试要求。
2. **CPU 参考训练闭环**：U1–U4、U5 负例、toy LoRA、RNG 与单步位移。
3. **自适应预测闭环**：完整梯度审计、reservoir、基底刷新、坐标/风险/成本/IPW、计算回流。
4. **GPU 后端**：固定版本 verl/vLLM/FSDP，真续写、参数分片、U6/U7 测试实现与待硬件验证记录。
5. **数据、对照、账本与评测**：数学任务、两轨比较、部署/研究成本、HVD 和独立解题。
6. **最小证伪与移交**：逐级审计配置、统计报告、实际受限成本方差预报、4×A100/8×5090 部署预检。

## Confidence and Gaps

数学反例和符号分析是本次精确核查，不是训练研究结果。上游源码会漂移，GPU 依赖版本与完整 revision 必须在适配阶段冻结；本次没有下载权重或数据、启动远端实验或验证显存/吞吐。小规模审计样本量、A100 显存、5090 节点互联等在执行前明确，均不妨碍先完成代码。

## Sources

技术入口与版本依据见 STACK.md；系统结构依据见 ARCHITECTURE.md；逐条原文问题与精确反例见 PITFALLS.md。研究范围以 PROJECT.md 的用户确认记录及根目录论文框架为准。
