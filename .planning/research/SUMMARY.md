# Project Research Summary

**Updated:** 2026-09-15；遵循用户最新的代码优先和奥卡姆剃刀要求。

## Key Findings

- 主方法和核心对照已由论文及用户确定，不需要继续泛化需求或创建实验治理框架。
- NumPy/PyTorch用于CPU参考，verl/vLLM/FSDP用于真实训练；版本信息见STACK.md，写对应模块时核查实际源码即可。
- 梯度符号、含p_min的分配、全空间残差和随机数隔离是需要写对的代码。
- A≤0的一般性不盈利结论有精确反例；直接实现实际p的数值方差成本计算，理论讨论不阻止编码。
- 配置、日志、结果和少量文档足够支撑首轮实验。未知元数据记为缺失，统计不确定直接报告。

## Implications for Roadmap

Phase 1完成全部主方法、核心对照、训练后端和实验脚本；Phase 2到服务器运行最小证伪。取消原“研究合同→多级验收→Go/No-Go”的流程。

## Sources

技术来源见STACK.md，算法与实现笔记见PITFALLS.md；当前权威范围和工作方式是PROJECT.md与AGENTS.md。
