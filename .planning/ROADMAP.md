# Roadmap: GRACE-GC

## Overview

**先完整实现代码，再进行最小证伪实验。** 采用奥卡姆剃刀原则，普通函数、配置和脚本足以解决的问题不增加额外框架。

本路线图已按2026-09-15用户纠正更新，直接执行该方向，无待审批步骤。取代原六阶段方案和研究合同前置阶段。

## Phases

- [x] **Phase 1: 完整实现 GRACE 与实验代码** — 主方法、核心对照、GPU后端和实验脚本全部写完，完成本机可跑的测试。
- [ ] **Phase 2: 运行最小证伪实验** — 将代码放到服务器，用小配置观察真实结果，再决定后续实验。

## Phase Details

### Phase 1: 完整实现 GRACE 与实验代码

**Goal:** 得到有实际实现、可部署运行的完整研究代码，本机完成必要的正确性验证。
**Depends on:** Nothing (first phase)
**Requirements:** CODE-01, CODE-02, CODE-03, CODE-04, CODE-05, CODE-06, CODE-07, CODE-08, CODE-09, CODE-10, CODE-11, CODE-12, CODE-13, CODE-14, CODE-15, CODE-16, CODE-17, CODE-18
**Success Criteria** (what must be TRUE):

1. 全空间HT、续写分配、tiny LoRA与预测器有实现，并通过相关CPU数学/数值测试。
2. verl/vLLM/FSDP两阶段生成、双流更新、保存恢复和核心对照有实际代码。
3. 数学数据、真实成本记录、独立评测、前缀审计和结果脚本可以通过简单配置调用。
4. 提供A100/5090配置和运行说明，如实列出需要在服务器补跑的检查。

**Plans:** 4 个短执行清单已完成。`.planning/phases/` 未保留，现役说明以 README 和本文件为准。

建议编码顺序：

1. 项目骨架、核心估计器/分配器、CPU tiny LoRA。
2. 历史梯度、基底、预测器、审计和自适应训练。
3. verl/vLLM/FSDP连接、分布式更新、核心对照。
4. 数据、计时、评测、前缀审计、最小实验入口与README。

以上都是同一代码阶段内的实现顺序。不要把准备研究合同、理论重推或尚无GPU验证变成停工理由；遇到具体问题直接修复。

### Phase 2: 运行最小证伪实验

**Goal:** 用完成的代码在真实服务器获得小规模结果，判断方法与现象的后续研究方向。
**Depends on:** Phase 1
**Requirements:** RUN-01, RUN-02, RUN-03
**Success Criteria** (what must be TRUE):

1. 在目标GPU上完成一次训练/审计链路的实际运行，出现工程错误能够复现定位。
2. 小规模实验输出原始观测、梯度统计、解题结果和实际成本。
3. 按结果说明支持、反例或不确定之处；不以额外效果阈值决定程序能否继续运行。

**Plans:** TBD — 代码完成后按实际机器与可用checkpoint选择小配置。

## 工作约定

- 日常实现、测试、修复和本地提交直接推进，不重复请求路线图或阶段批准。
- 不增加来源登记齐全、最低样本量、置信区间宽度、盈利预测等启动门槛。
- 论文原有统计筛选只用于分析，同时保留全体样本摘要；不足时照常保存结果并说明。
- 非法概率、维度错误、文件不可读等实际错误仍正常报错修复。
- 规划硬件仍是 4×A100 与后续 8×5090；当前代码只接受单卡。
- 大规模确认实验、外部完整基线及扩展消融后置。

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. 完整实现 GRACE 与实验代码 | 4/4 | Complete | 2026-09-15 |
| 2. 运行最小证伪实验 | 0/TBD | Not started | - |

Phase 1 代码已落地。本机（2026-09-16）：`pytest tests -k "not complete_final_expression and not u6_gpu"` → 214 passed, 9 deselected。真实 GPU 实验未跑到可引用；2026-09-16 两次服务器 smoke 都不能当 Phase 2。按 README 先 smoke Full-PG。
