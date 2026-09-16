# GRACE-GC — 项目约定

## 用户已确定的工作方式

先完成论文方法、核心对照和实验脚本的代码，再在服务器运行最小证伪。写代码采用奥卡姆剃刀原则：能用简单函数、普通配置和少量脚本完成，就不增加框架、审批或流程状态。

2026-09-15 用户的本次纠正替代此前较重的初始化流程。日常实现、修复、测试和本地提交直接推进；不再要求先调用某个 GSD 命令、批准路线图或通过文档门槛。GSD 仅用于需要时记录简短计划和进度，用户不必逐步解锁。不要自动生成额外的研究合同、覆盖矩阵、风险评分器或完成认证系统。

## 实现原则

- 首版范围：完整 GRACE、Full-PG、Uniform-HT、Uniform-CV、Reward-CV、Prompt-CV、GRPO、GRPO-short，以及最小证伪所需数据、日志、评测和审计脚本。
- 先实现已有论文方法。成本常数对照、不同秩等用参数切换；避免预先设计插件系统或多层抽象。
- 核心可在 CPU 导入与测试；真实训练沿论文的 verl/vLLM/FSDP 路线实现。GPU 依赖只在相应入口加载。
- 同一个方法的 CPU 参考与 GPU 实现共用数学函数，避免维护两套业务逻辑。
- 保留必要的正确性测试：全空间 HT、p=1 回归、梯度符号、固定全局 N、RNG 隔离、LoRA 参数布局和分布式归约。
- G 表示梯度上升方向；标准优化器的 .grad 对应 -G。全部 q/v LoRA A/B 属于目标；token log-prob 求和。
- actor、baseline、basis、predictor 在一个批次内保持不变；选择只读前缀；停止者 reward=null。这些是算法定义，不是额外流程。
- 保存每次运行的配置、seed、可获得的版本信息、结果和实际时间。缺少可选 hash/登记文件时照常运行并记录缺失。
- 官方 DAPO parquet 直接加载：同一题面金标冲突的整组丢掉并写入 `data_conflicts.json`，不要改回硬失败。MATH-500 套 DAPO 同款 `Answer:` 指令。`format_warmup` 是共享格式 SFT，不是 `predictor.warmup_steps`。SFT 只训推理开头，不训 `Answer:`、金标和 EOS，避免短答交卷。
- `backend=gpu_verl` 只是配置别名，实现是 HF actor + vLLM 两阶段，不是 verl PPO。第一份 GPU 作业先 Full-PG。`lora/step-N` 是 vLLM adapter id（首次 sync 为 `step-2`），不是 `state.step`。
- 只有会使程序无法执行或算法计算无效的输入才报错，例如文件不可读、p≤0、维度不一致。小样本、未达论文目标、宽置信区间都正常输出结果。
- 论文中用于分析的筛选条件和阈值作为可配置的统计选项，报告全部观测与所选子集；不作为启动/继续实验的条件。
- 不新增最低样本量、置信区间宽度要求或自动 Go/No-Go 门槛。理论疑点记录在实现笔记，不能要求先完成理论重推才写代码或跑实验。
- 简单记录哪些测试已跑、哪些需要 GPU。合成测试数据与真实实验结果分开；不能虚构测试或性能数字。
- 当前 Windows 本机无真实实验条件。规划硬件是 4×A100，随后两组各 8×RTX5090；**现在能启动的只有单卡**（`a100_1.yaml` / `rtx5090_1.yaml`），`n_gpu>1` 会拒绝。
- 一次 run 的配置、轨迹、健康事实和开始/结束时间写进 `--run-dir`。同名再跑（非续训）会加 UTC 后缀，不覆盖上一份。
- 中文沟通，代码标识符使用英文。

## 项目文件

- .planning/PROJECT.md：范围与优先级。
- .planning/ROADMAP.md：代码完成 → 最小证伪。
- .planning/REQUIREMENTS.md：精简功能清单。
- .planning/MINIMAL_FALSIFICATION.md：实验脚本的用法设想。
- .planning/research/PITFALLS.md：数学与实现笔记，按实现需要查阅。
- docs/TEST_STATUS.md：已跑/未跑测试；不写虚构 GPU 数字。
- README.md：服务器全流程。现役工作树若未推送，clone GitHub 拿到的是上一份。
