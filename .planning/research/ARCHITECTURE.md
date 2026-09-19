# GRACE 简单实现结构

**Updated:** 2026-09-19；职责已按此落地，细节以代码和 [README](../../README.md) 为准。

## 模块按实际职责拆分

| 模块 | 职责 |
|------|------|
| core | 估计器、概率分配、RNG与LoRA参数布局；CPU可用 |
| predictor | reservoir、basis、特征、坐标/风险/成本头、IPW |
| trainer | CPU小模型及GRACE训练步骤、核心对照、checkpoint |
| backends | 单卡 HF actor + vLLM 两阶段、LoRA 同步；FSDP 包装与归约参考不代表多卡已接线 |
| data / evaluation | 数学数据、验证、完整作答评测 |
| audit | 独立续写、梯度统计与成本方差分析 |
| scripts / configs | 训练、审计、图表、运行配置与README |

包名拟用grace_gc。先写简单函数和普通数据结构，确有重复再抽取公共接口。不要预设插件注册系统、复杂能力协商或运行准入状态机。

## 训练数据流

1. 读取普通配置和输入，保存seed及可获得的版本信息；根据历史成本确定本批N。
2. actor/baseline/basis/预测器保持本批不变，生成前缀并获取detach特征。
3. 对未自然结束者计算实际p，从独立选择RNG抽Z，只续写被选者。
4. 完成者取得真实奖励和负PG损失；step前抽样获取完整per-sample梯度。
5. 合成负预测校正，按全局N归一化，执行一次归约/clip/optimizer.step。
6. 更新历史数据与预测器/basis/baseline；保存训练状态、概率、奖励和实际时间。

自然结束者p=Z=1，停止者reward=null。CPU与GPU实现共用核心数学；不把GPU import放到核心包初始化中。

## 日志与实验

一个运行目录保存配置、训练日志、结果和checkpoint即可；有需要再增加审计向量文件。日志包含实际p/Z、seed、快照和basis标识以解释更新，不为“记录完整性”另造认证层。

vLLM prefix caching复用KV，不代表自动恢复续写随机状态。写GPU适配时核查并测试对应版本的采样行为；相关检查服务于算法实现。

## Sources

- [vLLM prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- [vLLM reproducibility](https://docs.vllm.ai/en/v0.22.0/usage/reproducibility/)
- 本地论文§5.3及§6。
