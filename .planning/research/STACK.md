# Stack Research — GRACE-GC

**Researched:** 2026-09-15
**Status:** 官方资料核查与实现建议；未安装/验证 GPU 栈。

## Recommended Stack

| 层 | 选择 | 版本与置信度 |
|---|---|---|
| 数学参考 | NumPy、Python 标准库 | 本机 NumPy 1.26.4 / Python 3.11.7 已通过元数据检查；高 |
| CPU autograd 与 tiny LoRA | PyTorch、pytest | 本机 torch 2.7.1+cpu、pytest 7.4.0；是否运行正确留给 Phase 2；高 |
| 配置/合同 | YAML、JSON Schema 或等效严格校验、JSONL/Parquet | 本机 PyYAML 6.0.3；具体锁版本在 Phase 1；中 |
| GPU actor | verl + FSDP，必要时 FSDP2 | verl v0.9.0 是本次核查的候选发布标签；适配前固定 commit；中 |
| GPU rollout | vLLM，明确两阶段请求与独立 RNG 合同 | 候选 verl 发布要求 vLLM ≥0.18；不能仅靠该下界选择任意最新版；中 |
| 模型 | Qwen/Qwen3-4B-Base，q_proj/v_proj LoRA r=16 α=32 dropout=0 | 官方 config 已核实；权重 revision 尚未冻结；高 |
| 数学数据与奖励 | BytedTsinghua-SIA/DAPO-Math-17k、Hugging Face Math-Verify | 官方入口核实；数据 revision、清洗 hash 与依赖版本待冻结；高 |
| 部署 | Linux/NVIDIA，CPU 与 GPU extras 分离 | A100 4 卡优先，5090 两组 8 卡随后；硬件兼容待实测 |

## Version Policy

当前 latest 文档与 v0.9.0 发布源码已经有差异：latest 安装文档描述 uv 环境，而标签 pyproject 使用 setuptools。Phase 4 必须围绕同一个完整 commit 检查依赖、trainer/worker API、容器与示例，禁止把不同时间版本的接口拼在一起。本次不声称存在已验证的 CUDA/PyTorch/vLLM/verl 组合。

模型 config 的 36 层、hidden_size=2560、32 query heads、8 KV heads、head_dim=128 与框架一致；据此计算 q/v rank-16 LoRA 参数数为 5,898,240。代码仍应从实际 trainable 参数枚举获得布局与 hash，不能硬编码此维数。

选择纯 PyTorch 参考 actor 作为数学与接口对照；GPU 后端缺失时应提供明确的安装/预检错误，核心 NumPy/CPU 功能保持可用。主实验不默认量化，因为这会改变框架的梯度目标与硬件比较口径。

## Sources

- [verl v0.9.0 发布](https://github.com/verl-project/verl/releases/tag/v0.9.0)
- [v0.9.0 pyproject](https://raw.githubusercontent.com/verl-project/verl/v0.9.0/pyproject.toml)
- [verl 当前安装文档](https://verl.readthedocs.io/en/latest/start/install.html)
- [verl LoRA 配置](https://github.com/verl-project/verl/blob/main/docs/advance/ppo_lora.rst)
- [Qwen3-4B-Base config](https://huggingface.co/Qwen/Qwen3-4B-Base/raw/main/config.json)
- [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- [Math-Verify](https://github.com/huggingface/Math-Verify)
