# 单层梯度可预测性实验

这个实验回答一个窄问题：在相同 frozen-prefix audit 数据上，只预测某一层的 Q/V LoRA 梯度，或者只预测该层沿参考训练方向的标量收益，是否比预测全局梯度更容易。

脚本默认只做 CPU 上较快的对照：4 个代表层、rank 8/64、zero/constant/Ridge。用 `--layers 12` 可以只跑一层；用 `--layers 0,12,24,35` 做快速层筛选。MLP 和多窗口特征都是可选项。

## 运行

先使用已有的 replay、reference replay、reference-check replay 和窗口特征目录：

```bash
python scripts/expected_gain_single_layer_suite.py \
  --replay-dir RUNS/predictor-replay \
  --reference-dir RUNS/reference-replay \
  --reference-check-dir RUNS/reference-check-replay \
  --feature-dir RUNS/multiwindow-features \
  --split-manifest RUNS/expected-gain-suite/split.json \
  --layers 0,12,24,35 \
  --run-dir RUNS/single-layer-suite
```

先只验证某一层时：

```bash
python scripts/expected_gain_single_layer_suite.py \
  --replay-dir RUNS/predictor-replay \
  --reference-dir RUNS/reference-replay \
  --reference-check-dir RUNS/reference-check-replay \
  --feature-dir RUNS/multiwindow-features \
  --split-manifest RUNS/expected-gain-suite/split.json \
  --layers 12 \
  --ranks 8,64 \
  --run-dir RUNS/single-layer-12
```

加入已经提取的无参数多窗口特征：

```bash
python scripts/expected_gain_single_layer_suite.py \
  --replay-dir RUNS/predictor-replay \
  --reference-dir RUNS/reference-replay \
  --reference-check-dir RUNS/reference-check-replay \
  --feature-dir RUNS/multiwindow-features \
  --include-multiwindow \
  --layers 12 \
  --run-dir RUNS/single-layer-12-multiwindow
```

## 输出

`single_layer_summary.json` 中包含：

- 每层 Q/V 参数维度、参考梯度范数、参考池方向余弦；
- `direct_layer_gain`：直接预测该层的 `mean_gradient · reference_layer_direction`；
- 每个 rank 和 predictor 的单层坐标 MSE；
- 单层均值梯度残差与 zero baseline（这里的 zero baseline 是 prefix mean gradient 的残差，不是 continuation second moment）；
- 投影能量比例；
- 投影梯度诱导的标量收益指标，包括 MSE、(R^2) 和符号准确率；
- 只依据 validation residual 选择的配置。

`basis_coefficients.npz` 保存每个 `layer_<id>_rank_<k>` 的小 Gram 分解系数。完整的 D×k 基底无需常驻磁盘，可以用训练集均值块和该系数重建。

## 解释规则

- 单层坐标 MSE 下降，只说明该层的低维表示更容易拟合；
- 单层 projected gain 超过常数基线，才说明它可能包含可用于排序的训练信号；
- 最终仍需用 held-out continuation、真实 AdamW 更新和同容量 vLLM refill/no-refill 对照验证；
- 只挑选 diagnostic 上表现好的层会产生选择偏差，层应只由 train/validation 选择。
