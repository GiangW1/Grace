# 分层 q/v 基底与无参数多窗口池化

本实验复用已有 `replay_expected_gain.py` 的完整梯度均值、独立的 `reference-audit` 和 `reference-check-audit`。不重新采样回答，也不改变训练算法。先对每个冻结前缀做一次 actor 前向，提取最终隐藏层的 prompt 均值、回答前 128 token 均值、中段均值、末 128 token 均值。中段为空时其向量为零；没有可学习的注意力权重。对照使用原特征；实验组在原特征后追加这四个窗口均值，保留原本的最后 token、中层隐藏态、熵、长度和 baseline。

```bash
python scripts/extract_expected_gain_windows.py \
  --replay-dir RUNS/predictor-replay --checkpoint CHECKPOINT.npz \
  --model-path MODEL --run-dir RUNS/multiwindow-features

python scripts/expected_gain_structured_suite.py \
  --replay-dir RUNS/predictor-replay \
  --reference-dir RUNS/reference-replay \
  --reference-check-dir RUNS/reference-check-replay \
  --feature-dir RUNS/multiwindow-features \
  --split-manifest RUNS/expected-gain-suite/split.json \
  --run-dir RUNS/structured-gain
```

后一个脚本在 CPU 上运行。它将同一训练集的全局 mean-SVD 64 维，与按深度四等分再区分 q/v 的八块基底比较；每块最多 8 维，总预算最多 64 维。每层 q/v 另外报告两个独立参考池的方向余弦和能量占比。两种基底分别配原特征与追加窗口特征，用零、常数和 Ridge（惩罚 1/10/100）预测完整梯度坐标；同时比较两种特征对标量期望收益的直接预测。只按 validation 的全空间残差选配置，diagnostic 问题独立报告。小样本数值秩不足时记录实际秩，不把它伪装成 64 维。

`structured_gain_summary.json` 保存划分、逐项结果和耗时；`basis_coefficients.npz` 保存小 Gram 分解系数。结合报告中的 `train_indices` 与 block segments，可按 `U_block = G_train_block.T @ coefficient` 重建基底。特征目录保存对应 actor、checkpoint、前缀顺序和布局的身份信息，套件会核对，避免错配。GPU 前向只提取特征；本地 CPU 测试不会产生真实 A100 性能数据。
