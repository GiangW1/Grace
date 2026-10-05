#!/usr/bin/env bash
set -euo pipefail
exec /SSD/00/wja/GRACE/runs/conda/envs/grace/bin/python -u \
  /SSD/00/wja/Grace-pr14-learning-completion-20261004/scripts/supervise_learning_completion.py \
  --run-dir /SSD/00/wja/grace-pr14-learning-completion-36-20261004-run \
  --data-path /SSD/00/wja/GRACE/data/dapo/data/dapo-math-17k.parquet \
  --gpus 1 3 \
  --request-concurrency 32 --vllm-memory-utilization 0.65 --max-num-seqs 32 --batch-continuations \
  --config /SSD/00/wja/Grace-pr14-learning-completion-20261004/configs/default.yaml \
  --config /SSD/00/wja/Grace-pr14-learning-completion-20261004/configs/experiments/reasoning_mechanism_audit.yaml \
  --config /SSD/00/wja/Grace-pr14-learning-completion-20261004/configs/experiments/learning_completion_overlay.yaml \
  >> /SSD/00/wja/grace-pr14-learning-completion-36-20261004-run/logs/supervisor.log 2>&1
