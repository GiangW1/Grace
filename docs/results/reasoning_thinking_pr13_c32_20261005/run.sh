#!/usr/bin/env bash
set -euo pipefail
unset GRACE_METRIC_FILE GRACE_METRIC_NAME GRACE_DIFFICULTY_MANIFEST GRACE_DIFFICULTY_COUNTS
exec /SSD/00/wja/GRACE/runs/conda/envs/grace/bin/python -u \
  /SSD/00/wja/Grace-pr13-8cac23f-20261004/scripts/supervise_thinking_audit.py \
  --run-dir /SSD/00/wja/grace-pr13-thinking-v3-8cac23f-c32-20261004-run \
  --data-dir /SSD/00/wja/grace-pr13-thinking-v3-8cac23f-c32-20261004-run/inputs \
  --model-path /SSD/00/wja/GRACE/data/Qwen3-4B \
  --gpus 0 2 --n-problems 32 --n-continuations 32 --request-concurrency 32 --port 18013 \
  --vllm-memory-utilization 0.65 --max-num-seqs 32 --batch-continuations --feature-batch-size 2 \
  >> /SSD/00/wja/grace-pr13-thinking-v3-8cac23f-c32-20261004-run/logs/supervisor.log 2>&1
