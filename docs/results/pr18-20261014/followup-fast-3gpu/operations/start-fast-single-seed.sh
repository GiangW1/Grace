#!/usr/bin/env bash
set -u
run=/mnt/mydata/wja/grace/runs/pr18-20261014/followup-fast-3gpu
ops=/mnt/mydata/wja/grace/operations/pr18-20261014
mkdir -p "$run"
echo $$ > "$ops/fast-single-seed.pid"
export CUDA_VISIBLE_DEVICES=7,2,3
exec /mnt/mydata/wja/grace/venv/bin/python /home/wja/grace-pr18-fast/scripts/run_small_core_experiment.py \
  --followup --seeds 17 \
  --hardware /home/wja/grace-pr18-fast/configs/hardware/rtx5090_rollout_3.yaml \
  --model-path /mnt/mydata/wja/grace/assets/models/Qwen3-4B \
  --data-path /mnt/mydata/wja/grace/assets/datasets/dapo-math-17k.parquet \
  --eval-data-path /mnt/mydata/wja/grace/assets/datasets/math500.jsonl \
  --run-dir "$run" --execute
