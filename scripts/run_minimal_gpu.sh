#!/usr/bin/env bash
# Sequential single-GPU experiment; environment and assets come from local setup.
set -eo pipefail
cd "$(dirname "$0")/.."
source runs/setup/env.sh
source runs/setup/fetch-assets.exports
experiment_root="${1:-runs/minimal-chain-$(date -u +%Y%m%d-%H%M%S)}"
if (($#)); then shift; fi
methods=("$@")
if ((${#methods[@]} == 0)); then methods=(full_pg grace uniform_cv grpo); fi
mkdir -p "$experiment_root/logs"
exec > >(tee -a "$experiment_root/console.log") 2>&1
trap 'status=$?; echo "experiment exit=$status at $(date -u +%FT%TZ) root=$experiment_root"' EXIT
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
git log -1 --oneline
python -m pytest tests | tee "$experiment_root/logs/tests.log"
common=(--config configs/default.yaml --config configs/experiments/minimal_gpu.yaml --config configs/hardware/a100_1.yaml --backend gpu_verl --model-path "$MODEL" --seed 17)
for method in "${methods[@]}"; do
  python scripts/train.py "${common[@]}" --method "$method" --data-path "$TRAIN_DATA" \
    --num-steps 40 --run-dir "$experiment_root/$method/train" \
    | tee "$experiment_root/logs/$method-train.stdout"
  train_run=$(awk '$1 == "run_dir" {path=$2} END {print path}' "$experiment_root/logs/$method-train.stdout")
  for step in 0 20 40; do
    python scripts/evaluate.py --generate "${common[@]}" --data-path "$EVAL_DATA" \
      --checkpoint "$train_run/checkpoints/step_$step.npz" \
      --run-dir "$experiment_root/$method/eval-$step" \
      | tee "$experiment_root/logs/$method-eval-$step.stdout"
  done
  python scripts/audit.py --generate "${common[@]}" --data-path "$TRAIN_DATA" \
    --checkpoint "$train_run/checkpoint.npz" --run-dir "$experiment_root/$method/audit" \
    | tee "$experiment_root/logs/$method-audit.stdout"
  python scripts/summarize_minimal.py "$experiment_root"
done
python scripts/summarize_minimal.py "$experiment_root"
