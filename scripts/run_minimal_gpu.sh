#!/usr/bin/env bash
# Sequential single-GPU comparison with one shared SFT actor per training seed.
set -eo pipefail
cd "$(dirname "$0")/.."
if [[ -f runs/setup/env.sh ]]; then source runs/setup/env.sh; fi
if [[ -f runs/setup/fetch-assets.exports ]]; then source runs/setup/fetch-assets.exports; fi
: "${MODEL:?set MODEL to the local model path}"
: "${TRAIN_DATA:?set TRAIN_DATA to the training data}"
: "${EVAL_DATA:?set EVAL_DATA to the evaluation data}"
experiment_root="${1:-runs/minimal-chain-$(date -u +%Y%m%d-%H%M%S)}"
if (($#)); then shift; fi
methods=("$@")
if ((${#methods[@]} == 0)); then methods=(full_pg grace uniform_cv grpo); fi
comparison_mode="${COMPARISON_MODE:-steps}"
post_train_stages="${POST_TRAIN_STAGES:-eval audit batch-audit}"
if [[ "$comparison_mode" != steps && "$comparison_mode" != wall ]]; then
  echo "COMPARISON_MODE must be steps or wall" >&2; exit 2
fi
if [[ "$comparison_mode" == wall && -z "${RUN_WALL_SECONDS:-}" && "${methods[0]}" != full_pg ]]; then
  echo "Wall comparison needs Full-PG first, or an explicit RUN_WALL_SECONDS reference" >&2; exit 2
fi
# Never mix a repeated invocation with a previous chain.
if [[ -e "$experiment_root" && ( ! -d "$experiment_root" || -n "$(find "$experiment_root" -mindepth 1 -maxdepth 1 -print -quit)" ) ]]; then
  experiment_root="${experiment_root}-$(date -u +%Y%m%d-%H%M%S)-$$"
fi
mkdir -p "$experiment_root/logs"
exec > >(tee -a "$experiment_root/console.log") 2>&1
echo "experiment_root $experiment_root"
trap 'status=$?; echo "experiment exit=$status at $(date -u +%FT%TZ) root=$experiment_root"' EXIT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
git log -1 --oneline
python -m pytest tests | tee "$experiment_root/logs/tests.log"
read -r -a seeds <<< "${SEEDS:-17 23 41}"
python - "$experiment_root" "${SEEDS:-17 23 41}" "$comparison_mode" "${methods[@]}" <<'PY'
import json, sys
from pathlib import Path
root, seeds, mode, *methods = sys.argv[1:]
(Path(root) / "experiment.json").write_text(json.dumps({"requested_seeds": seeds.split(), "requested_methods": methods,
    "comparison_mode": mode, "budget_scope": "training setup and complete batch boundaries, including persistence; shared SFT/eval/audits separate"}, indent=2))
PY
record_stage() {
  python - "$seed_root" "$1" "$2" "$3" <<'PY'
import json, sys
from pathlib import Path
from grace_gc.logging_util.run_dir import atomic_write_text
root, method, stage, log = sys.argv[1:]
paths = [line[len("run_dir "):].strip() for line in Path(log).read_text().splitlines() if line.startswith("run_dir ")]
if not paths:
    raise RuntimeError(f"no actual run_dir in {log}")
path = Path(root) / "stages.json"
manifest = json.loads(path.read_text()) if path.exists() else {}
manifest.setdefault(method, {})[stage] = Path(paths[-1]).resolve().relative_to(Path(root).resolve()).as_posix()
atomic_write_text(path, json.dumps(manifest, indent=2))
print(paths[-1])
PY
}
for seed in "${seeds[@]}"; do
  seed_root="$experiment_root/seed-$seed"
  mkdir -p "$seed_root/logs"
  common=(--config configs/default.yaml --config configs/experiments/minimal_gpu.yaml
    --config configs/experiments/minimal_gpu_repaired.yaml
    --config "${EXPERIMENT_CONFIG:-configs/experiments/minimal_gpu_deeper.yaml}"
    --config configs/hardware/a100_1.yaml --backend gpu_verl --model-path "$MODEL" --seed "$seed")
  if [[ -n "${COMMON_CONFIG:-}" ]]; then common+=(--config "$COMMON_CONFIG"); fi
  if [[ -n "${ABLATION_CONFIG:-}" ]]; then common+=(--config "$ABLATION_CONFIG"); fi
  init_args=()
  if [[ -n "${SHARED_INIT_CHAIN:-}" ]]; then
    shared_checkpoint=$(python - "$SHARED_INIT_CHAIN/seed-$seed" <<'PY'
import json, sys
from pathlib import Path
from scripts.summarize_minimal import stage_path
root = Path(sys.argv[1])
stages = json.loads((root/"stages.json").read_text())
print(stage_path(root, stages["shared"]["init"])/"checkpoints/step_0.npz")
PY
)
    init_args+=(--init-checkpoint "$shared_checkpoint")
  fi
  # Full-PG entry performs the shared reasoning-lead SFT once, with zero RL steps.
  python scripts/train.py "${common[@]}" --method full_pg --data-path "$TRAIN_DATA" \
    --config configs/experiments/minimal_gpu_train_memory.yaml \
    --num-steps 0 "${init_args[@]}" --run-dir "$seed_root/shared-init" | tee "$seed_root/logs/shared-init.stdout"
  init_run=$(record_stage shared init "$seed_root/logs/shared-init.stdout")
  wall_budget="${RUN_WALL_SECONDS:-}"
  target_step=""
  for method in "${methods[@]}"; do
    budget_args=()
    steps="${TRAIN_STEPS:-40}"
    if [[ "$comparison_mode" == wall && -n "$wall_budget" ]]; then
      budget_args+=(--run-wall-seconds "$wall_budget")
      steps="${MAX_STEPS:-10000}"
    fi
    if [[ -n "$target_step" ]]; then budget_args+=(--target-step-seconds "$target_step"); fi
    python scripts/train.py "${common[@]}" --method "$method" --data-path "$TRAIN_DATA" \
      --config configs/experiments/minimal_gpu_train_memory.yaml \
      --init-checkpoint "$init_run/checkpoints/step_0.npz" \
      --num-steps "$steps" "${budget_args[@]}" --run-dir "$seed_root/$method/train" | tee "$seed_root/logs/$method-train.stdout"
    train_run=$(record_stage "$method" train "$seed_root/logs/$method-train.stdout")
    if [[ "$comparison_mode" == wall && "$method" == full_pg ]]; then
      read -r reference_wall target_step < <(python - "$train_run/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
c = s["cost_control"]
print(c["global_wall_seconds"], c.get("target_step_seconds") or "")
PY
)
      if [[ -z "$wall_budget" ]]; then wall_budget="$reference_wall"; fi
      python - "$seed_root" "$wall_budget" "$target_step" "$train_run" <<'PY'
import json, sys
from pathlib import Path
root, wall, step, source = sys.argv[1:]
(Path(root)/"wall_budget.json").write_text(json.dumps({"run_wall_seconds":float(wall),
    "target_step_seconds":float(step) if step else None,"reference_method":"full_pg","reference_run":source,
    "note":"same requested wall budget; actual batch-boundary overshoot is reported, not hidden"},indent=2))
PY
    fi
    final_step=$(python - "$train_run/summary.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["step"])
PY
)
    if [[ " $post_train_stages " == *" eval "* ]]; then
    for step in 0 20 40; do
      if [[ ! -f "$train_run/checkpoints/step_$step.npz" ]]; then continue; fi
      python scripts/evaluate.py --generate "${common[@]}" --method "$method" --data-path "$EVAL_DATA" \
        --checkpoint "$train_run/checkpoints/step_$step.npz" \
        --run-dir "$seed_root/$method/eval-$step" | tee "$seed_root/logs/$method-eval-$step.stdout"
      record_stage "$method" "eval-$step" "$seed_root/logs/$method-eval-$step.stdout"
      if [[ "$step" == "$final_step" ]]; then
        record_stage "$method" eval-final "$seed_root/logs/$method-eval-$step.stdout"
      fi
    done
    if [[ "$final_step" != 0 && "$final_step" != 20 && "$final_step" != 40 ]]; then
      python scripts/evaluate.py --generate "${common[@]}" --method "$method" --data-path "$EVAL_DATA" \
        --checkpoint "$train_run/checkpoint.npz" --run-dir "$seed_root/$method/eval-final" \
        | tee "$seed_root/logs/$method-eval-final.stdout"
      record_stage "$method" eval-final "$seed_root/logs/$method-eval-final.stdout"
    fi
    fi
    if [[ " $post_train_stages " == *" audit "* ]]; then
    for stage in audit audit-training; do
      audit_extra=()
      if [[ "$stage" == audit-training ]]; then audit_extra=(--config configs/experiments/minimal_gpu_audit_training.yaml); fi
      python scripts/audit.py --generate "${common[@]}" --method "$method" --data-path "$TRAIN_DATA" \
        --config configs/experiments/minimal_gpu_audit.yaml "${audit_extra[@]}" \
        --checkpoint "$train_run/checkpoint.npz" --run-dir "$seed_root/$method/$stage" \
        | tee "$seed_root/logs/$method-$stage.stdout"
      record_stage "$method" "$stage" "$seed_root/logs/$method-$stage.stdout"
    done
    fi
    if [[ " $post_train_stages " == *" batch-audit "* ]]; then
    if [[ "$method" != grpo && "$method" != grpo_short ]]; then
      python scripts/audit_batch.py --generate "${common[@]}" --method "$method" --data-path "$TRAIN_DATA" \
        --config configs/experiments/minimal_gpu_train_memory.yaml \
        --checkpoint "$train_run/checkpoint.npz" --run-dir "$seed_root/$method/batch-audit" \
        | tee "$seed_root/logs/$method-batch-audit.stdout"
      record_stage "$method" batch-audit "$seed_root/logs/$method-batch-audit.stdout"
    else
      python - "$seed_root/$method/batch-audit" "$train_run/checkpoint.npz" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]); path.mkdir(parents=True, exist_ok=True)
(path/"batch_audit_summary.json").write_text(json.dumps({"status":"not_applicable",
    "reason":"GRPO has a different objective; this audit estimates R-minus-b HT/CV", "checkpoint":sys.argv[2]},indent=2))
PY
    fi
    fi
    python scripts/summarize_minimal.py "$seed_root"
  done
  python scripts/summarize_minimal.py "$seed_root"
done
python scripts/summarize_seeds.py "$experiment_root"
