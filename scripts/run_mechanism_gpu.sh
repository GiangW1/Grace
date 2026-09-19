#!/usr/bin/env bash
# Four independent training arms, shared actor/input seed/N; one frozen multi-arm audit.
set -eo pipefail
cd "$(dirname "$0")/.."
if [[ -f runs/setup/env.sh ]]; then source runs/setup/env.sh; fi
if [[ -f runs/setup/fetch-assets.exports ]]; then source runs/setup/fetch-assets.exports; fi
root="${1:-runs/mechanism-$(date -u +%Y%m%d-%H%M%S)}"
if [[ -e "$root" ]]; then root="$root-$(date -u +%Y%m%d-%H%M%S)-$$"; fi
mkdir -p "$root/logs" "$root/configs"
export COMPARISON_MODE=steps POST_TRAIN_STAGES=eval
unset RUN_WALL_SECONDS
# COMMON_CONFIG is appended after the default deeper recipe, before fixed-N/arm settings.
python - "$root" <<'PY'
import json, sys
from pathlib import Path
import yaml
from grace_gc.config import merge_configs
root = Path(sys.argv[1])
common = yaml.safe_load(Path("configs/experiments/mechanism_fixed.yaml").read_text())
arms = {"grace": {}, "p1": {"allocation": {"beta": 1.0}},
        "m0": {"predictor": {"control_variate": False}}, "uniform": {"allocation": {"uniform_shrink": 1.0}}}
for name, changes in arms.items():
    config = merge_configs(common, changes)
    config["experiment_variant"] = name
    (root/"configs"/f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
(root/"mechanism.json").write_text(json.dumps({"arms": {}, "design": "shared actor; fixed N=16, 4 prompts; same input seed and data; all arms keep predictor work",
    "scope": "Independent training trajectories diverge after interventions. Input/actor matching is checked from logs; identical decoded suffixes or equal runtime are not assumed."}, indent=2))
PY
source_chain="${SHARED_INIT_CHAIN:-}"
for variant in grace p1 m0 uniform; do
  export ABLATION_CONFIG="$root/configs/$variant.yaml"
  if [[ -n "$source_chain" ]]; then export SHARED_INIT_CHAIN="$source_chain"; fi
  bash scripts/run_minimal_gpu.sh "$root/$variant" grace | tee "$root/logs/$variant.stdout"
  actual=$(python - "$root/logs/$variant.stdout" <<'PY'
import sys
from pathlib import Path
print([line[len("experiment_root "):] for line in Path(sys.argv[1]).read_text().splitlines() if line.startswith("experiment_root ")][-1])
PY
)
  if [[ "$variant" == grace ]]; then source_chain="$actual"; fi
  python - "$root" "$variant" "$actual" <<'PY'
import json, sys
from pathlib import Path
root, name, actual = map(str, sys.argv[1:])
path = Path(root)/"mechanism.json"
data = json.loads(path.read_text()); data["arms"][name] = str(Path(actual).resolve())
path.write_text(json.dumps(data, indent=2))
PY
done
read -r -a seeds <<< "${SEEDS:-17 23 41}"
for seed in "${seeds[@]}"; do
  train_run=$(python - "$source_chain/seed-$seed" <<'PY'
import json, sys
from pathlib import Path
from scripts.summarize_minimal import stage_path
root = Path(sys.argv[1]); stages = json.loads((root/"stages.json").read_text())
print(stage_path(root, stages["grace"]["train"]))
PY
)
  python scripts/measure_command.py --log "$root/command_timing.jsonl" \
    --stage "seed-$seed/batch-audit" -- python scripts/audit_batch.py --generate --config "$train_run/config.yaml" \
    --checkpoint "$train_run/checkpoint.npz" \
    --batch-shape training --training-run "$train_run" --run-dir "$root/batch-audit-seed-$seed" \
    | tee "$root/logs/batch-audit-seed-$seed.stdout"
done
python scripts/summarize_mechanism.py "$root"
echo "mechanism_root $root"
