"""Exercise the shell orchestrator with synthetic training commands, never a GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_wrapper_four_arms_shared_init_and_one_audit(tmp_path):
    root = Path(__file__).resolve().parents[1]
    git = shutil.which("git")
    bash = Path(git).parents[1]/"bin/bash.exe" if os.name == "nt" and git else Path(shutil.which("bash") or "/missing")
    if not bash.is_file():
        pytest.skip("Bash unavailable")
    scripts = tmp_path/"scripts"; scripts.mkdir()
    config = tmp_path/"configs/experiments"; config.mkdir(parents=True)
    for name in ("run_mechanism_gpu.sh", "summarize_mechanism.py", "measure_command.py"):
        shutil.copyfile(root/"scripts"/name, scripts/name)
    shutil.copyfile(root/"configs/experiments/mechanism_fixed.yaml", config/"mechanism_fixed.yaml")
    # The actual wrapper, config writer, manifest resolver and final summarizer run.
    # Only expensive child training/audit commands are replaced by this fixture.
    (scripts/"run_minimal_gpu.sh").write_text("#!/usr/bin/env bash\nset -e\npython scripts/fake_train.py \"$1\"\n", encoding="utf-8")
    (scripts/"fake_train.py").write_text('''
import json, os, sys
from pathlib import Path
import yaml
from grace_gc.config import merge_configs
root = Path(sys.argv[1]); seed = root/"seed-17"
train = seed/"grace/train"; train.mkdir(parents=True)
cfg = {"data_path": "inherited-training-data.jsonl", "baseline": {"prescan_prior_strength": 0}}
if os.environ.get("COMMON_CONFIG"):
    cfg = merge_configs(cfg, yaml.safe_load(Path(os.environ["COMMON_CONFIG"]).read_text()))
cfg = merge_configs(cfg, yaml.safe_load(Path(os.environ["ABLATION_CONFIG"]).read_text()))
assert os.environ["POST_TRAIN_STAGES"] == "eval"
assert cfg["cost_control"]["mode"] == "fixed" and cfg["n_start"] == 16
(train/"config.yaml").write_text(yaml.safe_dump(cfg))
(train/"initial_actor.json").write_text(json.dumps({"actor_sha256": "synthetic-common-actor"}))
(train/"summary.json").write_text(json.dumps({"step": 1}))
(train/"steps.jsonl").write_text(json.dumps({"step": 1, "n": 16})+"\\n")
(train/"trajectories.jsonl").write_text((json.dumps({"step": 1,"problem_id": "p","prompt_token_ids": [1,2]})+"\\n")*16)
(seed/"stages.json").write_text(json.dumps({"shared": {"init": "shared-init"}, "grace": {"train": "grace/train"}}))
(root/"test_parent_environment.json").write_text(json.dumps({"shared": os.environ.get("SHARED_INIT_CHAIN"), "common": os.environ.get("COMMON_CONFIG")}))
print("experiment_root", root)
''', encoding="utf-8")
    (scripts/"audit_batch.py").write_text('''
import argparse, json
from pathlib import Path
import yaml
p=argparse.ArgumentParser(); p.add_argument("--config"); p.add_argument("--checkpoint"); p.add_argument("--run-dir"); p.add_argument("--training-run"); p.add_argument("--batch-shape"); p.add_argument("--generate",action="store_true")
a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text())
assert cfg["data_path"] == "inherited-training-data.jsonl"
assert set(cfg["batch_audit"]["interventions"]) == {"p1","m0","uniform"}
assert a.batch_shape == "training"
root=Path(a.run_dir); root.mkdir(parents=True)
(root/"batch_audit_summary.json").write_text(json.dumps({"status":"synthetic_test_only","fixed_n":16}))
print("run_dir",root)
''', encoding="utf-8")
    bindir = tmp_path/"bin"; bindir.mkdir()
    shim = bindir/"python"
    shim.write_text(f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    candidate = tmp_path/"candidate.yaml"
    candidate.write_text("baseline:\n  prescan_prior_strength: 2\n", encoding="utf-8")
    env = {**os.environ, "PATH": str(bindir)+os.pathsep+os.environ.get("PATH", ""),
           "PYTHONPATH": str(root), "SEEDS": "17", "COMMON_CONFIG": str(candidate)}
    env.pop("SHARED_INIT_CHAIN", None); env.pop("TRAIN_DATA", None)
    subprocess.run([str(bash), "scripts/run_mechanism_gpu.sh", "result"], cwd=tmp_path, env=env,
                   check=True, capture_output=True, text=True, timeout=60)
    result = tmp_path/"result"
    manifest = json.loads((result/"mechanism.json").read_text())
    assert set(manifest["arms"]) == {"grace", "p1", "m0", "uniform"}
    assert len(list(result.glob("batch-audit-seed-*"))) == 1
    timed = [json.loads(line) for line in (result/"command_timing.jsonl").read_text().splitlines()]
    assert len(timed) == 1 and timed[0]["stage"] == "seed-17/batch-audit"
    assert timed[0]["exit_code"] == 0 and timed[0]["wall_seconds"] > 0
    for name in ("p1", "m0", "uniform"):
        parent = json.loads((result/name/"test_parent_environment.json").read_text())
        assert Path(parent["shared"]) == Path("result/grace")
    summary = json.loads((result/"mechanism_summary.json").read_text())
    assert all(summary["seeds"]["seed-17"]["pairing"].values())
    import yaml
    configs = {name: yaml.safe_load((result/name/"seed-17/grace/train/config.yaml").read_text()) for name in manifest["arms"]}
    assert all(c["baseline"]["prescan_prior_strength"] == 2 for c in configs.values())
    assert configs["p1"]["allocation"]["beta"] == 1
    assert configs["m0"]["predictor"]["control_variate"] is False
    assert configs["uniform"]["allocation"]["uniform_shrink"] == 1
