"""Actual Bash/ledger/summarizers; only train/evaluate generate synthetic outputs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("selection, expected_steps", [(None, [0, 10, 20]), ("all", [0, 5, 10, 20])])
def test_budget_wrapper_evaluates_published_cutoff_and_deduplicates_final(tmp_path, selection, expected_steps):
    root = Path(__file__).resolve().parents[1]
    git = shutil.which("git")
    bash = Path(git).parents[1]/"bin/bash.exe" if os.name == "nt" and git else Path(shutil.which("bash") or "/missing")
    if not git or not bash.is_file():
        pytest.skip("Git/Bash unavailable")
    scripts = tmp_path/"scripts"; scripts.mkdir()
    for name in ("run_minimal_gpu.sh", "measure_command.py", "summarize_minimal.py", "summarize_seeds.py", "summarize_cost_quality.py"):
        shutil.copyfile(root/"scripts"/name, scripts/name)
    for relative in ("default.yaml", "experiments/minimal_gpu.yaml", "experiments/minimal_gpu_repaired.yaml",
                     "experiments/minimal_gpu_deeper.yaml", "experiments/minimal_gpu_train_memory.yaml"):
        path = tmp_path/"configs"/relative; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/"configs"/relative, path)
    hardware = tmp_path/"configs/hardware/custom.yaml"; hardware.parent.mkdir()
    hardware.write_text("hardware:\n  name: synthetic-5090\n  n_gpu: 1\n", encoding="utf-8")
    # The wrapper's real pytest preflight runs a small existing CPU test file.
    # This avoids recursively executing this shell test or mocking pytest itself.
    (tmp_path/"tests").mkdir()
    shutil.copyfile(root/"tests/test_command_timing.py", tmp_path/"tests/test_command_timing.py")
    child = r'''
import argparse, hashlib, json, os
from pathlib import Path
import yaml
from grace_gc.config import merge_configs
p=argparse.ArgumentParser()
p.add_argument('--config',action='append',default=[])
for name in ('run-dir','method','backend','model-path','data-path','eval-data-path','init-checkpoint','checkpoint','seed','num-steps','run-wall-seconds','target-step-seconds'):
    p.add_argument('--'+name)
a,extra=p.parse_known_args()
cfg={}
for path in a.config:
    cfg=merge_configs(cfg,yaml.safe_load(Path(path).read_text()))
assert cfg['hardware']['name']=='synthetic-5090'
assert os.environ.get('GRACE_COMMAND_START_MONOTONIC')
run=Path(a.run_dir); run.mkdir(parents=True)
cfg.update(seed=int(a.seed), method=a.method, backend=a.backend)
(run/'config.yaml').write_text(yaml.safe_dump(cfg))
(run/'synthetic_invocation.json').write_text(json.dumps(vars(a)))
if Path(__file__).name=='train.py':
    assert a.eval_data_path==os.environ['EVAL_DATA']
    init=int(a.num_steps)==0
    if not init:
        assert a.run_wall_seconds=='30' and int(a.num_steps)==10000
        assert Path(a.init_checkpoint).is_file()
    rows=[]
    for step,cost in ([(0,1)] if init else [(0,5),(5,15),(10,25),(20,45)]):
        path=run/f'checkpoints/step_{step}.npz'; path.parent.mkdir(exist_ok=True)
        raw=f'SYNTHETIC TEST ONLY snapshot {step}'.encode(); path.write_bytes(raw)
        rows.append({'step':step,'path':f'checkpoints/step_{step}.npz','sha256':hashlib.sha256(raw).hexdigest(),
                     'available_at':'2026-09-19T00:00:00+00:00','available_command_wall_seconds':cost,
                     'available_global_wall_seconds':cost-1})
    final=rows[-1]
    (run/'checkpoint.npz').write_bytes((run/final['path']).read_bytes())
    (run/'checkpoints.json').write_text(json.dumps({'latest':'checkpoint.npz','latest_sha256':final['sha256'],'steps':rows}))
    (run/'summary.json').write_text(json.dumps({'run_status':'complete','step':final['step'],
        'cost_control':{'global_wall_seconds':45,'target_step_seconds':2},'cumulative_wall_seconds':45}))
    (run/'compute_ledger.json').write_text(json.dumps({'hardware':'synthetic-5090','wall_seconds':45,'gpu_reserved_seconds':45}))
    (run/'initial_actor.json').write_text(json.dumps({'actor_sha256':'synthetic-common-actor'}))
    (run/'steps.jsonl').write_text(''.join(json.dumps({'step':i,'n':16,'warmup':True})+'\n' for i in range(1,final['step']+1)))
else:
    assert '--generate' in extra and a.data_path==os.environ['EVAL_DATA']
    ckpt=Path(a.checkpoint)
    step=int(ckpt.stem.removeprefix('step_')) if ckpt.stem.startswith('step_') else 20
    result={'checkpoint':str(ckpt.resolve()),'checkpoint_step':step,'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
      'avg':.5,'pass_at_k':1.,'requested_k':4,'n_problems':1,'per_problem':[{'problem_id':'synthetic','avg':.5}],
      'evaluation_manifest':{'ordered_records_sha256':'synthetic','reward_protocol_version':2,'samples_per_problem':4,
         'temperature':.6,'top_p':.95,'max_new_tokens':2048,'sample_batch_size':1,'sample_seed_start':int(a.seed)}}
    (run/'eval_summary.json').write_text(json.dumps(result))
print('run_dir',run)
'''
    for name in ("train.py", "evaluate.py"):
        (scripts/name).write_text(child, encoding="utf-8")
    bindir = tmp_path/"bin"; bindir.mkdir()
    shim = bindir/"python"
    shim.write_text(f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    env = {**os.environ, "PATH": str(bindir)+os.pathsep+os.environ.get("PATH", ""), "PYTHONPATH": str(root),
           "GIT_DIR": subprocess.check_output([git, "rev-parse", "--absolute-git-dir"], cwd=root, text=True).strip(),
           "SEEDS": "17", "MODEL": "synthetic-model", "TRAIN_DATA": "train.jsonl", "EVAL_DATA": "eval.jsonl",
           "HARDWARE_CONFIG": "configs/hardware/custom.yaml", "COMPARISON_MODE": "wall", "RUN_WALL_SECONDS": "30",
           "POST_TRAIN_STAGES": "eval", "MAX_STEPS": "10000"}
    for name in ("SHARED_INIT_CHAIN", "COMMON_CONFIG", "ABLATION_CONFIG", "EXPERIMENT_CONFIG", "EVAL_STEPS"):
        env.pop(name, None)
    if selection:
        env["EVAL_STEPS"] = selection
    completed = subprocess.run([str(bash), "scripts/run_minimal_gpu.sh", "result", "grace"], cwd=tmp_path,
        env=env, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = tmp_path/"result"
    stages = json.loads((result/"seed-17/stages.json").read_text())["grace"]
    assert sorted(int(name.removeprefix("eval-")) for name in stages if name.startswith("eval-") and name != "eval-final") == expected_steps
    assert stages["eval-final"] == stages["eval-20"]
    assert not (result/"seed-17/grace/eval-final").exists()
    timed = [json.loads(line) for line in (result/"command_timing.jsonl").read_text().splitlines()]
    assert [row["stage"] for row in timed] == ["seed-17/shared-init", "seed-17/grace/train", *[f"seed-17/grace/eval-{step}" for step in expected_steps]]
    assert all(row["exit_code"] == 0 and row["wall_seconds"] > 0 for row in timed)
    assert json.loads((result/"command_timing.summary.json").read_text())["n_commands"] == len(timed)
    budget = json.loads((result/"seed-17/wall_budget.json").read_text())
    assert budget["source"] == "explicit" and budget["reference_method"] is None
    assert budget["run_wall_seconds"] == 30
    curve = json.loads((result/"cost_quality.json").read_text())["seeds"]["seed-17"]["grace"]
    assert curve["at_budget"]["30.0"]["step"] == 10
    assert curve["hardware"] == "synthetic-5090"
