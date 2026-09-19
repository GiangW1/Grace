#!/usr/bin/env python3
"""Four controls: Full-PG/frozen GRACE, each on one GPU and actor + rollout GPUs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.logging_util.run_dir import RunDirectory, resolve_run_dir
from scripts.summarize_minimal import read_json, read_rows, finite
from scripts.summarize_seeds import pairing_issues, seed_interval


def measured(command, env, log, stage):
    subprocess.run([sys.executable, 'scripts/measure_command.py', '--log', str(log),
                    '--stage', stage, '--', *command], cwd=ROOT, env=env, check=True)


def command_wall(log, stage):
    rows = [row for row in read_rows(Path(log)) if row.get('stage') == stage]
    if not rows or any(row.get('exit_code') != 0 for row in rows):
        return None
    return sum(row['wall_seconds'] for row in rows)


def summarize_matrix(root):
    root = Path(root)
    manifest = read_json(root/'matrix.json')
    observations, pairs = [], []
    for seed in manifest.get('seeds', []):
        name = f'seed-{seed}'
        prep = root/name
        fit = read_json(prep/'offline/offline_summary.json')
        fit_wall = command_wall(root/'commands.jsonl', f'{name}/offline-fit')
        init_wall = command_wall(root/'commands.jsonl', f'{name}/shared-init')
        for label, n_gpu in [('one_gpu', 1), ('multi_gpu', len(manifest['devices']))]:
            chain = prep/label
            rows = read_json(chain/name/'comparison.json').get('methods', [])
            methods = {row['method']: row for row in rows}
            for method in ('full_pg', 'grace'):
                row = methods.get(method, {})
                wall = command_wall(chain/'command_timing.jsonl', f'{name}/{method}/train')
                copy_wall = command_wall(chain/'command_timing.jsonl', f'{name}/shared-init')
                offline_wall = fit_wall if method == 'grace' else 0.0
                parts = (wall, copy_wall, init_wall, offline_wall)
                complete_cost = all(finite(value) for value in parts)
                observations.append({'seed':seed, 'hardware':label, 'method':method,
                    'n_gpu':n_gpu, 'comparison':row or None,
                    'train_process_wall_seconds':wall,
                    'train_process_gpu_seconds':None if wall is None else n_gpu*wall,
                    'shared_actor_load_wall_seconds':copy_wall,
                    'shared_sft_process_wall_seconds':init_wall,
                    'offline_fit_process_wall_seconds':offline_wall,
                    'offline_fit':fit if method == 'grace' else None,
                    'cold_total_wall_seconds':sum(parts) if complete_cost else None,
                    'cold_total_gpu_seconds':n_gpu*(wall+copy_wall)+init_wall+offline_wall if complete_cost else None,
                    'cost_scope':'shared SFT + shared actor load + RL subprocess + one full offline fit for each hypothetical standalone GRACE run; excludes eval/audits/tests; no amortization assumption'})
            a, b = methods.get('grace', {}), methods.get('full_pg', {})
            issues = pairing_issues(seed, a, b)
            if not all(finite(row.get('final_avg4')) for row in (a,b)):
                issues.append('final_evaluation_missing')
            pairs.append({'seed':seed, 'hardware':label, 'issues':issues,
                          'delta':a['final_avg4']-b['final_avg4'] if not issues else None})
    aggregate = []
    for label in ('one_gpu', 'multi_gpu'):
        deltas = [row['delta'] for row in pairs if row['hardware'] == label and not row['issues']]
        for method in ('full_pg', 'grace'):
            rows = [row['comparison'] or {} for row in observations if row['hardware'] == label and row['method'] == method]
            avg = [row['final_avg4'] for row in rows if finite(row.get('final_avg4'))]
            passes = [row['final_pass4'] for row in rows if finite(row.get('final_pass4'))]
            aggregate.append({'hardware':label, 'method':method, 'n_evaluated_seeds':len(avg),
                'avg4_mean':float(np.mean(avg)) if avg else None,
                'avg4_seed_sd':float(np.std(avg, ddof=1)) if len(avg)>1 else None,
                'pass4_mean':float(np.mean(passes)) if passes else None})
        aggregate.append({'hardware':label, 'contrast':'grace minus full_pg',
            'mean_delta':float(np.mean(deltas)) if deltas else None, **seed_interval(deltas)})
    result = {'manifest':manifest, 'observations':observations, 'pairs':pairs, 'aggregate':aggregate,
              'note':'Compare algorithms within the same hardware layout. Warm consumer budgets exclude offline fitting/shared SFT; cold totals add them and are not equal-budget endpoints. Seed intervals use paired training seeds. Missing data are unavailable, never zero.'}
    RunDirectory(root).write_json('matrix_summary.json', result)
    return result


def run_matrix(args):
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('repeated seeds are not independent replicates; supply unique seeds')
    devices = [value.strip() for value in args.devices.split(',') if value.strip()]
    if len(devices) < 2 or len(set(devices)) != len(devices) or '-1' in devices:
        raise ValueError('--devices needs distinct CUDA device IDs: one actor and at least one rollout worker')
    root = resolve_run_dir(args.run_dir).resolve()
    run = RunDirectory(root)
    run.write_run_meta(kind='offline_matrix', started=datetime.now(timezone.utc).isoformat())
    manifest = {'seeds':args.seeds, 'devices':devices, 'steps':args.steps, 'fit_steps':args.fit_steps,
                'consumer_wall_seconds':args.wall_seconds, 'status':'running',
                'variants':['one_gpu/full_pg','one_gpu/grace','multi_gpu/full_pg','multi_gpu/grace']}
    run.write_json('matrix.json', manifest)
    multi_config = root/'rollout_hardware.yaml'
    multi_config.write_text(yaml.safe_dump({'hardware':{'n_gpu':len(devices)},
        'rollout':{'workers':len(devices)-1}, 'vllm':{'tensor_parallel':1}}), encoding='utf-8')
    eval_config = root/'evaluation_hardware.yaml'
    eval_config.write_text(yaml.safe_dump({'hardware':{'n_gpu':1}, 'rollout':{'workers':0}}), encoding='utf-8')
    config_paths = ['configs/default.yaml', 'configs/experiments/minimal_gpu.yaml',
        'configs/experiments/minimal_gpu_repaired.yaml', args.experiment_config,
        args.hardware_config, 'configs/experiments/minimal_gpu_train_memory.yaml']
    common = [token for path in config_paths for token in ('--config', path)]
    common += ['--backend','gpu_verl','--model-path',args.model,'--data-path',args.train_data,
               '--eval-data-path',args.eval_data]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':devices[0]}
    log = root/'commands.jsonl'
    try:
        measured([sys.executable, '-m', 'pytest', 'tests'], env, log, 'cpu-regression')
        for seed in args.seeds:
            name = f'seed-{seed}'
            prep = root/name
            shared_root = prep/'shared'
            init = shared_root/name/'init'
            measured([sys.executable,'scripts/train.py',*common,'--seed',str(seed),
                '--method','full_pg','--num-steps','0','--run-dir',str(init)], env, log, f'{name}/shared-init')
            RunDirectory(shared_root/name).write_json('stages.json', {'shared':{'init':'init'}})
            measured([sys.executable,'scripts/train_offline_predictor.py',*common,'--seed',str(seed),
                '--num-steps',str(args.fit_steps),'--init-checkpoint',str(init/'checkpoints/step_0.npz'),
                '--run-dir',str(prep/'offline')], env, log, f'{name}/offline-fit')
            artifact = read_json(prep/'offline/offline_summary.json')['artifact']
            for label in ('one_gpu', 'multi_gpu'):
                # Both methods use precisely the same config layering as the donor.
                child_env = {**env, 'MODEL':args.model, 'TRAIN_DATA':args.train_data, 'EVAL_DATA':args.eval_data,
                    'SEEDS':str(seed), 'TRAIN_STEPS':str(args.steps), 'EXPERIMENT_CONFIG':args.experiment_config,
                    'HARDWARE_CONFIG':args.hardware_config, 'SHARED_INIT_CHAIN':str(shared_root),
                    'OFFLINE_PREDICTOR':artifact, 'RUN_TESTS':'0', 'COMMON_CONFIG':'',
                    'POST_HARDWARE_CONFIG':str(eval_config),
                    'ABLATION_CONFIG':str(multi_config) if label == 'multi_gpu' else '',
                    'CUDA_VISIBLE_DEVICES':','.join(devices) if label == 'multi_gpu' else devices[0],
                    'COMPARISON_MODE':'wall' if args.wall_seconds is not None else 'steps',
                    'RUN_WALL_SECONDS':str(args.wall_seconds) if args.wall_seconds is not None else ''}
                measured(['bash','scripts/run_minimal_gpu.sh',str(prep/label),'full_pg','grace'],
                         child_env, log, f'{name}/{label}')
        manifest['status'] = 'complete'
    except BaseException:
        manifest['status'] = 'failed'
        raise
    finally:
        run.write_json('matrix.json', manifest)
        summarize_matrix(root)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=os.environ.get('MODEL'), required=not os.environ.get('MODEL'))
    parser.add_argument('--train-data', default=os.environ.get('TRAIN_DATA'), required=not os.environ.get('TRAIN_DATA'))
    parser.add_argument('--eval-data', default=os.environ.get('EVAL_DATA'), required=not os.environ.get('EVAL_DATA'))
    parser.add_argument('--devices', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3'))
    parser.add_argument('--seeds', nargs='+', type=int, default=[17,23,41])
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--fit-steps', type=int, default=40)
    parser.add_argument('--wall-seconds', type=float, help='same consumer training budget; full offline cost is reported separately')
    parser.add_argument('--hardware-config', default='configs/hardware/a100_1.yaml')
    parser.add_argument('--experiment-config', default='configs/experiments/offline_comparison.yaml')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    print('matrix_root', run_matrix(args))


if __name__ == '__main__':
    main()
