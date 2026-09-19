#!/usr/bin/env python3
"""Fit on the existing calibration split with a fixed actor, then export inference state."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.config import merge_configs
from grace_gc.data.math_data import load_training_data
from grace_gc.logging_util.run_dir import RunDirectory, resolve_run_dir
from grace_gc.predictor.offline import export_predictor
from grace_gc.trainer.checkpoint import load_checkpoint
from grace_gc.trainer.loop import build_run_config, run_training
from grace_gc.versions import sha256_mapping, sha256_file


def fit_offline(cfg, directory):
    started = time.perf_counter()
    root = resolve_run_dir(directory, resume=False)
    root.mkdir(parents=True, exist_ok=True)
    RunDirectory(root).write_run_meta(kind='offline_fit', started=datetime.now(timezone.utc).isoformat())
    if not cfg.get('init_checkpoint'):
        raise ValueError('offline fitting needs the shared initial actor checkpoint')
    buckets, report = load_training_data(cfg['data_path'], cfg.get('eval_data_path'),
                                        seed=int(cfg.get('split_seed', 17)))
    records = buckets.get('calib', [])
    if not records:
        raise ValueError('calibration split is empty; supply data with a calibration split')
    data = root/'calibration.jsonl'
    data.write_text(''.join(json.dumps({**asdict(record), 'prompt':record.messages or record.prompt,
                                       'split':'train'}, ensure_ascii=False)+'\n'
                            for record in records), encoding='utf-8')
    before = load_checkpoint(cfg['init_checkpoint'])
    effective = merge_configs(cfg, {'method':'grace', 'data_path':str(data), 'resume':None,
        'offline_predictor':None, 'save_initial_checkpoint':True,
        'run_wall_seconds':None, 'session_wall_seconds':None,
        'format_warmup':{'steps':0}, 'optim':{'lr':0.0, 'weight_decay':0.0},
        'cost_control':{'mode':'fixed', 'enabled':False},
        # Warmup collects p=1 labels while calibration retains deployment beta.
        # The actor is fixed, so policy-age eviction/decay is inapplicable here.
        'predictor':{'fixed_basis':True, 'audit_s':1.0, 'fresh_samples_per_problem':0,
                     'warmup_steps':int(cfg['num_steps']), 'refresh_after_warmup':True,
                     'max_age_steps':None, 'age_half_life':None, 'shrink_calibration':'holdout'}})
    result = run_training(effective, root/'fit')
    checkpoint = Path(result['run_dir'])/'checkpoint.npz'
    after = load_checkpoint(checkpoint)
    for key in ('actor', 'actor_full'):
        if sha256_mapping(before.get(key)) != sha256_mapping(after.get(key)):
            raise RuntimeError('offline fitting changed the shared actor')
    artifact = export_predictor(checkpoint, root/'artifact')
    n_gpu = max(int((cfg.get('hardware') or {}).get('n_gpu', 1)), 1) if cfg.get('backend') == 'gpu_verl' else 0
    elapsed = time.perf_counter()-started
    summary = {'artifact':str(artifact.resolve()), 'artifact_sha256':sha256_file(artifact),
        'fit_run':result['run_dir'], 'calibration_problems':len(records),
        'source_data_sha256':sha256_file(cfg['data_path']), 'data_exclusion':report,
        'actor_unchanged':True, 'wall_seconds':elapsed, 'n_gpu_reserved':n_gpu,
        'gpu_reserved_seconds':elapsed*n_gpu,
        'scope':'calibration preparation, full fixed-actor fitting, export; process launch/shutdown and shared actor SFT are separate',
        'note':'Add this cost to offline consumers; reuse does not make pretraining free. No minimum signal or gamma target is imposed.'}
    (root/'offline_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',action='append',default=[])
    parser.add_argument('--init-checkpoint',required=True)
    parser.add_argument('--data-path',required=True)
    parser.add_argument('--eval-data-path')
    parser.add_argument('--model-path')
    parser.add_argument('--backend',default='gpu_verl')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--num-steps',type=int,default=40)
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args(argv)
    cfg=build_run_config(args.config,{key:value for key,value in vars(args).items()
                                    if key not in {'config','run_dir'} and value is not None})
    result=fit_offline(cfg,args.run_dir)
    print('offline_artifact',result['artifact'])
    return 0


if __name__=='__main__':
    raise SystemExit(main())
