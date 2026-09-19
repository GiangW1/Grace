import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_offline_comparison_gpu as matrix


def test_duplicate_seed_is_not_counted_as_independent_evidence():
    with pytest.raises(ValueError, match='unique seeds'):
        matrix.run_matrix(SimpleNamespace(seeds=[17,17]))


def test_matrix_reuses_actor_and_artifact_and_accounts_cold_cost(tmp_path, monkeypatch):
    calls = []
    def measured(command, env, log, stage):
        calls.append((command, dict(env), stage))
        wall = 7 if stage.endswith('offline-fit') else 2
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with Path(log).open('a') as stream:
            stream.write(json.dumps({'stage':stage, 'wall_seconds':wall, 'exit_code':0})+'\n')
        if stage.endswith('offline-fit'):
            dest = Path(command[command.index('--run-dir')+1])
            dest.mkdir(parents=True)
            (dest/'offline_summary.json').write_text(json.dumps({'artifact':str(dest/'frozen.npz')}))
        if command[0] == 'bash':
            chain = Path(command[2])
            (chain/'seed-17').mkdir(parents=True)
            (chain/'seed-17/comparison.json').write_text(json.dumps({'methods':[
                {'method':method, 'final_avg4':score, 'final_pass4':.8}
                for method,score in [('full_pg',.6),('grace',.65)]]}))
            rows = [{'stage':f'seed-17/{stage}', 'wall_seconds':value, 'exit_code':0}
                    for stage,value in [('shared-init',3),('full_pg/train',5),('grace/train',5)]]
            (chain/'command_timing.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    monkeypatch.setattr(matrix, 'measured', measured)
    args = SimpleNamespace(devices='2,3,5,7', run_dir=str(tmp_path/'matrix'), seeds=[17], steps=40,
        fit_steps=32, wall_seconds=None, hardware_config='configs/hardware/a100_1.yaml',
        experiment_config='configs/experiments/offline_comparison.yaml',
        model='synthetic', train_data='synthetic-train', eval_data='synthetic-eval')
    root = matrix.run_matrix(args)
    chains = [item for item in calls if item[0][0] == 'bash']
    assert len(chains) == 2
    assert chains[0][1]['OFFLINE_PREDICTOR'] == chains[1][1]['OFFLINE_PREDICTOR']
    assert chains[0][1]['SHARED_INIT_CHAIN'] == chains[1][1]['SHARED_INIT_CHAIN']
    assert [item[1]['CUDA_VISIBLE_DEVICES'] for item in chains] == ['2','2,3,5,7']
    assert all(item[0][-2:] == ['full_pg','grace'] for item in chains)
    assert sum(item[2].endswith('offline-fit') for item in calls) == 1
    report = matrix.summarize_matrix(root)
    rows = {(row['hardware'],row['method']):row for row in report['observations']}
    assert rows['one_gpu','full_pg']['cold_total_wall_seconds'] == 10
    assert rows['one_gpu','grace']['cold_total_wall_seconds'] == 17
    assert rows['multi_gpu','grace']['cold_total_gpu_seconds'] == 41
    # Missing pairing evidence must not turn synthetic scores into valid claims.
    assert all(row['delta'] is None and row['issues'] for row in report['pairs'])
    (root/'commands.jsonl').unlink()
    assert all(row['cold_total_gpu_seconds'] is None for row in matrix.summarize_matrix(root)['observations'])


def test_offline_audit_does_not_apply_online_warmup():
    import numpy as np
    from grace_gc.audit.batch_audit import _intervention_allocations
    from grace_gc.trainer.methods import method_spec
    payload = {'step':0, 'offline_predictor':{'path':'artifact'},
               'basis':{'basis_id':2, 'predictor_synced_basis_id':2}}
    cfg = {'predictor':{'warmup_steps':20}, 'allocation':{'beta':.5, 'p_min':.2}}
    result = _intervention_allocations({'actual':{}}, np.ones((4,2)), np.ones(4), np.ones(4),
        np.zeros(4,dtype=bool), method_spec('grace'), payload, cfg, .5)
    assert np.allclose(result['actual']['p'], .5)
