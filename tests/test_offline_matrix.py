import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_offline_comparison_gpu as matrix


@pytest.fixture
def completed_matrix(tmp_path):
    from tests.test_cost_quality import fixture
    (tmp_path/'matrix.json').write_text(json.dumps({'seeds':[17,23], 'devices':['0','1','2','3'],
                                                  'consumer_wall_seconds':30., 'status':'complete'}))
    for seed, offset in [(17,0.), (23,.02)]:
        for label in ('one_gpu','multi_gpu'):
            chain = tmp_path/f'seed-{seed}'/label
            # These are synthetic quality/cost records, not GPU measurements.
            train, _ = fixture(chain)
            seed_root = train.parent.parent
            if seed != 17:
                seed_root = seed_root.rename(chain/f'seed-{seed}')
            shutil.copytree(seed_root/'grace', seed_root/'full_pg')
            stages, comparisons = {}, []
            for method in ('full_pg','grace'):
                folder = seed_root/method
                for step in (0,10,20):
                    path = folder/f'eval-{step}/eval_summary.json'
                    row = json.loads(path.read_text())
                    row['checkpoint'] = str(folder/f'train/checkpoints/step_{step}.npz')
                    row['avg'] = row['avg']+offset if method == 'grace' else .6
                    row['pass_at_k'] = row['avg']+.05
                    path.write_text(json.dumps(row))
                stages[method] = {'train':f'{method}/train',
                    **{f'eval-{s}':f'{method}/eval-{s}' for s in (0,10,20)},
                    'eval-final':f'{method}/eval-20'}
                comparisons.append({'method':method, 'actual_seed':seed,
                    'shared_initial_actor_hash':f'actor-{seed}', 'evaluation_manifest':row['evaluation_manifest'],
                    'final_avg4':row['avg'], 'final_pass4':row['pass_at_k'], 'final_training_step':20})
            (seed_root/'stages.json').write_text(json.dumps(stages))
            (seed_root/'comparison.json').write_text(json.dumps({'methods':comparisons}))
            commands = [{'stage':f'seed-{seed}/{method}/train', 'wall_seconds':50., 'exit_code':0}
                        for method in ('full_pg','grace')]
            (chain/'command_timing.jsonl').write_text('\n'.join(map(json.dumps, commands)))
    return tmp_path


@pytest.mark.parametrize('budget,expected_avg,expected_delta,step', [
    (30., .56, -.04, 10), (None, .91, .31, 20)])
def test_matrix_quality_uses_budget_endpoint_or_fixed_step_final(completed_matrix, budget, expected_avg, expected_delta, step):
    path = completed_matrix/'matrix.json'
    manifest = json.loads(path.read_text())
    manifest['consumer_wall_seconds'] = budget
    path.write_text(json.dumps(manifest))
    report = matrix.summarize_matrix(completed_matrix)
    for label in ('one_gpu','multi_gpu'):
        grace = next(r for r in report['aggregate'] if r.get('method') == 'grace' and r['hardware'] == label)
        assert grace['avg4_mean'] == pytest.approx(expected_avg)
        assert grace['pass4_mean'] == pytest.approx(expected_avg+.05)
        assert grace['avg4_seed_sd'] == pytest.approx(2**.5*.01)
        contrast = next(r for r in report['aggregate'] if r.get('contrast') and r['hardware'] == label)
        assert contrast['mean_delta'] == pytest.approx(expected_delta)
        assert contrast['n_seeds'] == 2
        assert sum(contrast['seed_ci95'])/2 == pytest.approx(expected_delta)
    for row in report['observations']:
        assert row['selected_evaluation']['step'] == step
        assert row['train_process_wall_seconds'] == 50.
        if row['method'] == 'grace':
            assert row['comparison']['final_avg4'] >= .9  # Keep the actual overshoot endpoint.
    assert all(not row['issues'] for row in report['pairs'])


@pytest.mark.parametrize('missing', ['budget_too_short','clock','hash','stages'])
def test_matrix_never_falls_back_to_final_when_budget_evidence_missing(completed_matrix, missing):
    for path in completed_matrix.glob('seed-*/*/seed-*/grace/train/checkpoints.json'):
        index = json.loads(path.read_text())
        for row in index['steps']:
            if missing == 'budget_too_short': row['available_command_wall_seconds'] = 31.
            if missing == 'clock': row.pop('available_command_wall_seconds')
            if missing == 'hash': row['sha256'] = 'wrong-checkpoint'
        path.write_text(json.dumps(index))
    if missing == 'stages':
        for path in completed_matrix.glob('seed-*/*/seed-*/stages.json'):
            stages = json.loads(path.read_text())
            stages.pop('grace')
            path.write_text(json.dumps(stages))
    report = matrix.summarize_matrix(completed_matrix)
    assert all(row['delta'] is None and row['issues'] for row in report['pairs'])
    grace = [row for row in report['aggregate'] if row.get('method') == 'grace']
    assert all(row['avg4_mean'] is None and row['n_evaluated_seeds'] == 0 for row in grace)


def test_matrix_budget_pair_uses_selected_evidence_when_final_is_unavailable(completed_matrix):
    for path in completed_matrix.glob('seed-*/*/seed-*/comparison.json'):
        comparison = json.loads(path.read_text())
        for row in comparison['methods']:
            row.update(final_avg4=None, final_pass4=None, evaluation_manifest={},
                       source_issues=[{'stage':'eval-final', 'reason':'checkpoint_source_unverified'}])
        path.write_text(json.dumps(comparison))
    for path in completed_matrix.glob('seed-*/*/seed-*/*/eval-20/eval_summary.json'):
        path.unlink()
    report = matrix.summarize_matrix(completed_matrix)
    assert all(not row['issues'] for row in report['pairs'])
    contrasts = [row for row in report['aggregate'] if row.get('contrast')]
    assert all(row['mean_delta'] == pytest.approx(-.04) for row in contrasts)


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
