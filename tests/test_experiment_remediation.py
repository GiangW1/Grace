import hashlib
import json
import tarfile

import pytest

from scripts.archive_run import archive_run
from scripts.summarize_seeds import summarize


def test_archive_inventory_covers_excluded_states(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    (root / 'health.json').write_text('{}')
    (root / 'archive_manifest.json').write_text('{"stale": true}')
    raw = b'checkpoint' * 100
    (root / 'checkpoint.npz').write_bytes(raw)
    output = tmp_path / 'run.tar.gz'
    manifest = archive_run(root, output, max_file_mib=.0001)
    by_name = {row['path']: row for row in manifest['files']}
    assert by_name['health.json']['included']
    assert not by_name['checkpoint.npz']['included']
    assert by_name['checkpoint.npz']['sha256'] == hashlib.sha256(raw).hexdigest()
    with tarfile.open(output) as archive:
        saved = json.load(archive.extractfile('run/archive_manifest.json'))
        assert archive.getnames().count('run/archive_manifest.json') == 1
        assert hashlib.sha256(archive.extractfile('run/health.json').read()).hexdigest() == by_name['health.json']['sha256']
    assert saved == manifest
    assert 'archive_manifest.json' not in by_name
    included = archive_run(root, output, max_file_mib=.0001, include_checkpoints=True)
    assert all(row['included'] for row in included['files'])


def test_seed_summary_uses_seed_units_and_retains_missing(tmp_path):
    (tmp_path / 'experiment.json').write_text(json.dumps({'requested_seeds': ['17', '23', '41']}))
    for seed, grace, full in [(17, .6, .5), (23, .4, .5)]:
        folder = tmp_path / f'seed-{seed}'
        folder.mkdir()
        manifest = {'ordered_records_sha256': 'questions', 'reward_protocol_version': 'test',
                    'samples_per_problem': 4, 'temperature': .6, 'top_p': .95,
                    'max_new_tokens': 4096, 'sample_batch_size': 4, 'sample_seed_start': seed}
        metadata = {'actual_seed': seed, 'shared_initial_actor_hash': f'actor-{seed}', 'evaluation_manifest': manifest}
        rows = [{'method': 'grace', 'final_avg4': grace, **metadata},
                {'method': 'full_pg', 'final_avg4': full, **metadata}]
        (folder / 'comparison.json').write_text(json.dumps({'methods': rows}))
    result = summarize(tmp_path)
    comparison = result['paired'][0]
    assert comparison['mean_delta'] == pytest.approx(0)
    assert len(comparison['pairs']) == 2
    assert result['missing_seed_reports'] == ['41']
    assert result['unit'] == 'training seed'


def test_seed_summary_does_not_pair_unknown_sources(tmp_path):
    folder = tmp_path / 'seed-17'
    folder.mkdir()
    (folder / 'comparison.json').write_text(json.dumps({'methods': [
        {'method': 'grace', 'final_avg4': .8}, {'method': 'full_pg', 'final_avg4': .5}]}))
    result = summarize(tmp_path)
    assert result['paired'][0]['mean_delta'] is None
    assert result['paired'][0]['incomparable']
    assert result['methods'][0]['per_seed'][0]['final_avg4'] == .5


def test_revised_shared_actor_four_method_cpu_chain(tmp_path, monkeypatch):
    """Exercise the new experiment plumbing; all data/models are synthetic."""
    import numpy as np
    from grace_gc.config import default_config, merge_configs
    from grace_gc.trainer.loop import run_training
    from grace_gc.trainer.checkpoint import load_checkpoint
    import grace_gc.trainer.loop as loop

    monkeypatch.setattr(loop, 'collect_environment', lambda *a, **k: {'missing': []})
    cfg = merge_configs(default_config(), {
        'backend': 'cpu_tiny', 'method': 'full_pg', 'num_steps': 0,
        'save_initial_checkpoint': True, 'checkpoint_every': 2,
        'decision_tokens': 2, 'max_new_tokens': 5, 'prompt_max_tokens': 8,
        'n_start': 4, 'n_prompts': 2, 'format_warmup': {'steps': 1, 'batch_size': 2},
        'baseline': {'prescan': 1, 'batch_prescan': True},
        'predictor': {'k': 2, 'warmup_steps': 2, 'refresh_every': 4, 'refresh_after_warmup': True,
                      'audit_s': 1., 'reservoir_size': 32, 'coord_kind': 'ridge',
                      'feature_scaler': 'fixed', 'shrink_calibration': 'holdout',
                      'ridge_weight_normalization': 'mean', 'basis_fit_only': True,
                      'basis_ipw': True, 'epochs': 1},
        'optim': {'log_update_geometry': True},
    })
    shared = run_training(cfg, tmp_path / 'shared')
    source = str(tmp_path / 'shared/checkpoints/step_0.npz')
    initial = load_checkpoint(source)
    hashes = set()
    for method in ('full_pg', 'grace', 'uniform_cv', 'grpo'):
        folder = tmp_path / method
        run = run_training(merge_configs(cfg, {'method': method, 'num_steps': 4,
                                               'init_checkpoint': source}), folder)
        assert shared['run_status'] == run['run_status'] == 'complete'
        step0 = load_checkpoint(folder / 'checkpoints/step_0.npz')
        for name, values in initial['actor_full'].items():
            np.testing.assert_array_equal(step0['actor_full'][name], values)
        assert not (folder / 'format_warmup.json').exists()
        hashes.add(json.loads((folder / 'initial_actor.json').read_text())['actor_sha256'])
        assert load_checkpoint(folder / 'checkpoint.npz')['step'] == 4
        rows = [json.loads(line) for line in (folder / 'trajectories.jsonl').read_text().splitlines()]
        assert all(row['reward'] is None for row in rows if row['z_continue'] == 0)
        assert all(row['snapshot_sha'] for row in rows)
    assert len(hashes) == 1
