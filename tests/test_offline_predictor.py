import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from grace_gc.config import merge_configs
from grace_gc.trainer import loop
from grace_gc.trainer.checkpoint import load_checkpoint
from grace_gc.versions import sha256_mapping
from tests.test_fixed_basis_supervision import _training_config


@pytest.fixture
def donor(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, 'collect_environment', lambda *a, **k: {'missing': []})
    cfg = _training_config()
    cfg['save_initial_checkpoint'] = True
    path = tmp_path/'donor'
    loop.run_training(cfg, path)
    return cfg, path


def test_offline_freezes_all_learning_and_resumes_without_source(donor, tmp_path, monkeypatch):
    from grace_gc.predictor.offline import export_predictor
    from grace_gc.trainer import algorithm, supervision
    cfg, source = donor
    artifact = export_predictor(source/'checkpoint.npz', tmp_path/'artifact')
    before = load_checkpoint(source/'checkpoint.npz')

    def unexpected(*a, **k):
        raise AssertionError('offline predictor must not collect or fit supervision')
    monkeypatch.setattr(algorithm, 'update_predictor_from_reservoir', unexpected)
    monkeypatch.setattr(supervision, 'collect_fresh_supervision', unexpected)
    offline = merge_configs(cfg, {'num_steps': 3, 'offline_predictor': str(artifact),
        'init_checkpoint': str(source/'checkpoints'/'step_0.npz'),
        'predictor': {'fresh_samples_per_problem': 2, 'audit_s': 1.0, 'warmup_steps': 20}})
    run = tmp_path/'offline'
    loop.run_training(offline, run)
    final = load_checkpoint(run/'checkpoint.npz')
    assert final['reservoir']['items'] == []
    np.testing.assert_array_equal(final['basis']['u'], before['basis']['u'])
    for field in ('coord', 'ridge_w', 'risk', 'cost', 'scaler', 'm_shrink'):
        assert sha256_mapping(final['predictor'][field]) == sha256_mapping(before['predictor'][field])
    rows = [json.loads(x) for x in (run/'steps.jsonl').read_text().splitlines()]
    assert all(x['n_audited'] == 0 and x['predictor_frozen'] for x in rows)
    assert all(x['predictor']['fresh_supervision']['n'] == 0 for x in rows)
    with np.load(run/'checkpoint.npz', allow_pickle=True) as archive:
        raw = archive['payload'].item()
    assert raw['predictor'] is None and 'u' not in raw['basis']
    assert (run/raw['offline_predictor']['path']).is_file()
    # Move the complete consumer run. Resume resolves its local dependency.
    moved = tmp_path/'moved'
    shutil.copytree(run, moved)
    artifact.unlink()
    resumed = merge_configs(offline, {'resume': str(moved/'checkpoints'/'step_1.npz'),
                                     'init_checkpoint': None, 'num_steps': 2})
    loop.run_training(resumed, tmp_path/'resumed')
    actual = load_checkpoint(tmp_path/'resumed'/'checkpoint.npz')
    for field in ('actor', 'predictor', 'rng', 'reservoir'):
        assert sha256_mapping(actual[field]) == sha256_mapping(final[field])


def test_offline_p1_matches_full_pg(donor, tmp_path):
    from grace_gc.predictor.offline import export_predictor
    cfg, source = donor
    artifact = export_predictor(source/'checkpoint.npz', tmp_path/'artifact')
    hashes = []
    for method in ('full_pg', 'grace'):
        run = tmp_path/method
        options = {'method': method, 'num_steps': 3, 'allocation': {'beta': 1.0},
                   'init_checkpoint': str(source/'checkpoints'/'step_0.npz')}
        if method == 'grace': options['offline_predictor'] = str(artifact)
        loop.run_training(merge_configs(cfg, options), run)
        hashes.append(sha256_mapping(load_checkpoint(run/'checkpoint.npz')['actor']))
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize('options,field', [({'predictor':{'feature_mode':'decision'}}, 'feature_mode'),
    ({'method':'prompt_cv'}, 'method_features'), ({'method':'reward_cv'}, 'risk_mode')])
def test_offline_rejects_wrong_feature_protocol(donor, tmp_path, options, field):
    from grace_gc.predictor.offline import export_predictor
    cfg, source = donor
    artifact = export_predictor(source/'checkpoint.npz', tmp_path/'artifact')
    with pytest.raises(ValueError, match=field):
        loop.run_training(merge_configs(cfg, {'offline_predictor':str(artifact), **options}), tmp_path/'wrong')


def test_fit_uses_only_calibration_and_deployment_beta(donor, tmp_path):
    from scripts.train_offline_predictor import fit_offline
    from grace_gc.predictor.offline import read_predictor
    cfg, source = donor
    data = tmp_path/'data.jsonl'
    rows = [{'problem_id': str(i), 'prompt': [{'role':'user', 'content':f'Compute {i}+1'}],
             'answer':str(i+1), 'split':'calib' if i < 24 else 'train'} for i in range(30)]
    data.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    fit_cfg = merge_configs(cfg, {'data_path':str(data), 'num_steps':4,
        'hardware':{'n_gpu':4},  # A CPU override must never invent GPU usage.
        'init_checkpoint':str(source/'checkpoints/step_0.npz'), 'allocation':{'beta':.6}})
    result = fit_offline(fit_cfg, tmp_path/'fit-source')
    artifact = read_predictor(result['artifact'])
    assert result['actor_unchanged'] and result['calibration_problems'] == 24
    assert result['gpu_reserved_seconds'] == 0  # CPU fixture, never GPU evidence.
    exported = [json.loads(line) for line in (tmp_path/'fit-source/calibration.jsonl').read_text().splitlines()]
    assert {row['problem_id'] for row in exported} == {str(i) for i in range(24)}
    assert all(isinstance(row['prompt'], list) for row in exported)
    steps = [json.loads(line) for line in (tmp_path/'fit-source/fit/steps.jsonl').read_text().splitlines()]
    assert all(row['n_continued'] == row['n'] for row in steps)
    assert artifact['source']['run_config']['allocation']['beta'] == .6
    assert any(row['predictor'].get('calibration_p_mean') is not None
               and row['predictor']['calibration_p_mean'] < 1 for row in steps)
    assert not any(key.startswith('opt_') for key in artifact['predictor'])
    loop.run_training(merge_configs(cfg, {'offline_predictor':result['artifact'],
        'init_checkpoint':str(source/'checkpoints/step_0.npz'), 'num_steps':1}), tmp_path/'consumer')


def test_corrupt_artifact_and_archive_dependency(donor, tmp_path):
    from grace_gc.predictor.offline import export_predictor, read_predictor
    from scripts.archive_run import archive_run
    import tarfile
    cfg, source = donor
    artifact = export_predictor(source/'checkpoint.npz', tmp_path/'source')
    run = tmp_path/'consumer'
    loop.run_training(merge_configs(cfg, {'offline_predictor':str(artifact), 'num_steps':1}), run)
    archive = tmp_path/'run.tar.gz'
    archive_run(run, archive, max_file_mib=0, include_paths=[Path('checkpoint.npz')])
    with tarfile.open(archive) as tar:
        assert f'consumer/{artifact.name}' in tar.getnames()
    body = read_predictor(artifact)
    body['predictor']['m_shrink'] += .1
    np.savez(artifact, payload=body)
    with pytest.raises(ValueError, match='hash'):
        read_predictor(artifact)


def test_artifact_mapping_hash_uses_buffer_without_changing_digest(monkeypatch):
    import hashlib
    from grace_gc import versions
    u = np.arange(24, dtype=np.float64).reshape(6,4)[:,::2]
    expected = hashlib.sha256(b'u'+np.ascontiguousarray(u).tobytes()).hexdigest()
    real = hashlib.sha256
    class Digest:
        def __init__(self): self.inner = real()
        def update(self, value):
            if not isinstance(value, bytes) or value != b'u':
                assert isinstance(value, np.ndarray) and value.flags.c_contiguous
            self.inner.update(value)
        def hexdigest(self): return self.inner.hexdigest()
    monkeypatch.setattr(versions.hashlib, 'sha256', Digest)
    assert versions.sha256_mapping({'u':u}) == expected
