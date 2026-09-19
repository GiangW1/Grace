"""Compact fixed-basis labels preserve online supervision, not full gradients."""

import copy

import numpy as np
import pytest

from grace_gc.predictor.reservoir import GradientReservoir, ReservoirItem
from grace_gc.predictor.risk import full_space_residual


@pytest.mark.parametrize('rank_deficient', [False, True])
def test_compact_labels_preserve_projection_norm_and_full_residual(rank_deficient):
    rng = np.random.default_rng(5)
    u = rng.normal(size=(13, 3))
    if rank_deficient:
        u[:, 2] = u[:, 0]
    reservoir = GradientReservoir(8)
    gradients = [rng.normal(size=13), np.zeros(13)]
    for i, g in enumerate(gradients):
        reservoir.add(ReservoirItem(str(i), g, .4, .7, np.ones(2), 0., 3))
    reservoir.compact(u, 3)
    restored = GradientReservoir.from_state_dict(reservoir.state_dict())
    assert restored.fixed_basis_id == 3
    for item, g in zip(restored.items, gradients):
        assert item.g is None
        np.testing.assert_allclose(item.coordinates(u, 3), g @ u)
        assert item.gradient_norm_sq() == pytest.approx(g @ g)
        for f in (np.zeros(3), rng.normal(size=3)):
            assert item.residual(f, u, 3) == pytest.approx(float(full_space_residual(g, f, u)), abs=1e-12)
    # Clearing a reservoir reference must not clear this batch's actor label.
    assert gradients[0].shape == (13,)
    with pytest.raises(ValueError, match='basis'):
        restored.items[0].coordinates(u, 4)
    with pytest.raises(ValueError, match='basis'):
        restored.compact(u, 4)
    with pytest.raises(ValueError, match='compact'):
        restored.recent_matrix()
    restored.add(ReservoirItem('new', gradients[0], .5, 1., np.ones(2), 1., 3), u=u)
    assert restored.items[-1].g is None


def test_compact_predictor_update_matches_full_labels():
    torch = pytest.importorskip('torch')
    from grace_gc.predictor.heads import PredictorHeads
    from grace_gc.predictor.update import diagnose_frozen_predictor, update_predictor_from_reservoir

    rng = np.random.default_rng(23)
    u = rng.normal(size=(17, 2))
    full = GradientReservoir(20)
    for i in range(12):
        full.add(ReservoirItem(str(i), rng.normal(size=17) if i else np.zeros(17),
                              .3 + .05*i, .8, rng.normal(size=4), float(i % 2), 2,
                              cost_feat=np.ones(3), realized_cost=7., remaining_cost=8.,
                              prefix_finished=False, observed_step=i))
    compact = copy.deepcopy(full)
    compact.compact(u, 2)
    split = {key: np.arange(12) % 3 == side for side, key in enumerate(('fit', 'hold', 'calibration'))}
    split.update(basis=split['fit'], age_weight=np.linspace(.3, 1., 12))
    predictions, reports, diagnostics = [], [], []
    for reservoir in (full, compact):
        torch.manual_seed(31)
        heads = PredictorHeads(4, 2, hidden_risk=8, coord_kind='ridge',
                               feature_scaler='fixed', shrink_calibration='holdout')
        reports.append(update_predictor_from_reservoir(heads, reservoir, u, np.random.default_rng(9),
                       epochs=2, split=split, basis_id=2))
        predictions.append(heads.forward_numpy(np.stack([it.features for it in reservoir.items])))
        diagnostics.append(diagnose_frozen_predictor(heads, reservoir.items, u, only_unseen=False, basis_id=2))
    for field in ('f', 'r_hat', 'c_hat'):
        np.testing.assert_allclose(getattr(predictions[0], field), getattr(predictions[1], field), rtol=1e-6, atol=1e-8)
    assert reports[0]['m_shrink'] == pytest.approx(reports[1]['m_shrink'])
    assert reports[0]['supervision']['n_zero_gradient'] == reports[1]['supervision']['n_zero_gradient'] == 1
    assert diagnostics[0]['residual_to_m0_ratio'] == pytest.approx(diagnostics[1]['residual_to_m0_ratio'])


def test_float32_gradient_compacts_with_float64_norm():
    g = np.random.default_rng(13).normal(size=1000).astype(np.float32)
    u = g.astype(np.float64)[:, None]
    item = ReservoirItem('a', g, 1., 1., np.ones(2), 1., 1)
    item.compact(u, 1)
    assert item.residual(np.ones(1), u, 1) == pytest.approx(0., abs=1e-10)


def test_compact_state_rejects_missing_norm_and_wrong_basis():
    reservoir = GradientReservoir(2)
    reservoir.add(ReservoirItem('a', np.ones(3), 1., 1., np.ones(2), 1., 1))
    reservoir.compact(np.eye(3, 2), 1)
    raw = reservoir.state_dict()
    raw['items'][0]['g_norm_sq'] = None
    with pytest.raises(ValueError, match='compact'):
        GradientReservoir.from_state_dict(raw)


def _training_config():
    from grace_gc.config import default_config, merge_configs

    return merge_configs(default_config(), {
        'method': 'grace', 'num_steps': 5, 'n_start': 4, 'n_prompts': 2,
        'decision_tokens': 2, 'max_new_tokens': 5, 'prompt_max_tokens': 8,
        'format_warmup': {'steps': 0},
        'baseline': {'mode': 'fixed', 'fixed_value': .5, 'prescan': 0},
        'cost_control': {'mode': 'fixed'}, 'checkpoint_every': 1,
        'predictor': {'k': 2, 'warmup_steps': 2, 'refresh_every': 1,
                      'refresh_after_warmup': True, 'basis_variant': 'pca', 'basis_center': 'none',
                      'audit_s': 1., 'coord_kind': 'ridge', 'feature_scaler': 'fixed',
                      'shrink_calibration': 'holdout', 'reservoir_size': 32, 'epochs': 1,
                      'fixed_basis': True},
    })


def test_fixed_basis_training_online_heads_and_resume(tmp_path, monkeypatch):
    import json
    from grace_gc.config import merge_configs
    from grace_gc.trainer import loop
    from grace_gc.trainer.checkpoint import load_checkpoint
    from grace_gc.versions import sha256_mapping

    monkeypatch.setattr(loop, 'collect_environment', lambda *a, **k: {'missing': []})
    cfg = _training_config()
    run = tmp_path/'full'
    loop.run_training(cfg, run)
    snapshots = [load_checkpoint(run/'checkpoints'/f'step_{i}.npz') for i in range(2, 6)]
    assert len({p['basis']['basis_id'] for p in snapshots}) == 1
    assert snapshots[0]['basis']['basis_id'] > 0
    for payload in snapshots:
        assert payload['basis']['fixed']
        np.testing.assert_array_equal(payload['basis']['u'], snapshots[0]['basis']['u'])
        assert all(item['g'] is None for item in payload['reservoir']['items'])
    assert sha256_mapping(snapshots[0]['predictor']) != sha256_mapping(snapshots[-1]['predictor'])
    # The frozen U has one artifact per checkpoint directory, not one per step.
    assert len(list((run/'checkpoints').glob('basis-*.npy'))) == 1
    resumed = tmp_path/'resume'
    loop.run_training(merge_configs(cfg, {'num_steps': 3, 'resume': str(run/'checkpoints'/'step_2.npz')}), resumed)
    actual = load_checkpoint(resumed/'checkpoint.npz')
    for field in ('actor', 'actor_full', 'predictor', 'rng', 'reservoir'):
        assert sha256_mapping(actual[field]) == sha256_mapping(snapshots[-1][field]), field
    rows = [json.loads(line) for line in (run/'trajectories.jsonl').read_text().splitlines()]
    assert all(row['reward'] is None for row in rows if row['z_continue'] == 0)
    steps = [json.loads(line) for line in (run/'steps.jsonl').read_text().splitlines()]
    assert [row['n'] for row in steps] == [4]*5
    for row in steps:
        details = row['timing_details']['wall_seconds']
        assert set(details) == {'prefix_generate', 'prefix_features', 'predictor_forward', 'suffix_generate'}
        assert sum(details[key] for key in ('prefix_generate', 'prefix_features', 'predictor_forward')) <= row['phases']['prefix']
        assert details['suffix_generate'] <= row['phases']['continue']
    with pytest.raises(ValueError, match='fixed_basis'):
        loop.run_training(merge_configs(cfg, {'num_steps': 1, 'resume': str(run/'checkpoint.npz'),
                          'predictor': {'fixed_basis': False}}), tmp_path/'invalid')


def test_fixed_basis_p1_keeps_full_space_actor_update(tmp_path, monkeypatch):
    from grace_gc.config import merge_configs
    from grace_gc.trainer import loop
    from grace_gc.trainer.checkpoint import load_checkpoint
    from grace_gc.versions import sha256_mapping

    monkeypatch.setattr(loop, 'collect_environment', lambda *a, **k: {'missing': []})
    hashes = []
    for method in ('full_pg', 'grace'):
        run = tmp_path/method
        loop.run_training(merge_configs(_training_config(), {'method': method, 'allocation': {'beta': 1.}}), run)
        hashes.append(sha256_mapping(load_checkpoint(run/'checkpoint.npz')['actor']))
    assert hashes[0] == hashes[1]


def test_zero_labels_do_not_freeze_placeholder_basis(tmp_path, monkeypatch):
    from grace_gc.trainer import loop
    from grace_gc.trainer.checkpoint import load_checkpoint

    monkeypatch.setattr(loop, 'collect_environment', lambda *a, **k: {'missing': []})
    original = loop.make_tiny_engines

    def zero_rewards(*args, **kwargs):
        engine = original(*args, **kwargs)
        engine.reward_fn = lambda *a, **k: .5
        return engine

    monkeypatch.setattr(loop, 'make_tiny_engines', zero_rewards)
    run = tmp_path/'zero'
    loop.run_training(_training_config(), run)
    payload = load_checkpoint(run/'checkpoint.npz')
    assert payload['basis']['basis_id'] == 0
    assert not payload['basis']['fixed']
    assert payload['reservoir']['fixed_basis_id'] is None
    assert all(item['g'] is not None for item in payload['reservoir']['items'])
