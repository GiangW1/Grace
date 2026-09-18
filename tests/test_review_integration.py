"""Selected review suggestions must reach the shared training path unchanged."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from grace_gc.core.layout import collect_lora_layout, pack_grads
from grace_gc.core.losses import prediction_grad_correction
from grace_gc.core.rng import IsolatedRNG
from grace_gc.predictor.heads import predictor_from_spec
from grace_gc.predictor.reservoir import GradientReservoir
from grace_gc.trainer.algorithm import TrainState, run_algorithm1_step
from grace_gc.trainer.baseline import baseline_from_config
from grace_gc.trainer.cpu_tiny import TinyLoRAActor
from grace_gc.trainer.methods import method_spec
from grace_gc.trainer.tiny_engine import make_tiny_engines


def run_batch(method='grace', *, baseline=None, cv=True, shrink=0., beta=.5, reward=None):
    torch.manual_seed(73)
    actor = TinyLoRAActor()
    engines = make_tiny_engines(actor, actor.vocab)
    if reward is not None:
        engines.reward_fn = lambda *a, **k: reward
    prompts = [[1, 2], [1, 2], [2, 3], [2, 3]]
    layout = collect_lora_layout(actor.named_lora_params())
    spec = method_spec(method)
    predictor = None
    if spec.use_predictor:
        dim = engines.prefix_features(prompts, np.ones(4, dtype=int), [.5]*4)['features'].shape[1]
        predictor = predictor_from_spec(dim, 2, {'coord_kind': 'ridge', 'hidden_risk': 4})
        predictor.constant_cost = False
        predictor.forward_numpy = lambda feats, costs=None: SimpleNamespace(
            f=np.full((len(feats), 2), .25), r_hat=np.arange(1, len(feats)+1, dtype=float)**2,
            c_hat=np.arange(1, len(feats)+1, dtype=float))
    bcfg = {'prescan': 0, **(baseline or {})}
    state = TrainState(spec, baseline_from_config(bcfg), IsolatedRNG.create(71), layout,
                       np.eye(layout.dim, 2), predictor, GradientReservoir(32),
                       basis_id=1, predictor_synced_basis_id=1)
    cfg = {'decision_tokens': 2, 'max_new_tokens': 5, 'n_prompts': 2,
           'baseline': bcfg, 'allocation': {'beta': beta, 'p_min': .2, 'uniform_shrink': shrink},
           'predictor': {'warmup_steps': 0, 'audit_s': 0., 'control_variate': cv},
           'optim': {'grad_clip': 0}}
    opt = torch.optim.SGD(actor.trainable_params(), lr=.01)
    out = run_algorithm1_step(engines, state, prompts, ['a', 'a', 'b', 'b'], ['', '', '', ''], cfg, opt)
    values = pack_grads([(name, value.detach().numpy()) for name, value in actor.named_lora_params()], layout)
    return out, state, values


@pytest.mark.parametrize('method', ['grpo', 'grpo_short'])
def test_grpo_removes_prescan_without_changing_group_objective(method):
    plain, plain_state, plain_actor = run_batch(method, baseline={'prescan': 0})
    requested, state, actor = run_batch(method, baseline={'prescan': 8})
    assert requested['prescan_records'] == [] and requested['n_prescan'] == 0
    assert state.prescan_rng is None and not state.baseline.values
    assert plain_state.rng.state_dict()['counters'] == state.rng.state_dict()['counters']
    assert [r.full_token_ids for r in plain['records']] == [r.full_token_ids for r in requested['records']]
    assert [r.advantage for r in plain['records']] == [r.advantage for r in requested['records']]
    np.testing.assert_array_equal(plain_actor, actor)


def test_fixed_and_smoothed_baselines_use_only_prebatch_information():
    fixed, state, _ = run_batch('full_pg', baseline={'mode': 'fixed', 'fixed_value': .3, 'prescan': 4}, reward=0.)
    assert fixed['n_prescan'] == 0 and not state.baseline.values
    assert all(r.baseline_b == .3 and r.advantage == -.3 for r in fixed['records'])
    smooth, state, _ = run_batch('full_pg', baseline={
        'prescan': 4, 'batch_prescan': True, 'prescan_prior_strength': 2., 'prescan_prior_mean': .5}, reward=0.)
    assert smooth['n_prescan'] == 2 and len(smooth['prescan_records']) == 8
    assert all(r.baseline_b == pytest.approx(1/6) and r.advantage == pytest.approx(-1/6) for r in smooth['records'])
    assert all(r['baseline_after_prescan'] == pytest.approx(1/6) for r in smooth['prescan_records'])
    # Current zero rewards update history only after those frozen advantages.
    assert state.baseline.get('a') == pytest.approx(.3/6)


def test_no_cv_switch_changes_only_current_batch_correction():
    enabled, _, actor_on = run_batch(cv=True)
    disabled, _, actor_off = run_batch(cv=False)
    assert enabled['n_stopped'] > 0
    np.testing.assert_array_equal(enabled['p'], disabled['p'])
    np.testing.assert_array_equal(enabled['z'], disabled['z'])
    assert all(np.count_nonzero(r.f) == 0 for r in disabled['records'])
    assert all(r.reward is None for r in disabled['records'] if r.z == 0)
    correction = prediction_grad_correction(enabled['u_frozen'], np.stack([r.f for r in enabled['records']]),
                                          enabled['z'], enabled['p'], len(enabled['records']))
    assert np.linalg.norm(correction) > 0
    np.testing.assert_allclose(actor_on - actor_off, -.01*correction, atol=1e-8, rtol=1e-5)
    assert enabled['control_variate_used'] and not disabled['control_variate_used']


def test_full_components_p_one_has_same_frozen_batch_update_as_full_pg():
    full, _, full_actor = run_batch('full_pg', beta=1.)
    grace, _, grace_actor = run_batch(beta=1.)
    assert all(r.p == r.z == 1 for r in grace['records'])
    assert [r.full_token_ids for r in full['records']] == [r.full_token_ids for r in grace['records']]
    np.testing.assert_array_equal(full_actor, grace_actor)


def test_uniform_shrink_final_probabilities_reach_sampling_and_records():
    adaptive, _, _ = run_batch(shrink=0.)
    uniform, _, _ = run_batch(shrink=1.)
    active = np.array([r.prefix_tokens < r.response_tokens or r.z == 0 for r in adaptive['records']])
    cost = np.array([r.c_hat for r in adaptive['records']])
    q = np.dot(adaptive['p'][active], cost[active]) / cost[active].sum()
    np.testing.assert_allclose(uniform['p'][active], q)
    np.testing.assert_array_equal(uniform['p'], [r.p for r in uniform['records']])
    selection = IsolatedRNG.create(71).bernoulli('selection', uniform['p'])
    selection[~active] = 1.
    np.testing.assert_array_equal(uniform['z'], selection)


def test_baseline_modes_survive_checkpoint_and_effective_configuration(tmp_path, monkeypatch):
    from grace_gc.config import default_config, merge_configs
    from grace_gc.trainer.loop import run_training
    from grace_gc.trainer.checkpoint import load_checkpoint
    import grace_gc.trainer.loop as loop

    monkeypatch.setattr(loop, 'collect_environment', lambda *a, **k: {'missing': []})
    cfg = merge_configs(default_config(), {'method': 'full_pg', 'num_steps': 1,
        'n_start': 2, 'n_prompts': 2, 'decision_tokens': 2, 'max_new_tokens': 5,
        'baseline': {'mode': 'fixed', 'fixed_value': .3}})
    run_training(copy.deepcopy(cfg), tmp_path/'first')
    first = load_checkpoint(tmp_path/'first/checkpoint.npz')
    # Resuming must describe the loaded baseline, despite a different request.
    cfg['baseline'] = {'mode': 'ema', 'prescan': 4}
    cfg['resume'] = str(tmp_path/'first/checkpoint.npz')
    run_training(cfg, tmp_path/'resumed')
    resumed = load_checkpoint(tmp_path/'resumed/checkpoint.npz')
    assert first['baseline'] == resumed['baseline']
    assert resumed['run_config']['baseline']['mode'] == 'fixed'
    assert resumed['run_config']['baseline']['fixed_value'] == .3
