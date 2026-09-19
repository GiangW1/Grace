"""Independent current-policy supervision and expanded behavior coverage."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from grace_gc.backends import logprob_probe
from grace_gc.core.layout import collect_lora_layout
from grace_gc.core.rng import IsolatedRNG
from grace_gc.predictor.heads import predictor_from_spec
from grace_gc.predictor.reservoir import GradientReservoir
from grace_gc.trainer.algorithm import TrainState, run_algorithm1_step
from grace_gc.trainer.baseline import HistoricalBaseline
from grace_gc.trainer.cpu_tiny import TinyLoRAActor
from grace_gc.trainer.methods import method_spec
from grace_gc.trainer.supervision import collect_fresh_supervision
from grace_gc.trainer.tiny_engine import make_tiny_engines


def setup_tiny():
    torch.manual_seed(73)
    actor = TinyLoRAActor()
    engines = make_tiny_engines(actor, actor.vocab)
    prompts = [[1, 2], [1, 3], [2, 3]]
    feats = engines.prefix_features(prompts, np.array([1, 1, 1]), [.5]*3)['features']
    layout = collect_lora_layout(actor.named_lora_params())
    predictor = predictor_from_spec(feats.shape[1], 2, {'coord_kind': 'ridge', 'hidden_risk': 4})
    state = TrainState(method_spec('grace'), HistoricalBaseline(), IsolatedRNG.create(71),
                       layout, np.eye(layout.dim, 2), predictor, GradientReservoir(32))
    return actor, engines, state, prompts


def test_fresh_labels_keep_actor_rng_grad_baseline_and_rollout_unchanged():
    actor, engines, state, prompts = setup_tiny()
    engines.last_rollout = {'sentinel': [1, 2]}
    for p in engines.trainable_params():
        p.grad = torch.ones_like(p)
    rng = copy.deepcopy(state.rng.state_dict())
    params = [p.detach().clone() for p in engines.trainable_params()]
    heads = copy.deepcopy(state.predictor.state_dict())
    cfg = {'decision_tokens': 2, 'max_new_tokens': 5,
           'predictor': {'fresh_samples_per_problem': 2, 'fresh_every': 1}}
    items, rows, metrics = collect_fresh_supervision(engines, state, prompts, ['a', 'b', 'a'], ['', '', ''], cfg)
    assert len(items) == 4 and metrics['generated_tokens'] == sum(row['response_tokens'] for row in rows)
    assert all(item.source == 'fresh_policy' and item.p == item.s == 1 for item in items)
    assert all(not row['used_in_actor_update'] for row in rows)
    assert repr(rng) == repr(state.rng.state_dict())
    assert not state.baseline.values and not state.reservoir.items
    assert engines.last_rollout == {'sentinel': [1, 2]}
    for before, now in zip(params, engines.trainable_params()):
        torch.testing.assert_close(before, now, rtol=0, atol=0)
        assert torch.all(now.grad == 1)
    from grace_gc.versions import sha256_mapping
    assert sha256_mapping(heads) == sha256_mapping(state.predictor.state_dict())
    # Recreating this step's independent stream reproduces the labels exactly.
    again, replay, _ = collect_fresh_supervision(engines, state, prompts, ['a', 'b', 'a'], ['', '', ''], cfg)
    assert rows == replay
    for a, b in zip(items, again):
        np.testing.assert_array_equal(a.g, b.g)


def test_fresh_supervision_does_not_change_current_actor_update_or_stopper_rewards():
    outputs = []
    for count in (0, 1):
        actor, engines, state, prompts = setup_tiny()
        # A synchronized frozen head exercises actual stopping on this batch.
        state.basis_id = state.predictor_synced_basis_id = 1
        opt = torch.optim.AdamW(engines.trainable_params(), lr=.001)
        cfg = {'decision_tokens': 2, 'max_new_tokens': 5, 'n_prompts': 3,
               'baseline': {'prescan': 0}, 'allocation': {'beta': .2, 'p_min': .2},
               'predictor': {'warmup_steps': 0, 'audit_s': 0., 'epochs': 1,
                             'fresh_samples_per_problem': count, 'fresh_every': 1}}
        out = run_algorithm1_step(engines, state, prompts, ['a', 'b', 'c'], ['', '', ''], cfg, opt)
        outputs.append((out, [p.detach().clone() for p in engines.trainable_params()]))
    left, right = outputs[0][0], outputs[1][0]
    np.testing.assert_array_equal(left['z'], right['z'])
    assert left['n_stopped'] > 0
    assert all(r.reward is None for r in right['records'] if r.z == 0)
    for a, b in zip(outputs[0][1], outputs[1][1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert right['predictor']['fresh_supervision']['n'] == 3


def test_behavior_batch_includes_stopped_prefixes_and_reports_incomplete_scores(monkeypatch):
    monkeypatch.setattr(logprob_probe, 'hf_response_logprobs',
                        lambda model, ids, plen, *a, **k: np.full(len(ids)-plen, -.5))
    records = [
        SimpleNamespace(z=1, prompt_len=1, full_token_ids=[1, 2, 3], problem_id='a',
                        rollout_token_logprobs=[-.5, -.5], rollout_logprob_sum=-1.),
        SimpleNamespace(z=0, prompt_len=1, full_token_ids=None, prefix_token_ids=[1, 3], problem_id='b',
                        rollout_token_logprobs=[-.6], rollout_logprob_sum=-.6),
        SimpleNamespace(z=1, prompt_len=1, full_token_ids=[1, 3], problem_id='c',
                        rollout_token_logprobs=[None], rollout_logprob_sum=None),
    ]
    out = logprob_probe.probe_behavior_batch(None, None, records, 0)
    assert out['checked_sequences'] == 3 and out['coverage_complete']
    assert out['status'] == 'partial' and out['n_tokens'] == 3
    assert out['mean_abs'] == pytest.approx(.1/3)
    assert out['details'][1]['observed_scope'] == 'stopped_prefix'
    assert out['details'][2]['missing_behavior_positions'] == [0]
    assert records[1].z == 0 and records[1].full_token_ids is None


def test_behavior_limit_round_robins_across_problem_outcome(monkeypatch):
    monkeypatch.setattr(logprob_probe, 'hf_response_logprobs', lambda *a, **k: np.array([-.5]))
    def rec(pid):
        return SimpleNamespace(z=1, prompt_len=1, full_token_ids=[1, 2], problem_id=pid,
                               rollout_token_logprobs=[-.5], rollout_logprob_sum=-.5)
    out = logprob_probe.probe_behavior_batch(None, None, [rec('a'), rec('a'), rec('b')], 0, max_sequences=2)
    assert [row['problem_id'] for row in out['details']] == ['a', 'b']
    assert not out['coverage_complete']


def test_behavior_batch_does_not_replace_missing_behavior_with_new_generation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('missing original behavior must remain unavailable')
    monkeypatch.setattr(logprob_probe, 'probe_first_completed', forbidden)
    rec = SimpleNamespace(z=0, prompt_len=1, full_token_ids=None, prefix_token_ids=[1, 2],
                          problem_id='q', rollout_token_logprobs=None)
    out = logprob_probe.probe_behavior_batch(None, None, [rec], 0)
    assert out['details'][0]['reason'] == 'original_behavior_scores_missing'


def test_fresh_reward_cv_uses_frozen_success_and_keeps_r_minus_b(monkeypatch):
    _, engines, state, prompts = setup_tiny()
    state.spec = method_spec('reward_cv')
    monkeypatch.setattr(state.predictor, 'forward_success', lambda feats: np.full(len(feats), .9))
    items, rows, _ = collect_fresh_supervision(
        engines, state, prompts, ['a', 'b', 'c'], ['', '', ''],
        {'decision_tokens': 2, 'max_new_tokens': 5,
         'predictor': {'fresh_samples_per_problem': 1, 'warmup_steps': 0}})
    for item, row in zip(items, rows):
        assert item.reward_feat[0] == pytest.approx(.9)
        assert row['q_hat'] == pytest.approx(.9)
        assert row['advantage'] == row['reward'] - row['baseline_b']


def test_predictable_fresh_chain_resumes_same_learned_state(tmp_path, monkeypatch):
    from grace_gc.config import default_config, merge_configs
    from grace_gc.trainer.checkpoint import load_checkpoint
    from grace_gc.trainer.loop import run_training
    from grace_gc.versions import sha256_mapping
    import grace_gc.trainer.loop as loop

    monkeypatch.setattr(loop, 'collect_environment', lambda *a, **k: {'missing': []})
    original_engines = loop.make_tiny_engines
    def synthetic_engines(*args, **kwargs):
        engines = original_engines(*args, **kwargs)
        engines.reward_fn = lambda ids, gold, **kw: float(ids[-1] % 2)
        return engines
    monkeypatch.setattr(loop, 'make_tiny_engines', synthetic_engines)
    cfg = merge_configs(default_config(), {
        'num_steps': 3, 'n_prompts': 4, 'n_start': 8, 'decision_tokens': 2, 'max_new_tokens': 5,
        'prompt_max_tokens': 8, 'save_initial_checkpoint': True, 'baseline': {'prescan': 1},
        'predictor': {'k': 2, 'coord_kind': 'ridge', 'hidden_risk': 4, 'epochs': 1,
                      'warmup_steps': 1, 'audit_s': 1., 'refresh_every': 1,
                      'basis_variant': 'predictable_crossfit', 'basis_ipw': True,
                      'max_age_steps': 2, 'age_half_life': 1,
                      'fresh_samples_per_problem': 1, 'fresh_every': 1,
                      'shrink_calibration': 'holdout', 'feature_scaler': 'fixed'},
    })
    run_training(cfg, tmp_path/'continuous')
    run_training(merge_configs(cfg, {'num_steps': 2}), tmp_path/'resumed')
    run_training(merge_configs(cfg, {'num_steps': 1, 'resume': str(tmp_path/'resumed/checkpoint.npz')}), tmp_path/'resumed')
    a, b = [load_checkpoint(tmp_path/name/'checkpoint.npz') for name in ('continuous', 'resumed')]
    assert any(np.linalg.norm(row['g']) > 0 for row in a['reservoir']['items'])
    for key in ('actor', 'predictor', 'rng', 'baseline', 'reservoir', 'basis', 'optimizer'):
        assert sha256_mapping({'value': a[key]}) == sha256_mapping({'value': b[key]}), key


def test_final_summary_uses_actual_budget_endpoint_instead_of_step40(tmp_path):
    import json
    from scripts.summarize_minimal import summarize

    train = tmp_path/'grace/train'
    train.mkdir(parents=True)
    cost = {'global_wall_seconds': 125., 'run_wall_seconds_budget': 120., 'run_overshoot_seconds': 5.}
    (train/'summary.json').write_text(json.dumps({'step': 47, 'cost_control': cost}))
    stages = {'grace': {'train': 'grace/train', 'eval-final': 'grace/eval-final'}}
    for stage, value in [('eval-40', .1), ('eval-final', .7)]:
        folder = tmp_path/'grace'/stage
        folder.mkdir()
        (folder/'eval_summary.json').write_text(json.dumps({'avg': value, 'checkpoint': str(train/'checkpoint.npz')}))
    (tmp_path/'stages.json').write_text(json.dumps(stages))
    summarize(tmp_path)
    row = json.loads((tmp_path/'comparison.json').read_text())['methods'][0]
    assert row['final_avg4'] == .7 and row['final_training_step'] == 47
    assert row['cost_control'] == cost


def test_missing_final_does_not_promote_old_step40(tmp_path):
    import json
    from scripts.summarize_minimal import summarize

    train, evaluation = tmp_path/'grace/train', tmp_path/'grace/eval-40'
    train.mkdir(parents=True); evaluation.mkdir()
    (train/'summary.json').write_text(json.dumps({'step': 55}))
    (evaluation/'eval_summary.json').write_text(json.dumps({'avg': .9, 'checkpoint': str(train/'checkpoints/step_40.npz')}))
    summarize(tmp_path)
    row = json.loads((tmp_path/'comparison.json').read_text())['methods'][0]
    assert row['final_training_step'] == 55 and row['final_avg4'] is None


def test_stage_identity_rejects_other_seed_chain_and_stale_final(tmp_path):
    import json
    from scripts.summarize_minimal import checkpoint_matches, summarize

    train = tmp_path/'chain-A/seed-17/grace/train'
    train.mkdir(parents=True)
    assert not checkpoint_matches('/server/chain-A/seed-23/grace/train/checkpoint.npz', train)
    assert not checkpoint_matches('/server/chain-B/seed-17/grace/train/checkpoint.npz', train)
    assert checkpoint_matches('/server/chain-A/seed-17/grace/train/checkpoint.npz', train)
    # Relocation uses saved original identity, not the new directory's spelling.
    (train/'run_meta.json').write_text(json.dumps({'run_dir': '/server/chain-B/seed-17/grace/train'}))
    assert checkpoint_matches('/server/chain-B/seed-17/grace/train/checkpoint.npz', train)
    (train/'summary.json').write_text(json.dumps({'step': 55}))
    evaluation = train.parent/'eval-final'
    evaluation.mkdir()
    (evaluation/'eval_summary.json').write_text(json.dumps({'avg': .9,
        'checkpoint': '/server/chain-B/seed-17/grace/train/checkpoint.npz', 'checkpoint_step': 40}))
    summarize(train.parent.parent)
    row = json.loads((train.parent.parent/'comparison.json').read_text())['methods'][0]
    assert row['final_avg4'] is None
    assert any(item['reason'] == 'checkpoint_step_is_not_training_endpoint' for item in row['source_issues'])
