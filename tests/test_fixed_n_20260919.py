"""Fixed-N comparisons must bypass both wall and token recycling, including resume."""

import json

from grace_gc.config import default_config, merge_configs
from grace_gc.trainer import loop


def test_fixed_n_ignores_token_next_n_and_resumes_fixed(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, 'collect_environment', lambda *a, **k: {'missing': []})
    original = loop.run_algorithm1_step

    def token_recycling_suggestion(*args, **kwargs):
        result = original(*args, **kwargs)
        result['next_n'] = 8
        args[1].history_costs.append(.25)
        return result

    monkeypatch.setattr(loop, 'run_algorithm1_step', token_recycling_suggestion)
    cfg = merge_configs(default_config(), {
        'method': 'uniform_ht', 'num_steps': 2, 'n_start': 4, 'n_prompts': 2,
        'decision_tokens': 2, 'max_new_tokens': 5, 'prompt_max_tokens': 8,
        'format_warmup': {'steps': 0}, 'baseline': {'mode': 'fixed', 'prescan': 0},
        'cost_control': {'enabled': False, 'mode': 'fixed'},
        'predictor': {'warmup_steps': 0}, 'checkpoint_every': 1,
    })
    folder = tmp_path / 'fixed'
    loop.run_training(cfg, folder)
    rows = [json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    assert [row['n'] for row in rows] == [4, 4]
    resumed = merge_configs(cfg, {'num_steps': 1, 'resume': str(folder/'checkpoint.npz')})
    loop.run_training(resumed, tmp_path/'resume')
    rows = [json.loads(line) for line in (tmp_path/'resume'/'steps.jsonl').read_text().splitlines()]
    assert rows[-1]['n'] == 4


def test_four_signal_arms_share_actor_inputs_n_and_p1_warmup(tmp_path, monkeypatch):
    """Synthetic CPU integration, not a GPU performance/quality experiment."""
    from grace_gc.config import load_config
    from grace_gc.trainer.checkpoint import load_checkpoint
    from grace_gc.versions import sha256_mapping

    monkeypatch.setattr(loop, 'collect_environment', lambda *a, **k: {'missing': []})
    cfg = merge_configs(default_config(), load_config('configs/experiments/minimal_gpu_signal_candidate.yaml'), {
        'num_steps': 4, 'n_start': 4, 'n_prompts': 2,
        'decision_tokens': 2, 'max_new_tokens': 5, 'prompt_max_tokens': 8,
        'format_warmup': {'steps': 0}, 'baseline': {'prescan': 1, 'batch_prescan': True},
        'cost_control': {'mode': 'fixed'}, 'checkpoint_every': 1,
        'predictor': {'k': 2, 'warmup_steps': 2, 'refresh_after_warmup': True,
                      'refresh_every': 4, 'coord_kind': 'ridge', 'feature_scaler': 'fixed',
                      'shrink_calibration': 'holdout', 'reservoir_size': 32, 'epochs': 1},
    })
    shared = tmp_path/'shared'
    loop.run_training(merge_configs(cfg, {'method': 'full_pg', 'num_steps': 0}), shared)
    source = str(shared/'checkpoints'/'step_0.npz')
    arms = {'grace': {}, 'p1': {'allocation': {'beta': 1.}},
            'm0': {'predictor': {'control_variate': False}},
            'uniform': {'allocation': {'uniform_shrink': 1.}}}
    initial_hashes, inputs, warmup_hashes = set(), [], set()
    for name, knobs in arms.items():
        folder = tmp_path/name
        loop.run_training(merge_configs(cfg, {'method': 'grace', 'init_checkpoint': source}, knobs), folder)
        initial_hashes.add(json.loads((folder/'initial_actor.json').read_text())['actor_sha256'])
        steps = [json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
        assert [row['n'] for row in steps] == [4]*4
        assert all(row['audit_gradient_source'] == 'mandatory_actor_backward' for row in steps)
        trajectories = [json.loads(s) for s in (folder/'trajectories.jsonl').read_text().splitlines()]
        inputs.append([(row['step'], row['problem_id'], row['prompt_token_ids']) for row in trajectories])
        warmup_hashes.add(sha256_mapping(load_checkpoint(folder/'checkpoints'/'step_2.npz')['actor']))
        if name == 'p1':
            assert all(row['p_continue'] == row['z_continue'] == 1. for row in trajectories)
        if name == 'm0':
            assert all(not any(row['prediction_coords']) for row in trajectories)
        assert all(row['reward'] is None for row in trajectories if row['z_continue'] == 0)
    assert len(initial_hashes) == len(warmup_hashes) == 1
    assert inputs[1:] == [inputs[0]]*3
