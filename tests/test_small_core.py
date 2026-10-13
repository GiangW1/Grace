import copy
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from grace_gc.backends.small_core import (SmallCoreLinear, attach_small_core, fixed_basis_state,
                                          restore_fixed_bases, export_small_core_adapter)
from grace_gc.backends.hf_actor import named_lora_params
from grace_gc.core.coordinates import IdentityBasis, as_basis
from grace_gc.core.estimator import batch_ht_mean, dual_stream_mean
from grace_gc.core.layout import collect_lora_layout
from grace_gc.predictor.risk import full_space_residual
from grace_gc.trainer.checkpoint import save_checkpoint, load_checkpoint
from grace_gc.trainer.state_io import numpy_module_state, load_numpy_module_state


def small_model(basis="random", seed=17):
    model = nn.Module()
    model.q_proj, model.v_proj = nn.Linear(7, 5), nn.Linear(7, 3)
    model.config = SimpleNamespace(_name_or_path="fixture")
    return attach_small_core(model, {"parameterization": "small_core", "rank": 2,
                                    "alpha": 4, "basis": basis, "basis_seed": seed})


@pytest.mark.parametrize("basis", ["svd", "random"])
def test_zero_init_gradient_layout_and_frozen_bases(basis):
    model = small_model(basis)
    named = named_lora_params(model)
    assert collect_lora_layout(named).dim == 8
    assert all(name.endswith("lora_core") for name, _ in named)
    x = torch.randn(4, 7)
    block = model.q_proj
    torch.testing.assert_close(block(x), block.base_layer(x), rtol=0, atol=0)
    before = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    block(x).square().sum().backward()
    assert block.lora_core.grad.norm() > 0
    expected = block.scaling * block.fixed_left.T @ (2 * block.base_layer(x)).T @ x @ block.fixed_right.T
    torch.testing.assert_close(block.lora_core.grad, expected)
    opt = torch.optim.SGD([p for _, p in named], lr=.1)
    opt.step()
    for name, p in model.named_parameters():
        if name in before:
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
            assert p.grad is None


def test_random_basis_does_not_consume_global_rng():
    base = nn.Linear(7, 5)
    rng = torch.get_rng_state().clone()
    one = SmallCoreLinear(base, 2, 4, basis="random", seed=3)
    assert torch.equal(rng, torch.get_rng_state())
    two = SmallCoreLinear(copy.deepcopy(base), 2, 4, basis="random", seed=3)
    torch.testing.assert_close(one.fixed_left, two.fixed_left, rtol=0, atol=0)


def test_export_is_equivalent_standard_lora(tmp_path):
    st = pytest.importorskip("safetensors.torch")
    model = small_model()
    with torch.no_grad():
        model.q_proj.lora_core.normal_()
        model.v_proj.lora_core.normal_()
    export_small_core_adapter(model, tmp_path)
    state = st.load_file(str(tmp_path / "adapter_model.safetensors"))
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["r"] == 2 and config["lora_alpha"] == 4
    x = torch.randn(4, 7)
    for name in ("q_proj", "v_proj"):
        block = getattr(model, name)
        a = state[f"base_model.model.{name}.lora_A.weight"]
        b = state[f"base_model.model.{name}.lora_B.weight"]
        torch.testing.assert_close(block(x), block.base_layer(x) + block.scaling * (x @ a.T @ b.T))


def test_real_qwen3_peft_roundtrip_without_download(tmp_path):
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    config = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4,
                                      num_key_value_heads=2, head_dim=8)
    base = transformers.Qwen3ForCausalLM(config).eval()
    actor = attach_small_core(copy.deepcopy(base), {"rank": 2, "alpha": 4, "basis": "svd"})
    with torch.no_grad():
        for _, param in named_lora_params(actor):
            param.normal_(std=.1)
    export_small_core_adapter(actor, tmp_path)
    exported = peft.PeftModel.from_pretrained(base, tmp_path).eval()
    tokens = torch.tensor([[1, 3, 7, 2], [2, 8, 9, 1]])
    torch.testing.assert_close(actor(tokens).logits, exported(tokens).logits, atol=2e-6, rtol=2e-5)
    actor(tokens).logits.square().sum().backward()
    assert all(p.grad is not None and p.grad.norm() > 0 for _, p in named_lora_params(actor))


def test_actor_state_features_use_only_current_trainable_cores(monkeypatch, tmp_path):
    from grace_gc.backends import gpu_engine
    model = small_model()
    monkeypatch.setattr(gpu_engine, "prefix_feature_bundle", lambda *a, **kw:
                        {"features": np.ones((2, 3)), "prompt_features": np.zeros((2, 3)), "cost_feat": np.ones((2, 3))})
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=99)
    engines, _ = gpu_engine.make_gpu_engines(model, None, tokenizer,
                                            {"predictor": {"include_actor_state": True}}, tmp_path)
    bundle = engines.prefix_features([[1], [2]], [1, 1], [.5, .5])
    assert bundle["features"].shape == (2, 11)
    np.testing.assert_array_equal(bundle["features"][:, 3:], 0)
    with torch.no_grad():
        model.q_proj.lora_core.fill_(.25)
    changed = engines.prefix_features([[1], [2]], [1, 1], [.5, .5])
    np.testing.assert_array_equal(changed["features"][:, 3:7], .25)
    np.testing.assert_array_equal(changed["features"][:, :3], bundle["features"][:, :3])


def test_checkpoint_preserves_actual_fixed_bases_not_new_seed(tmp_path):
    from grace_gc.versions import sha256_named
    model = small_model(seed=3)
    with torch.no_grad():
        model.q_proj.lora_core.normal_()
    payload = {"actor": numpy_module_state(named_lora_params(model)),
               "small_core_bases": fixed_basis_state(model),
               "basis": {"u": IdentityBasis(8), "fixed": True}}
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, payload)
    restored = load_checkpoint(path)
    assert isinstance(restored["basis"]["u"], IdentityBasis)
    assert path.stat().st_size < 10000
    other = small_model(seed=99)
    assert sha256_named(named_lora_params(model)) != sha256_named(named_lora_params(other))
    restore_fixed_bases(other, restored)
    load_numpy_module_state(named_lora_params(other), restored["actor"])
    assert sha256_named(named_lora_params(model)) == sha256_named(named_lora_params(other))
    for name in ("q_proj", "v_proj"):
        torch.testing.assert_close(getattr(model, name).fixed_left, getattr(other, name).fixed_left)
        torch.testing.assert_close(getattr(model, name).fixed_right, getattr(other, name).fixed_right)
        torch.testing.assert_close(getattr(model, name).lora_core, getattr(other, name).lora_core)
    with pytest.raises(ValueError, match="all frozen bases"):
        restore_fixed_bases(other, {"actor": payload["actor"]})


def test_identity_large_dimension_stays_compact(tmp_path):
    u = IdentityBasis(18432)
    x = np.arange(u.dim, dtype=float)
    np.testing.assert_array_equal(u @ x, x)
    np.testing.assert_array_equal(x @ u, x)
    assert u.T @ u is u
    assert as_basis(u.tolist()) == u
    save_checkpoint(tmp_path / "u.npz", {"basis": {"u": u, "fixed": True}})
    assert (tmp_path / "u.npz").stat().st_size < 2000
    assert not list(tmp_path.glob("basis-*.npy"))


def test_direct_ht_exact_enumeration_and_p1():
    rng = np.random.default_rng(8)
    g, f = rng.normal(size=(3, 8)), rng.normal(size=(3, 8))
    p = np.array([.2, .7, 1.])
    mean = np.zeros(8)
    u = IdentityBasis(8)
    for bits in itertools.product([0., 1.], repeat=3):
        z = np.array(bits)
        prob = np.prod(np.where(z == 1, p, 1-p))
        chosen = z == 1
        got = dual_stream_mean(g[chosen], p[chosen], u, f, z, p, n=3)
        np.testing.assert_allclose(got, batch_ht_mean(g, f, p, z), atol=1e-14)
        mean += prob * got
    np.testing.assert_allclose(mean, g.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(dual_stream_mean(g, np.ones(3), u, f, np.ones(3), np.ones(3), 3), g.mean(0))
    np.testing.assert_allclose(full_space_residual(g, f, u), np.sum((g-f)**2, axis=1))


def test_core_real_backward_and_prediction_correction_sign():
    from grace_gc.trainer.actor_update import real_stream_backward_and_audit, apply_correction_clip_step
    model = small_model()
    named = named_lora_params(model)
    layout = collect_lora_layout(named)
    x = torch.randn(3, 7)
    score = lambda i: model.q_proj(x[i]).sum() + model.v_proj(x[i]).sum()
    advantage, p, z = np.array([.5, -.5, .5]), np.array([.4, .6, .8]), np.array([1., 0., 1.])
    opt = torch.optim.SGD([p for _, p in named], lr=.01)
    _, labels = real_stream_backward_and_audit(score, advantage, p, z, 3, z == 1,
                                              named, layout, np.ones(3, dtype=bool))
    f = np.random.default_rng(1).normal(size=(3, 8))
    expected = dual_stream_mean(np.stack([labels[0], labels[2]]), p[z == 1], IdentityBasis(8), f, z, p, 3)
    packed, _ = apply_correction_clip_step(named, layout, IdentityBasis(8), f, z, p, 3, opt, clip=0)
    np.testing.assert_allclose(packed, -expected, atol=1e-6)
    new = np.concatenate([p.detach().numpy().ravel() for _, p in named])
    np.testing.assert_allclose(new, .01 * expected, atol=1e-7)


def test_direct_training_checkpoint_resume_and_no_pca(tmp_path, monkeypatch):
    from grace_gc.config import default_config, merge_configs
    from grace_gc.trainer.loop import run_training
    monkeypatch.setattr("grace_gc.trainer.algorithm.refresh_basis",
                        lambda **kwargs: pytest.fail("direct prediction must not fit PCA"))
    cfg = merge_configs(default_config(), {"backend": "cpu_tiny", "num_steps": 3, "n_start": 8, "n_prompts": 4,
        "decision_tokens": 2, "max_new_tokens": 4, "baseline": {"mode": "fixed", "prescan": 0},
        "cost_control": {"mode": "fixed"},
        "predictor": {"target_space": "direct", "fixed_basis": True, "audit_s": 1., "warmup_steps": 1,
                      "shrink_calibration": "holdout", "epochs": 1}})
    root = tmp_path / "direct"
    assert run_training(cfg, root)["run_status"] == "complete"
    payload = load_checkpoint(root / "checkpoint.npz")
    assert isinstance(payload["basis"]["u"], IdentityBasis)
    assert payload["predictor"]["k"] == payload["layout_dim"]
    assert payload["basis"]["basis_id"] == 1
    cfg.update(resume=str(root / "checkpoint.npz"), num_steps=1)
    assert run_training(cfg, root)["run_status"] == "complete"
    assert load_checkpoint(root / "checkpoint.npz")["step"] == 4
    from grace_gc.audit.run import generate_bundles_tiny, _audit_u
    from grace_gc.audit.prefix_audit import audit_bundles
    from grace_gc.data.math_data import MathRecord
    audit_cfg = {**cfg, "checkpoint": str(root / "checkpoint.npz"), "analysis": {"gradient_target": "trajectory"},
                 "audit": {"baseline": .5, "jl_dim": 0}}
    bundles = generate_bundles_tiny([MathRecord("audit", "q1", "1")], 2, 4, [1, 2], 4, 9,
                                    audit_cfg, store_features=True, store_trajectory_gradients=True)
    assert bundles
    assert bundles[0].basis_gram == {"identity_dim": payload["layout_dim"]}
    summary = audit_bundles(bundles, _audit_u(bundles, audit_cfg),
                            {"gradient_target": "trajectory", "require_pre_emit": False}, np.random.default_rng(9))
    assert summary["orthogonal_energy"] < 1e-10


def test_direct_audit_geometry_has_zero_omission():
    from grace_gc.audit.mechanism import full_space_decomposition
    from grace_gc.audit.prefix_audit import orthogonal_energy
    g = np.random.default_rng(1).normal(size=(4, 8))
    u = IdentityBasis(8)
    assert orthogonal_energy(g, u) == 0
    row = full_space_decomposition(np.sum(g*g, axis=1), g, u.tolist(), np.zeros(8))
    np.testing.assert_allclose(row["subspace_omission_energy"], 0)
    np.testing.assert_allclose(row["prediction_error_energy"], np.sum(g*g, axis=1))


def test_suite_arms_keep_shared_protocol_and_fullpg_first(tmp_path):
    from grace_gc.config import load_config
    from scripts.run_small_core_experiment import main, arm_config, ROOT
    common = load_config(ROOT / "configs/experiments/small_core_comparison.yaml")
    full = arm_config(common, "core8_full_pg", 17)
    grace = arm_config(common, "core8_grace", 17)
    assert {k:v for k,v in full.items() if k != "method"} == {k:v for k,v in grace.items() if k != "method"}
    assert grace["predictor"]["target_space"] == "direct"
    assert full["enable_thinking"] is grace["enable_thinking"] is False
    assert main(["--model-path", "explicit-model", "--data-path", "train.jsonl", "--eval-data-path", "eval.jsonl",
                 "--arms", "core8_grace", "core8_full_pg", "--seeds", "17", "--run-dir", str(tmp_path / "suite")]) == 0
    manifest = json.loads((tmp_path / "suite/suite.json").read_text())
    assert manifest["status"] == "planned"
    assert manifest["jobs"][0]["arm"] == "core8_full_pg"
    for config in (tmp_path / "suite").glob("*.yaml"):
        generated = load_config(config)
        assert generated["enable_thinking"] is False
        assert generated["max_new_tokens"] == 8192
