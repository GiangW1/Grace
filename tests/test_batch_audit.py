"""Frozen-batch diagnostics use full generation, not claimed training savings."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from grace_gc.audit.batch_audit import OptimizerReplay, PairedMoments, audit_fixed_batches
from grace_gc.core.layout import collect_lora_layout
from grace_gc.data.math_data import MathRecord
from grace_gc.logging_util.run_dir import RunDirectory
from grace_gc.trainer.methods import method_spec
from grace_gc.trainer.state_io import optimizer_state


def test_online_paired_moments_match_full_matrix():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(9, 4)), rng.normal(size=(9, 4))
    moments = PairedMoments(4)
    for x, y in zip(a, b):
        moments.add(x, y)
    result = moments.summary()
    assert result["full_trace_sample_cov"] == pytest.approx(np.var(a, axis=0, ddof=1).sum())
    assert result["actual_trace_sample_cov"] == pytest.approx(np.var(b, axis=0, ddof=1).sum())
    assert result["paired_mean_difference_norm"] == pytest.approx(np.linalg.norm((b-a).mean(axis=0)))
    assert result["paired_trace_sample_cov"] == pytest.approx(np.var(b-a, axis=0, ddof=1).sum())
    single = PairedMoments(4); single.add(a[0], b[0])
    assert single.summary()["full_trace_sample_cov"] is None


def test_adam_replay_uses_moments_clip_and_never_changes_source():
    torch = pytest.importorskip("torch")
    p = torch.nn.Parameter(torch.tensor([.2, -.3]))
    opt = torch.optim.AdamW([p], lr=.01, weight_decay=.02)
    p.grad = torch.tensor([.4, -.5]); opt.step(); opt.zero_grad()
    snapshot = optimizer_state(opt)
    named = [("q_proj.lora_A", p)]
    replay = OptimizerReplay(named, collect_lora_layout(named), snapshot, {"grad_clip": .5})
    original = p.detach().clone()
    g = np.array([3., 4.])
    delta, stats = replay.step(g)
    assert stats["clip_triggered"] and stats["grad_norm_preclip"] == pytest.approx(5.)
    np.testing.assert_array_equal(p.detach().numpy(), original.numpy())
    expected = torch.nn.Parameter(original.clone())
    direct = torch.optim.AdamW([expected], lr=.01, weight_decay=.02)
    direct.load_state_dict(copy.deepcopy(opt.state_dict()))
    expected.grad = torch.tensor(-g / 10, dtype=expected.dtype); direct.step()
    np.testing.assert_allclose(delta, (expected-original).detach().numpy(), atol=1e-9)
    again, _ = replay.step(g)
    np.testing.assert_array_equal(delta, again)
    np.testing.assert_array_equal(snapshot["state"][0]["exp_avg"], optimizer_state(opt)["state"][0]["exp_avg"])


def _fake_context():
    torch = pytest.importorskip("torch")
    p = torch.nn.Parameter(torch.tensor([.2, -.3]))
    named = [("q_proj.lora_A", p)]
    def prefix(prompts, length, rng, stream):
        values = rng.integers(stream, 1, 5, size=len(prompts))
        return [list(x)+[int(v)]*length for x,v in zip(prompts, values)], np.zeros(len(prompts), bool)
    def continuation(prefixes, selected, length, rng):
        return [list(x)+[int(rng.integers("continuation", 1, 5))]*length if keep else None
                for x,keep in zip(prefixes, selected)]
    engines = SimpleNamespace(generate_prefix=prefix, continue_selected=continuation,
        decode=lambda ids: "Answer: " + str(int(ids[-1] % 2 == 0)), eos_id=99, last_rollout=None,
        named_lora=lambda: named, trainable_params=lambda: [p],
        logprob_one=lambda ids, plen: p[0]*ids[-1] + p[1]*sum(ids[plen:]))
    encode = lambda rec,n: ([[0] for _ in range(n)], [rec.problem_id]*n, [rec.answer]*n)
    opt = torch.optim.AdamW([p], lr=.01)
    return engines, collect_lora_layout(named), encode, optimizer_state(opt)


def test_fixed_n_p1_replays_identical_updates_and_accounts_all_generation(tmp_path):
    engines, layout, encode, opt = _fake_context()
    cfg = {"seed": 17, "max_new_tokens": 4, "decision_tokens": 2,
           "allocation": {"beta": 1., "p_min": .2}, "optim": {"grad_clip": 1.},
           "batch_audit": {"replicates": 3, "starts_per_prompt": 2, "baseline_mode": "fixed", "baseline": .5}}
    result = audit_fixed_batches([MathRecord("a", "qa", "1"), MathRecord("b", "qb", "1")],
        engines, layout, encode, {"optimizer": opt}, None, None, method_spec("uniform_ht"), cfg, RunDirectory(tmp_path))
    assert result["fixed_n"] == 4 and result["replicates"] == 3
    assert result["gradient"]["paired_mean_difference_norm"] == 0
    assert result["update"]["paired_mean_difference_norm"] == 0
    assert result["actual_generation_tokens"] == 48
    raw = [json.loads(x) for x in (tmp_path / "batch_audit_samples.jsonl").read_text().splitlines()]
    assert len(raw) == 12 and {r["problem_id"] for r in raw} == {"a", "b"}
    assert len({r["replicate_seed"] for r in raw}) == 3 and all(r["replicate_seed"] != 17 for r in raw)
    assert all(r["selected_reward"] == r["full_reward"] for r in raw)
    assert all(r["truncated"] and not r["natural_finish"] for r in raw)


def test_selection_isolated_and_stopped_rewards_remain_null(tmp_path):
    records = [MathRecord("a", "qa", "1"), MathRecord("b", "qb", "1")]
    cfg = {"seed": 9, "max_new_tokens": 4, "decision_tokens": 2,
           "allocation": {"beta": .2, "p_min": .2}, "batch_audit": {
               "replicates": 4, "starts_per_prompt": 2, "baseline_mode": "fixed", "baseline": .5}}
    snapshots = []
    for beta in (.2, 1.):
        engines, layout, encode, opt = _fake_context()
        config = copy.deepcopy(cfg); config["allocation"]["beta"] = beta
        run = RunDirectory(tmp_path / str(beta))
        result = audit_fixed_batches(records, engines, layout, encode, {"optimizer": opt}, None, None,
                                     method_spec("uniform_ht"), config, run)
        rows = [json.loads(x) for x in (run.root / "batch_audit_samples.jsonl").read_text().splitlines()]
        snapshots.append(rows)
        assert result["actual_generation_tokens"] == 64
    assert [r["full_token_ids"] for r in snapshots[0]] == [r["full_token_ids"] for r in snapshots[1]]
    stopped = [r for r in snapshots[0] if r["z"] == 0]
    assert stopped and all(r["selected_reward"] is None for r in stopped)


def test_nonzero_cv_uses_full_space_fixed_n_correction(tmp_path):
    engines, layout, encode, opt = _fake_context()
    engines.prefix_features = lambda prefixes,*args: {"features": np.zeros((len(prefixes), 2))}
    predictor = SimpleNamespace(constant_cost=True,
        forward_numpy=lambda features,*args: SimpleNamespace(f=np.tile([2., -3.], (len(features), 1)),
                                                            r_hat=np.ones(len(features))))
    cfg = {"seed": 9, "max_new_tokens": 4, "decision_tokens": 2,
           "allocation": {"beta": .5, "p_min": .2}, "batch_audit": {
               "replicates": 1, "starts_per_prompt": 2, "baseline_mode": "fixed", "baseline": .5}}
    result = audit_fixed_batches([MathRecord("a", "qa", "1"), MathRecord("b", "qb", "1")], engines,
        layout, encode, {"optimizer": opt}, predictor, np.eye(2), method_spec("uniform_cv"), cfg, RunDirectory(tmp_path))
    rows = [json.loads(x) for x in (tmp_path / "batch_audit_samples.jsonl").read_text().splitlines()]
    expected = np.zeros(2)
    for row in rows:
        ids = row["full_token_ids"]
        g = (row["full_reward"]-.5)*np.array([ids[-1], sum(ids[1:])])
        m = np.array([2., -3.])
        expected += (m+row["z"]/row["p"]*(g-m))/4
    means = np.load(tmp_path / "batch_audit_means.npz")
    np.testing.assert_allclose(means["actual_gradient"], expected)
    assert result["gradient"]["actual_trace_sample_cov"] is None


def test_independent_prescan_is_frozen_and_its_cost_is_included(tmp_path):
    records = [MathRecord("a", "qa", "1"), MathRecord("b", "qb", "1")]
    cfg = {"seed": 9, "max_new_tokens": 4, "decision_tokens": 2,
           "allocation": {"beta": .5}, "batch_audit": {"replicates": 2, "starts_per_prompt": 2,
               "baseline_mode": "prescan", "prescan_samples": 3}}
    engines, layout, encode, opt = _fake_context()
    result = audit_fixed_batches(records, engines, layout, encode, {"optimizer": opt}, None, None,
                                 method_spec("uniform_ht"), cfg, RunDirectory(tmp_path))
    baselines = json.loads((tmp_path / "batch_audit_baselines.json").read_text())
    rows = [json.loads(x) for x in (tmp_path / "batch_audit_samples.jsonl").read_text().splitlines()]
    assert result["prescan_generation_tokens"] == 24 and result["actual_generation_tokens"] == 56
    for problem in baselines["problems"]:
        assert len(problem["samples"]) == 3
        assert {row["baseline"] for row in rows if row["problem_id"] == problem["problem_id"]} == {problem["baseline"]}
    assert baselines["rng_before"]["seed"] != cfg["seed"]


def test_tiny_checkpoint_batch_audit_cli_and_cost_envelope(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import yaml
    from grace_gc.audit import batch_audit
    from grace_gc.trainer.checkpoint import save_checkpoint
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.state_io import numpy_module_state
    from scripts.audit_batch import main
    actor = TinyLoRAActor()
    optimizer = torch.optim.SGD(actor.trainable_params(), lr=.01)
    checkpoint = tmp_path / "checkpoint.npz"
    save_checkpoint(checkpoint, {"actor": numpy_module_state(actor.named_lora_params()),
        "actor_full": numpy_module_state(actor.named_all_params()), "optimizer": optimizer_state(optimizer),
        "spec": "full_pg", "step": 1, "baseline": {"values": {"p": .3}}})
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"problem_id": "p", "prompt": "1+1", "answer": "2"})+"\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"backend": "cpu_tiny", "max_new_tokens": 3, "decision_tokens": 1,
        "n_prompts": 1, "n_start": 1, "batch_audit": {"replicates": 1, "starts_per_prompt": 1}}), encoding="utf-8")
    monkeypatch.setattr(batch_audit, "collect_environment", lambda cfg: {})
    assert main(["--generate", "--config", str(config), "--data-path", str(data), "--split", "train",
                 "--checkpoint", str(checkpoint), "--run-dir", str(tmp_path / "audit")]) == 0
    summary = json.loads((tmp_path / "audit/batch_audit_summary.json").read_text())
    ledger = json.loads((tmp_path / "audit/compute_ledger.json").read_text())
    assert summary["method"] == "full_pg" and summary["optimizer_replay"]["class"] == "SGD"
    assert summary["actor_sha256_before"] == summary["actor_sha256_after"]
    assert ledger["rows"][-1]["status"] == "completed" and ledger["cpu_seconds"] >= 0


def test_early_finished_starts_keep_fixed_n_and_pay_no_suffix(tmp_path):
    engines, layout, encode, opt = _fake_context()
    generate = engines.generate_prefix
    def prefixes(prompts, length, rng, stream):
        values, _ = generate(prompts, length, rng, stream)
        return values, np.array([True, False, True, False])
    engines.generate_prefix = prefixes
    cfg = {"seed": 9, "max_new_tokens": 4, "decision_tokens": 2,
           "allocation": {"beta": .2}, "batch_audit": {"replicates": 1, "starts_per_prompt": 2,
                                                        "baseline_mode": "fixed"}}
    result = audit_fixed_batches([MathRecord("a", "qa", "1"), MathRecord("b", "qb", "1")],
        engines, layout, encode, {"optimizer": opt}, None, None, method_spec("uniform_ht"), cfg, RunDirectory(tmp_path))
    rows = [json.loads(x) for x in (tmp_path / "batch_audit_samples.jsonl").read_text().splitlines()]
    assert result["fixed_n"] == 4 and result["actual_generation_tokens"] == 12
    assert len(rows) == 4
    for i in (0, 2):
        assert rows[i]["p"] == rows[i]["z"] == 1
        assert rows[i]["natural_finish"] and rows[i]["full_token_ids"] == rows[i]["prefix_token_ids"]


def test_gpu_entry_wiring_and_failed_envelope_without_gpu(tmp_path, monkeypatch):
    from grace_gc.audit import batch_audit
    from grace_gc.trainer.checkpoint import save_checkpoint
    engines, layout, encode, opt = _fake_context()
    checkpoint = tmp_path / "checkpoint.npz"
    save_checkpoint(checkpoint, {"spec": "full_pg", "optimizer": opt,
        "run_config": {"max_new_tokens": 4, "decision_tokens": 2, "optim": {"grad_clip": .25}}})
    def initialize(records, *args, cache, payload, **kwargs):
        assert records == [] and payload["spec"] == "full_pg"
        cache.update(engines=engines, layout=layout, encode_fn=encode)
    monkeypatch.setattr(batch_audit, "generate_bundles_gpu", initialize)
    monkeypatch.setattr(batch_audit, "collect_environment", lambda cfg: {})
    cfg = {"backend": "gpu_verl", "checkpoint": str(checkpoint), "hardware": {"n_gpu": 1},
           "n_prompts": 1, "n_start": 1, "max_new_tokens": 8, "decision_tokens": 4,
           "batch_audit": {"replicates": 1}}
    records = [MathRecord("p", "q", "1")]
    result = batch_audit.run_batch_audit(records, cfg, tmp_path / "mock-gpu")
    assert result["manifest"]["horizon"] == 4 and result["fixed_n"] == 1
    assert result["optimizer_replay"]["grad_clip"] == .25
    assert result["manifest"]["training_settings_sources"]["optim"] == "checkpoint.run_config"
    with pytest.raises(ValueError, match="exactly one GPU"):
        batch_audit.run_batch_audit(records, {**cfg, "hardware": {"n_gpu": 2}}, tmp_path / "multi-gpu")
    save_checkpoint(checkpoint, {"spec": "full_pg"})
    with pytest.raises(ValueError, match="optimizer state"):
        batch_audit.run_batch_audit(records, cfg, tmp_path / "failed")
    ledger = json.loads((tmp_path / "failed/compute_ledger.json").read_text())
    assert ledger["rows"][-1]["status"] == "failed" and ledger["wall_seconds"] > 0


def test_frozen_actor_mutation_is_detected(tmp_path):
    torch = pytest.importorskip("torch")
    engines, layout, encode, opt = _fake_context()
    original = engines.logprob_one
    def faulty(ids, plen):
        with torch.no_grad():
            engines.trainable_params()[0].add_(.1)
        return original(ids, plen)
    engines.logprob_one = faulty
    cfg = {"seed": 9, "max_new_tokens": 4, "decision_tokens": 2,
           "batch_audit": {"replicates": 1, "starts_per_prompt": 1, "baseline_mode": "fixed"}}
    with pytest.raises(RuntimeError, match="frozen source state"):
        audit_fixed_batches([MathRecord("a", "qa", "1")], engines, layout, encode, {"optimizer": opt},
                            None, None, method_spec("full_pg"), cfg, RunDirectory(tmp_path))
