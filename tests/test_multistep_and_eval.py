import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from grace_gc.audit.run import run_audit
from grace_gc.data.math_data import load_math_records
from grace_gc.evaluation.generate import run_eval
from grace_gc.trainer.loop import build_run_config, run_training
from grace_gc.trainer.checkpoint import load_checkpoint


def _math_jsonl(path: Path) -> Path:
    rows = [
        {"problem_id": "a", "prompt": "1+1", "answer": "2"},
        {"problem_id": "b", "prompt": "2+2", "answer": "4"},
        {"problem_id": "c", "prompt": "3+3", "answer": "6"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_multistep_train_and_resume(tmp_path: Path):
    data = _math_jsonl(tmp_path / "math.jsonl")
    cfg = build_run_config(
        [],
        {
            "seed": 3,
            "backend": "cpu_tiny",
            "data_path": str(data),
            "n_prompts": 2,
            "n_start": 4,
            "decision_tokens": 2,
            "max_new_tokens": 3,
            "num_steps": 2,
            "predictor": {"k": 2, "warmup_steps": 0, "audit_s": 1.0, "reservoir_size": 16, "epochs": 1},
        },
    )
    first = run_training(cfg, tmp_path / "run1")
    assert first["run_status"] == "complete"
    assert first["summary"]["steps"] == 2
    assert first["summary"]["step"] == 2
    steps = [
        json.loads(line)
        for line in (tmp_path / "run1" / "steps.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [row["step"] for row in steps] == [1, 2]
    ckpt = load_checkpoint(first["summary"]["checkpoint"])
    assert ckpt["actor"]
    assert ckpt["optimizer"]
    assert ckpt["step"] == 2

    cfg2 = build_run_config(
        [],
        {
            "seed": 3,
            "backend": "cpu_tiny",
            "data_path": str(data),
            "n_prompts": 2,
            "n_start": 4,
            "decision_tokens": 2,
            "max_new_tokens": 3,
            "num_steps": 1,
            "resume": first["summary"]["checkpoint"],
            "predictor": {"k": 2, "warmup_steps": 0, "audit_s": 1.0, "reservoir_size": 16, "epochs": 1},
        },
    )
    second = run_training(cfg2, tmp_path / "run2")
    assert second["summary"]["step"] == 3


def test_eval_and_audit_generate(tmp_path: Path):
    data = _math_jsonl(tmp_path / "math.jsonl")
    recs = load_math_records(data)
    ev = run_eval(recs, {"backend": "cpu_tiny", "eval_k": 1, "eval_n": 2, "max_new_tokens": 3, "seed": 1}, tmp_path / "eval")
    assert ev["n_problems"] == 3
    capped = run_eval(
        recs,
        {"backend": "cpu_tiny", "eval": {"n_problems": 1, "k": 1, "n": 1}, "max_new_tokens": 3, "seed": 1},
        tmp_path / "eval-cap",
    )
    assert capped["n_problems"] == 1
    cap_splits = json.loads((tmp_path / "eval-cap" / "data_splits.json").read_text(encoding="utf-8"))
    assert cap_splits["n_source"] == 3
    assert cap_splits["looks_like_math500"] is False
    run_eval(
        recs,
        {
            "backend": "cpu_tiny",
            "eval": {"n_problems": 1, "k": 1, "n": 1},
            "max_new_tokens": 3,
            "seed": 1,
            "data_path": str(tmp_path / "MATH-500.jsonl"),
        },
        tmp_path / "eval-m500",
    )
    m500 = json.loads((tmp_path / "eval-m500" / "data_splits.json").read_text(encoding="utf-8"))
    assert m500["looks_like_math500"] is True
    eval_rows = [
        json.loads(line)
        for line in (tmp_path / "eval" / "eval_per_problem.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert eval_rows
    assert "extracted" in eval_rows[0]
    assert "response_tokens" in eval_rows[0]
    assert (tmp_path / "eval" / "run.log").is_file()
    au = run_audit(recs[:1], {"backend": "cpu_tiny", "n_prefixes": 1, "n_cont": 3, "decision_tokens": 2, "max_new_tokens": 2, "seed": 2}, tmp_path / "audit")
    assert (tmp_path / "audit" / "run.log").is_file()
    if au["n_bundles"] == 0:
        assert au.get("note") == "no prefixes"
    else:
        assert au["n_bundles"] == 1
        assert "variance_cost" in au
        bundle_path = tmp_path / "audit" / "audit_bundles.jsonl"
        assert bundle_path.is_file()
        first = json.loads(bundle_path.read_text(encoding="utf-8").splitlines()[0])
        assert "prefix_text" in first
        assert "suffix_texts" in first


def test_prompt_cv_uses_prompt_features():
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.core.rng import IsolatedRNG
    from grace_gc.predictor.heads import PredictorHeads
    from grace_gc.predictor.reservoir import GradientReservoir
    from grace_gc.trainer.algorithm import TrainState, run_algorithm1_step
    from grace_gc.trainer.baseline import HistoricalBaseline
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.methods import method_spec
    from grace_gc.trainer.tiny_engine import make_tiny_engines

    actor = TinyLoRAActor()
    engines = make_tiny_engines(actor, actor.vocab)
    layout = collect_lora_layout(actor.named_lora_params())
    captured = {}
    orig = engines.prefix_features

    def wrap(prefixes, prompt_lens, baselines):
        bundle = orig(prefixes, prompt_lens, baselines)
        captured["bundle"] = bundle
        return bundle

    engines.prefix_features = wrap
    seen = {}

    class SpyHeads(PredictorHeads):
        def forward_numpy(self, x, cost_feat=None):
            seen["x"] = np.asarray(x).copy()
            return super().forward_numpy(x, cost_feat)

    probe_rng = IsolatedRNG.create(0)
    prompts = [[1, 2, 3], [4, 5, 6]]
    prefixes, _ = engines.generate_prefix(prompts, 2, probe_rng, "token")
    probe = engines.prefix_features(prefixes, np.array([3, 3]), [0.4, 0.6])
    k = 2
    spy = SpyHeads(in_dim=int(probe["features"].shape[1]), k=k, hidden_coord=8, hidden_risk=8)
    opt = __import__("torch").optim.SGD(actor.trainable_params(), lr=0.01)
    state = TrainState(
        spec=method_spec("prompt_cv"),
        baseline=HistoricalBaseline(),
        rng=IsolatedRNG.create(1),
        layout=layout,
        u=np.eye(layout.dim, k, dtype=np.float64),
        predictor=spy,
        reservoir=GradientReservoir(capacity=8),
    )
    run_algorithm1_step(
        engines,
        state,
        prompts,
        ["a", "b"],
        ["1", "2"],
        {"decision_tokens": 2, "max_new_tokens": 3, "predictor": {"warmup_steps": 0, "audit_s": 0.0, "epochs": 1}},
        opt,
    )
    assert "x" in seen
    np.testing.assert_allclose(seen["x"], captured["bundle"]["prompt_features"])
    assert not np.allclose(seen["x"], captured["bundle"]["features"])


def test_prompt_cv_missing_features_raises():
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.core.rng import IsolatedRNG
    from grace_gc.predictor.reservoir import GradientReservoir
    from grace_gc.trainer.algorithm import TrainState, run_algorithm1_step
    from grace_gc.trainer.baseline import HistoricalBaseline
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.methods import method_spec
    from grace_gc.trainer.tiny_engine import make_tiny_engines

    actor = TinyLoRAActor()
    engines = make_tiny_engines(actor, actor.vocab)
    orig = engines.prefix_features

    def drop_prompt(prefixes, prompt_lens, baselines):
        bundle = dict(orig(prefixes, prompt_lens, baselines))
        bundle.pop("prompt_features", None)
        return bundle

    engines.prefix_features = drop_prompt
    opt = __import__("torch").optim.SGD(actor.trainable_params(), lr=0.01)
    layout = collect_lora_layout(actor.named_lora_params())
    state = TrainState(
        spec=method_spec("prompt_cv"),
        baseline=HistoricalBaseline(),
        rng=IsolatedRNG.create(2),
        layout=layout,
        u=np.eye(layout.dim, 2, dtype=np.float64),
        predictor=None,
        reservoir=GradientReservoir(capacity=4),
    )
    with pytest.raises(ValueError, match="prompt_features"):
        run_algorithm1_step(
            engines,
            state,
            [[1, 2, 3]],
            ["a"],
            ["1"],
            {"decision_tokens": 2, "max_new_tokens": 3, "predictor": {"warmup_steps": 0}},
            opt,
        )


def test_prompt_cv_train_step(tmp_path: Path):
    data = _math_jsonl(tmp_path / "math.jsonl")
    cfg = build_run_config(
        [],
        {
            "seed": 5,
            "method": "prompt_cv",
            "backend": "cpu_tiny",
            "data_path": str(data),
            "n_prompts": 2,
            "n_start": 4,
            "decision_tokens": 2,
            "max_new_tokens": 3,
            "num_steps": 1,
            "predictor": {"k": 2, "warmup_steps": 0, "audit_s": 0.0, "reservoir_size": 8, "epochs": 1},
        },
    )
    out = run_training(cfg, tmp_path / "prompt-cv")
    assert out["summary"]["method"] == "prompt_cv"
    assert out["run_status"] == "complete"
