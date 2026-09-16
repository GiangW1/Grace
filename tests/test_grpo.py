import numpy as np

from grace_gc.trainer.advantages import grpo_advantages, history_advantages
from grace_gc.trainer.baseline import HistoricalBaseline
from grace_gc.trainer.grace_step import decide_continuation
from grace_gc.trainer.methods import method_spec


def test_grpo_group_mean_no_std():
    rewards = [1.0, 0.0, 1.0, None]
    pids = ["a", "a", "b", "b"]
    adv = grpo_advantages(rewards, pids)
    np.testing.assert_allclose(adv[:2], [0.5, -0.5])
    assert adv[2] == 0.0
    assert adv[3] == 0.0


def test_history_advantage_uses_baseline():
    base = HistoricalBaseline()
    base.values["x"] = 0.25
    adv = history_advantages([1.0, None], ["x", "x"], base)
    np.testing.assert_allclose(adv, [0.75, 0.0])


def test_sample_starts_keeps_distinct_problems():
    from grace_gc.core.rng import IsolatedRNG
    from grace_gc.data.math_data import MathRecord
    from grace_gc.trainer.loop import sample_starts

    recs = [MathRecord(problem_id=str(i), prompt=f"q{i}", answer=str(i)) for i in range(20)]
    batch = sample_starts(recs, n_prompts=8, starts_per_prompt=2, rng=IsolatedRNG.create(0))
    pids = [rec.problem_id for rec in batch]
    assert len(pids) == 16
    assert pids[0::2] == pids[1::2]
    assert len(set(pids[0::2])) == 8
    small = [MathRecord(problem_id=str(i), prompt=f"q{i}", answer="0") for i in range(2)]
    overflow = sample_starts(small, n_prompts=5, starts_per_prompt=1, rng=IsolatedRNG.create(1))
    assert len(overflow) == 5
    assert {rec.problem_id for rec in overflow} <= {"0", "1"}


def test_grpo_always_continues():
    spec = method_spec("grpo")
    p, _ = decide_continuation(spec, np.ones(3), np.ones(3), np.zeros(3, dtype=bool), 0.5, 0.2, warmup=False)
    assert np.all(p == 1.0)
    p2, _ = decide_continuation(method_spec("grpo_short"), np.ones(2), np.ones(2), np.zeros(2, dtype=bool), 0.5, 0.2, False)
    assert np.all(p2 == 1.0)
