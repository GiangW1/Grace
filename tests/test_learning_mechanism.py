import json
from pathlib import Path

import numpy as np

from grace_gc.audit.learning_stats import mechanism_values, problem_bootstrap_report
from grace_gc.audit.qualification import classify_functional_recovery
from grace_gc.audit.streaming_replay import replay_statistics
from grace_gc.backends.verl_trainer import cap_colocated_vllm_config
from grace_gc.data.math_data import MathRecord, select_records_difficulty


def test_mechanism_statistics_use_cross_fitted_null():
    row = {"baseline": 0.5, "half_mean_reward": [0.25, 0.75], "problem_id": "p"}
    stats = {"hh": 4.0, "ga": 1.0, "gb": 2.0, "ab": 3.0, "s_a": 5.0, "s_b": 6.0}
    values = mechanism_values(row, stats)
    assert values["c_energy"] == 3.0
    assert values["gamma"] == 3.25
    assert values["variance"] == 2.5
    assert values["v_null"] == 5.75


def test_problem_bootstrap_is_clustered_by_problem():
    rows = []
    for problem in ("a", "b"):
        rows.append({"problem_id": problem, "baseline": 0.5,
                     "half_mean_reward": [0.0, 1.0],
                     "mechanism": {"euclidean": {"hh": 1., "ga": 0., "gb": 0., "ab": 1.,
                                                   "s_a": 2., "s_b": 2.}}})
    report = problem_bootstrap_report(rows, "euclidean", bootstrap=20, seed=3)
    assert report["n_problems"] == 2
    assert report["bootstrap_unit"] == "problem; all prefixes for a sampled problem move together"
    assert report["bootstrap_defined"] == 20


def test_streaming_replay_writes_only_scalar_trajectory_stats(tmp_path: Path):
    rows = [{"problem_id": "p", "path_id": "p:0", "t": 1024,
             "baseline": 0.5, "prompt_token_ids": [0], "prefix_token_ids": [0, 1],
             "answer_emitted": False, "finished": False,
             "continuation_records": [
                 {"token_ids": [0, 1, 2], "prompt_len": 1, "reward": 0., "baseline": 0.5},
                 {"token_ids": [0, 1, 3], "prompt_len": 1, "reward": 1., "baseline": 0.5},
             ]}]

    def score(tokens, prompt_len):
        _ = prompt_len
        return np.array([1., 0.]) if tokens[-1] == 2 else np.array([0., 1.])

    result = replay_statistics(rows, score, lambda row: np.array([1., 0.]), 2, tmp_path,
                               max_continuations=2, seed=1)
    assert result["storage"] == "sufficient_statistics"
    record = json.loads((tmp_path / "trajectory_scalars.jsonl").read_text().splitlines()[0])
    assert "trajectory_statistics" in record
    assert not (tmp_path / "mean_grads.npy").exists()
    assert (tmp_path / "batch_statistics.json").is_file()


def test_functional_qualification_requires_fixed_binary_probe():
    report = classify_functional_recovery([0., 1., 1.], majority=0.5)
    assert report["qualification_mean_reward"] == 2.0 / 3.0
    assert report["functional_recoverable"] is True


def test_difficulty_selection_respects_preregistered_mix(tmp_path: Path):
    manifest = tmp_path / "difficulty.json"
    manifest.write_text(json.dumps({"difficulty": {
        "e0": "easy", "e1": "easy", "m0": "medium", "m1": "medium",
        "m2": "medium", "h0": "hard", "h1": "hard",
    }}), encoding="utf-8")
    records = [MathRecord(pid, pid, "1") for pid in ("e0", "e1", "m0", "m1", "m2", "h0", "h1")]
    selected = select_records_difficulty(records, 6, manifest,
                                         {"easy": 1, "medium": 3, "hard": 2}, seed=4)
    assert len(selected) == 6
    labels = {"e0": "easy", "e1": "easy", "m0": "medium", "m1": "medium", "m2": "medium",
              "h0": "hard", "h1": "hard"}
    assert {label: sum(labels[r.problem_id] == label for r in selected)
            for label in ("easy", "medium", "hard")} == {"easy": 1, "medium": 3, "hard": 2}


def test_colocated_vllm_cap_preserves_lower_explicit_limit():
    capped = cap_colocated_vllm_config({}, {"gpu_memory_utilization": 0.5})
    assert capped["gpu_memory_utilization"] == 0.3
    assert capped["max_num_seqs"] == 2
    lower = cap_colocated_vllm_config({"colocated_vllm_memory_utilization": 0.2},
                                      {"gpu_memory_utilization": 0.1, "max_num_seqs": 1})
    assert lower["gpu_memory_utilization"] == 0.1
    assert lower["max_num_seqs"] == 1
