import json
from pathlib import Path

import numpy as np
import pytest

from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles
from grace_gc.audit.stats import elf, lag_index, plc
from grace_gc.data.math_data import MathRecord, load_math_records, records_for_split, split_records
from grace_gc.data.reward import extract_boxed, first_parseable_index, rule_reward
from grace_gc.evaluation.eval_full import EvalItem, evaluate_items
from grace_gc.evaluation.metrics import pass_at_k, time_to_target, wilson_interval


def test_dedup_and_split(tmp_path: Path):
    path = tmp_path / "math.jsonl"
    rows = [
        {"problem_id": "1", "prompt": "1+1", "answer": "2"},
        {"problem_id": "1b", "prompt": "1+1", "answer": "2"},
        {"problem_id": "2", "prompt": "2+2", "answer": "4"},
        {"problem_id": "3", "prompt": "3+3", "answer": "6"},
        {"problem_id": "4", "prompt": "4+4", "answer": "8"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    recs = load_math_records(path)
    assert len(recs) == 4
    splits = split_records(recs, train_frac=0.5, calib_frac=0.25, audit_frac=0.0, seed=0)
    assert set(splits) == {"train", "calib", "audit", "eval"}
    ids = {k: {r.problem_id for r in v} for k, v in splits.items()}
    assert ids["train"].isdisjoint(ids["eval"])
    assert ids["train"].isdisjoint(ids["calib"])


def test_hf_encode_uses_chat_template_when_present():
    from grace_gc.data.math_data import MathRecord
    from grace_gc.data.tokenize import encode_prompt_hf, encode_records_hf

    class Tok:
        chat_template = "dummy"

        def apply_chat_template(self, messages, enable_thinking=None, **kwargs):
            assert messages[0]["content"] == "1+1"
            assert kwargs.get("add_generation_prompt") is True
            assert enable_thinking is False
            return [[7, 8, 9]]

        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3]}

    assert encode_prompt_hf(Tok(), "1+1", 16) == [7, 8, 9]
    ids, pids, golds = encode_records_hf([MathRecord("p", "1+1", "2")], Tok(), 16)
    assert ids == [[7, 8, 9]]
    assert golds == ["2"]

    class Roles:
        chat_template = "dummy"

        def apply_chat_template(self, messages, enable_thinking=None, **kwargs):
            assert [m["role"] for m in messages] == ["system", "user"]
            assert messages[1]["content"] == "1+1"
            return [11, 12]

        def __call__(self, text, **kwargs):
            return {"input_ids": [1]}

    rec = MathRecord("p", "1+1", "2", messages=[{"role": "system", "content": "math"}, {"role": "user", "content": "1+1"}])
    assert encode_records_hf([rec], Roles(), 16)[0] == [[11, 12]]

    class Bare:
        chat_template = None

        def __call__(self, text, **kwargs):
            return {"input_ids": [4, 5]}

    assert encode_prompt_hf(Bare(), "1+1", 16) == [4, 5]

    class Thinks:
        chat_template = "dummy"

        def apply_chat_template(self, messages, **kwargs):
            return [1, 2]

        def decode(self, ids):
            return "<|im_start|>assistant\n<think>\n"

        def __call__(self, text, **kwargs):
            return {"input_ids": [1]}

    with pytest.raises(ValueError, match="thinking open"):
        encode_prompt_hf(Thinks(), "1+1", 16)

    class ClosedThink:
        chat_template = "dummy"

        def apply_chat_template(self, messages, enable_thinking=None, **kwargs):
            return [3, 4]

        def decode(self, ids):
            return "<|im_start|>assistant\n<think>\n\n</think>\n\n"

        def __call__(self, text, **kwargs):
            return {"input_ids": [1]}

    assert encode_prompt_hf(ClosedThink(), "1+1", 16) == [3, 4]

    class LongChat:
        chat_template = "dummy"

        def apply_chat_template(self, messages, enable_thinking=None, **kwargs):
            assert kwargs.get("truncation") is not True
            return list(range(20))

        def decode(self, ids):
            return "ok"

    assert encode_prompt_hf(LongChat(), "1+1", 8) == list(range(12, 20))

    class ContentChat:
        chat_template = "dummy"

        def apply_chat_template(self, messages, enable_thinking=None, **kwargs):
            body = "".join(str(item["content"]) for item in messages)
            return [0, 1] + [ord(ch) % 10 for ch in body] + [8, 9]

        def decode(self, ids):
            return "ok"

    fitted = encode_prompt_hf(ContentChat(), "abcdefghij", 8)
    assert fitted[:2] == [0, 1]
    assert fitted[-2:] == [8, 9]
    assert len(fitted) <= 8
    assert fitted[2:-2] == [ord(ch) % 10 for ch in "abcd"]


def test_encode_records_prompt_truncated_flag():
    from grace_gc.data.math_data import MathRecord
    from grace_gc.data.tokenize import encode_prompt_hf_detail, encode_records_hf

    class LongChat:
        chat_template = "dummy"

        def apply_chat_template(self, messages, enable_thinking=None, **kwargs):
            return list(range(20))

        def decode(self, ids):
            return "ok"

    ids, meta = encode_prompt_hf_detail(LongChat(), "1+1", 8)
    assert meta["prompt_truncated"] is True
    assert meta["untruncated_len"] == 20
    assert len(ids) <= 8
    flags = []
    encode_records_hf([MathRecord("p", "1+1", "2")], LongChat(), 8, prompt_meta=flags)
    assert flags[0]["prompt_truncated"] is True


def test_dapo_chat_prompt_and_ground_truth(tmp_path: Path):
    path = tmp_path / "dapo.jsonl"
    row = {
        "data_source": "math_dapo",
        "prompt": [
            {
                "role": "user",
                "content": "Solve the following math problem step by step.\n\n1+1",
            }
        ],
        "reward_model": {"ground_truth": "\\boxed{2}", "style": "rule-lighteval/MATH_v2"},
        "extra_info": {"index": "uuid-1", "split": "train"},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rec = load_math_records(path)[0]
    assert rec.prompt.startswith("Solve the following")
    assert "[" not in rec.prompt
    assert rec.messages == [
        {"role": "user", "content": "Solve the following math problem step by step.\n\n1+1"}
    ]
    assert rec.answer == "2"
    assert rec.problem_id == "uuid-1"
    assert rec.source == "math_dapo"
    assert rec.split is None
    splits = split_records([rec], train_frac=0.7, calib_frac=0.1, audit_frac=0.1, seed=0)
    assert sum(len(v) for v in splits.values()) == 1


def test_dapo_json_strings_keep_messages_and_gold(tmp_path: Path):
    path = tmp_path / "dapo-str.jsonl"
    row = {
        "prompt": json.dumps([{"role": "user", "content": "2+2"}]),
        "reward_model": json.dumps({"ground_truth": "4", "style": "rule"}),
        "extra_info": json.dumps({"index": "p-2", "split": "train"}),
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rec = load_math_records(path)[0]
    assert rec.messages == [{"role": "user", "content": "2+2"}]
    assert rec.prompt == "2+2"
    assert rec.answer == "4"
    assert rec.problem_id == "p-2"
    assert rec.split is None


def test_dapo_numpy_chat_rows_keep_messages():
    from grace_gc.data.math_data import _chat_messages, _prompt_text

    rows = np.array(
        [{"role": "user", "content": "1+1"}],
        dtype=object,
    )
    assert _chat_messages(rows) == [{"role": "user", "content": "1+1"}]
    assert _prompt_text(rows) == "1+1"


def test_later_holdout_mark_is_kept(tmp_path: Path):
    path = tmp_path / "hold.jsonl"
    rows = [
        {"prompt": "same q", "answer": "1"},
        {"prompt": "same q", "answer": "1", "split": "audit"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    recs = load_math_records(path)
    assert len(recs) == 1
    assert recs[0].split == "audit"
    assert [r.problem_id for r in split_records(recs, seed=17)["audit"]] == [recs[0].problem_id]


def test_numeric_zero_index_is_kept_as_problem_id(tmp_path: Path):
    path = tmp_path / "idx0.jsonl"
    rows = [
        {"prompt": "q0", "answer": "1", "extra_info": {"index": 0}},
        {"prompt": "q1", "answer": "1", "extra_info": {"index": 1}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    recs = load_math_records(path)
    assert [r.problem_id for r in recs] == ["0", "1"]
    hashed = split_records(
        [
            MathRecord(problem_id="deadbeefdeadbeef", prompt="q0", answer="1"),
            MathRecord(problem_id="1", prompt="q1", answer="1"),
        ]
        + [MathRecord(problem_id=str(i), prompt=f"q{i}", answer="1") for i in range(2, 2500)],
        seed=17,
    )
    kept = split_records(
        [MathRecord(problem_id=str(i), prompt=f"q{i}", answer="1") for i in range(2500)],
        seed=17,
    )
    assert {r.problem_id for r in hashed["audit"]} != {r.problem_id for r in kept["audit"]}


def test_missing_ground_truth_raises(tmp_path: Path):
    path = tmp_path / "n.jsonl"
    path.write_text(json.dumps({"prompt": "q"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ground_truth"):
        load_math_records(path)


def test_unmarked_dapo_scale_reserves_paper_calib_audit():
    recs = [MathRecord(problem_id=str(i), prompt=f"q{i}", answer="1") for i in range(2500)]
    a = split_records(recs, seed=17)
    b = split_records(recs, seed=17)
    assert len({r.problem_id for r in a["calib"]}) == 256
    assert len({r.problem_id for r in a["audit"]}) == 240
    assert len({r.problem_id for r in a["train"]}) == 2500 - 256 - 240
    assert a["eval"] == []
    assert {r.problem_id for r in a["audit"]} == {r.problem_id for r in b["audit"]}
    assert {r.problem_id for r in a["train"]}.isdisjoint({r.problem_id for r in a["audit"]})
    assert {r.problem_id for r in a["train"]}.isdisjoint({r.problem_id for r in a["calib"]})
    assert [r.problem_id for r in records_for_split(recs, "audit", seed=17)] == [
        r.problem_id for r in a["audit"]
    ]
    tiny = recs[:8]
    assert {r.problem_id for r in records_for_split(tiny, "eval", seed=17)} == {str(i) for i in range(8)}
    with pytest.raises(ValueError, match="MATH-500"):
        records_for_split(recs, "eval", seed=17)


def test_marked_split_honored(tmp_path: Path):
    path = tmp_path / "marked.jsonl"
    rows = [
        {"problem_id": "t", "prompt": "train q", "answer": "1", "split": "train"},
        {"problem_id": "e", "prompt": "eval q", "answer": "2", "split": "eval"},
        {"problem_id": "a", "prompt": "audit q", "answer": "3", "split": "audit"},
        {"problem_id": "c", "prompt": "calib q", "answer": "4", "split": "calib"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    splits = split_records(load_math_records(path), seed=0)
    assert [r.problem_id for r in splits["train"]] == ["t"]
    assert [r.problem_id for r in splits["eval"]] == ["e"]
    assert [r.problem_id for r in splits["audit"]] == ["a"]
    assert [r.problem_id for r in splits["calib"]] == ["c"]


def test_reward_and_parse_position():
    from grace_gc.data.reward import normalize_answer

    text = "think \\boxed{42} done"
    assert rule_reward(text, "42") == 1.0
    assert first_parseable_index(["think ", "\\boxed{42}", " done"]) == 1
    assert rule_reward(None, "1") is None
    assert extract_boxed("x \\boxed{\\frac{1}{2}} y") == "\\frac{1}{2}"
    assert rule_reward("think \\boxed{\\frac{1}{2}}", "\\frac{1}{2}") == 1.0
    assert normalize_answer(r"\left( 3, \dfrac{\pi}{2} \right)") == r"(3,\pi/2)"
    assert normalize_answer("(3, \u03c0/2)") == r"(3,\pi/2)"
    assert normalize_answer(r"\frac{\pi}{2}") == r"\pi/2"
    assert rule_reward(r"think \boxed{(3,\pi/2)}", r"\left( 3, \dfrac{\pi}{2} \right)") == 1.0
    assert rule_reward("The girl is Evelyn.", r"\text{Evelyn}") == 1.0
    assert rule_reward(r"think \boxed{Evelyn}", r"\text{Evelyn}") == 1.0
    assert rule_reward("The girl is Alice.", r"\text{Evelyn}") == 0.0


def test_rule_reward_verifies_extracted_pred(monkeypatch):
    calls = []

    def parse(value):
        return str(value)

    def verify(gold, pred):
        calls.append(pred)
        return pred == r"\boxed{0.5}"

    monkeypatch.setattr("grace_gc.data.reward.math_verify_fns", lambda: (parse, verify))
    looping = r"\boxed{0.5} " + ("x" * 80)
    assert rule_reward(looping, r"\frac{1}{2}") == 1.0
    assert calls[0] == r"\boxed{0.5}"


@pytest.mark.parametrize(
    "text,gold,expected",
    [
        ("Answer: 3", r"\left(3,\frac{\pi}{2}\right)", 0.0),
        (r"Answer: (3,\pi/2)", r"\left(3,\frac{\pi}{2}\right)", 1.0),
        ("Answer: (3, pi/2)", r"\left(3,\frac{\pi}{2}\right)", 1.0),
        ("Answer: (3, pi/3)", r"\left(3,\frac{\pi}{2}\right)", 0.0),
        ("Answer: (3, \u03c0/2)", r"\left(3,\frac{\pi}{2}\right)", 1.0),
        (r"Answer: p*i", r"\pi", 0.0),
        (r"Answer: pi", r"p i", 1.0),
        ("Answer: q-p", "p-q", 0.0),
        ("Answer: -q+p", "p-q", 1.0),
        ("Answer: 0.5", r"\frac{1}{2}", 1.0),
        ("I considered 42.\nAnswer: 41", "42", 0.0),
        ("I considered Evelyn.\nAnswer: Alice", r"\text{Evelyn}", 0.0),
        (r"Answer: 3\text{ cm}", r"5\text{ cm}", 0.0),
    ],
)
def test_reward_checks_complete_final_expression(text, gold, expected):
    pytest.importorskip("math_verify")
    assert rule_reward(text, gold) == expected


def test_rule_reward_none_pred_does_not_scan_prose(monkeypatch):
    def parse(value):
        return str(value)

    def verify(gold, pred):
        return "42" in pred and "42" in gold

    monkeypatch.setattr("grace_gc.data.reward.math_verify_fns", lambda: (parse, verify))
    assert rule_reward("the value is 42.", "42") == 0.0
    assert rule_reward("The girl is Evelyn.", r"\text{Evelyn}") == 1.0


def test_format_sft_trains_last_line_not_immediate_answer():
    from grace_gc.data.format_prompt import format_sft_answer_prefix, format_sft_lead, format_sft_response, format_sft_text

    lead = format_sft_lead()
    tail = format_sft_response("61")
    text = format_sft_text("61")
    assert lead.strip()
    assert not lead.lstrip().startswith("Answer:")
    assert tail == "Answer: 61"
    assert text.startswith(lead)
    assert text.endswith(tail)
    assert "Answer:" not in lead
    assert (lead + format_sft_answer_prefix()).startswith(lead)


def test_answer_already_emitted_ignores_process_boxed():
    from grace_gc.data.reward import answer_already_emitted

    assert answer_already_emitted("work \\boxed{27} more steps") is False
    assert answer_already_emitted("the answer is 10\nnow continue") is False
    assert answer_already_emitted("work\nAnswer: 343/27") is True
    assert answer_already_emitted("done \\boxed{27}") is True


def test_extract_prefers_later_answer_line():
    from grace_gc.data.reward import extract_answer

    assert extract_answer("work \\boxed{27} more\nAnswer: 343/27") == "343/27"
    assert extract_answer("Answer: 3\n\\boxed{(3,\\pi/2)}") == r"(3,\pi/2)"
    assert extract_answer("the answer is 90") == "90"
    assert extract_answer("Final Answer:\n90") == "90"


def test_normalize_degree_unit_and_frac():
    from grace_gc.data.reward import normalize_answer

    assert normalize_answer(r"90^\circ") == "90"
    assert normalize_answer(r"90^{\circ}") == "90"
    assert normalize_answer(r"\frac{14}{3}") == "14/3"
    assert rule_reward("Answer: 90", r"90^\circ") == 1.0
    assert rule_reward(r"Answer: 5", r"5\text{ cm}") == 1.0
    assert rule_reward(r"Answer: 3\text{ cm}", r"5\text{ cm}") == 0.0


def test_solve_instruction_and_math500_flag():
    from grace_gc.data.format_prompt import apply_solve_instruction, ensure_solve_instruction, has_solve_instruction
    from grace_gc.data.math_data import looks_like_math500

    bare = "How many primes?"
    wrapped = ensure_solve_instruction(bare)
    assert "Answer: $Answer" in wrapped
    assert has_solve_instruction(wrapped)
    assert ensure_solve_instruction(wrapped) == wrapped
    assert has_solve_instruction("Solve the following math problem step by step.\n\n1+1")
    recs = apply_solve_instruction([MathRecord("1", bare, "2")])
    assert "Answer: $Answer" in recs[0].prompt
    assert recs[0].messages[0]["content"] == recs[0].prompt
    eight = [MathRecord(str(i), "q", "1") for i in range(8)]
    assert looks_like_math500(eight) is False
    assert looks_like_math500(eight, n_source=500) is True
    assert looks_like_math500(eight, path="data/MATH-500/test.parquet") is True
    assert looks_like_math500([MathRecord("1", "q", "1", source="HuggingFaceH4/MATH-500")]) is True


def test_eval_metrics():
    items = [
        EvalItem("a", "1", ["\\boxed{1}", "no"], [False, False]),
        EvalItem("b", "2", ["\\boxed{3}"], [True]),
    ]
    out = evaluate_items(items, k=1)
    assert out["n_problems"] == 2
    assert out["parse_rate"] == pytest.approx(2 / 3)
    assert pass_at_k(4, 1, 1) > 0
    _p, lo, hi = wilson_interval(1, 2)
    assert 0 <= lo <= hi <= 1
    assert time_to_target([(1, 0.1), (2, 0.5), (3, 0.5)], 0.5) == 3


def test_audit_outputs_all_and_selected():
    rng = np.random.default_rng(0)
    g1 = np.array([[1.0, 0.0, 0.0], [1.01, 0.0, 0.0], [0.99, 0.0, 0.0], [1.0, 0.01, 0.0]])
    g2 = np.array([[0.0, 1.0, 0.0], [0.0, 1.01, 0.0], [0.0, 0.99, 0.0], [0.01, 1.0, 0.0]])
    bundles = [
        PrefixBundle("p", 8, np.array([1.0, 0.0, 1.0, 0.0]), g1, path_id="p:0"),
        PrefixBundle("p", 8, np.array([0.0, 1.0, 0.0, 1.0]), g2, path_id="p:1"),
        PrefixBundle("p", 16, np.array([0.0, 1.0, 0.0, 1.0]), g2, path_id="p:0"),
        PrefixBundle("q", 8, np.array([1.0, 1.0, 0.0, 0.0]), rng.normal(size=(4, 3)), path_id="q:0"),
    ]
    u = np.eye(3, 2)
    out = audit_bundles(bundles, u, {"answer_undecided": 0.15, "nonzero_signal_kappa": 0.0}, rng)
    assert out["n_bundles"] == 4
    assert out["plc"] is not None
    assert 0.0 <= float(out["plc"]) <= 1.0
    assert out["rho_l_all"] != 1.0
    assert abs(out["rho_l_all"] - 1.0) > 1e-6
    assert "all" in out["gates"] and "selected" in out["gates"]
    assert np.isnan(elf(np.array([np.nan]), np.array([np.nan])))
    assert plc(1, 1, 4) == 0.5
    assert np.isfinite(lag_index(np.linspace(1, 0, 5), np.linspace(0.5, 0, 5), np.arange(5.0)))
    assert elf(np.array([1.0]), np.array([np.nan])) == 1.0
    assert elf(np.array([np.nan]), np.array([1.0])) == 0.0
    assert elf(np.array([1.0, np.nan]), np.array([2.0, np.nan])) == 0.5
    wide = lag_index(np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([2.0, 4.0]), length=8.0)
    narrow = lag_index(np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([2.0, 4.0]), length=4.0)
    assert wide != narrow


def test_prompt_denom_equal_weights_finished_prefix():
    from grace_gc.audit.prefix_audit import PrefixBundle, _prompt_stats

    finished = PrefixBundle("p", 0, np.array([0.0]), np.array([[0.0, 0.0]]), finished=True, path_id="p:0")
    unfinished = PrefixBundle("p", 0, np.ones(8), np.ones((8, 2)), finished=False, path_id="p:1")
    var_g, _ = _prompt_stats([finished, unfinished])
    assert var_g == pytest.approx(1.0)


def test_unfinished_single_row_cond_var_is_nan():
    from grace_gc.audit.prefix_audit import _cond_var

    assert np.isnan(_cond_var(np.array([[1.0, 0.0]]), finished=False))
    assert _cond_var(np.array([[1.0, 0.0]]), finished=True) == 0.0


def test_text_gold_and_pi_frac_do_not_need_verify(monkeypatch):
    monkeypatch.setattr("grace_gc.data.reward.math_verify_fns", lambda: None)
    from grace_gc.data.reward import text_gold_in_response

    assert text_gold_in_response("The girl is Evelyn.", r"\text{Evelyn}")
    assert not text_gold_in_response("The girl is Alice.", r"\text{Evelyn}")
    assert rule_reward("The girl is Evelyn.", r"\text{Evelyn}") == 1.0
    assert rule_reward("The girl is Alice.", r"\text{Evelyn}") == 0.0
    assert rule_reward(r"\boxed{(3,\pi/2)}", r"\left( 3, \frac{\pi}{2} \right)") == 1.0


def test_lora_and_baseline_health_fields():
    from grace_gc.logging_util.forensics import lora_param_health, step_health
    from grace_gc.trainer.grace_step import StartRecord

    class _P:
        def __init__(self, data, grad=None):
            self._data = np.asarray(data, dtype=np.float32)
            self.grad = None if grad is None else _P(grad)

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._data

    health = lora_param_health(
        [
            ("q_proj.lora_A", _P([[1.0]], None)),
            ("q_proj.lora_B", _P([[0.0]], [[0.0]])),
        ]
    )
    assert health["n_lora_A_grad_missing"] == 1
    assert health["n_lora_B_grad_zero"] == 1
    assert health["lora_B_near_zero"] is True
    assert health["lora_A_grad_zero_expected"] is True

    rec = StartRecord(
        problem_id="p",
        finished=True,
        p=1.0,
        z=1.0,
        f=np.zeros(1),
        r_hat=0.0,
        c_hat=1.0,
        reward=0.0,
        advantage=0.0,
        g=None,
        audited=False,
        baseline_b=0.0,
    )
    state = type(
        "S",
        (),
        {
            "step": 1,
            "spec": type("M", (), {"use_predictor": False})(),
            "reservoir": None,
            "baseline": type("B", (), {"values": {"p": 0.0, "q": 0.5}})(),
            "basis_id": None,
        },
    )()
    out = step_health(state, {"records": [rec]})
    assert out["mean_baseline_b"] == pytest.approx(0.25)
    assert out["n_baseline"] == 2
    assert out["n_baseline_zero"] == 1
    assert out["baseline_collapsed_with_zero_reward"] is True
    rec.baseline_b = 0.5
    later = step_health(state, {"records": [rec]})
    assert later["n_baseline_zero"] == 1
    assert later["baseline_collapsed_with_zero_reward"] is False
