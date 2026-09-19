"""Compare saved evals, or run fresh evaluate.py processes on one checkpoint.

Examples:
  python scripts/check_eval_repeatability.py compare RUN_A RUN_B --run-dir runs/check
  python scripts/check_eval_repeatability.py repeat --config CFG --checkpoint CKPT \
      --data-path EVAL --repeats 2 --batch-sizes 1 4 --run-dir runs/repeats
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.logging_util.run_dir import RunDirectory, default_run_dir, resolve_run_dir

RUNTIME_ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "VLLM_ATTENTION_BACKEND", "VLLM_ENABLE_V1_MULTIPROCESSING",
                    "VLLM_BATCH_INVARIANT", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def _token_hash(ids):
    return None if ids is None else hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def _load(root):
    root = Path(root)
    summary = _json(root/"eval_summary.json")
    cfg = (yaml.safe_load((root/"config.yaml").read_text(encoding="utf-8")) or {}) if (root/"config.yaml").is_file() else {}
    env, actor = _json(root/"environment.json"), _json(root/"actor_source.json")
    checkpoint = summary.get("checkpoint_identity") or {}
    identity = {
        "actor_state": actor.get("actor_sha256") or checkpoint.get("actor_state_sha256"),
        "actor_hash_scope": actor.get("layout") if actor.get("actor_sha256") else checkpoint.get("scope"),
        "loaded_hf_lora_sha256": actor.get("actor_sha256"), "hf_numerics": actor.get("numerics"),
        "checkpoint_step": summary.get("checkpoint_step"), "seed": summary.get("seed"),
        "backend": cfg.get("backend"), "model_path": cfg.get("model_path"),
        "model_config_sha256": ((env.get("model") or {}).get("files", {}).get("config.json") or {}).get("sha256"),
        "prompt_max_tokens": cfg.get("prompt_max_tokens"), "lora_config": cfg.get("lora"),
        "vllm_config": cfg.get("vllm"), "engine": _json(root/"vllm_engine.json"),
        "tokenizer": _json(root/"tokenizer.json"), "packages": env.get("packages"),
        "vllm_version": (env.get("packages") or {}).get("vllm"),
        "runtime_environment": {key: env["env"][key] for key in RUNTIME_ENV_KEYS}
            if all(key in (env.get("env") or {}) for key in RUNTIME_ENV_KEYS) else None,
    }
    samples, issues = {}, []
    for row in _rows(root/"eval_per_problem.jsonl"):
        for i, answer in enumerate(row.get("answers") or []):
            key = (str(row.get("problem_id")), i)
            if key in samples:
                issues.append("duplicate_problem_sample:" + str(key))
            def at(field):
                values = row.get(field) or []
                return values[i] if i < len(values) else None
            samples[key] = {"answer": answer, "tokens": at("token_ids"), "seed": at("sample_seeds"),
                            "truncated": at("truncated"), "finish_reason": at("finish_reasons")}
    requests = {(str(r.get("problem_id")), r.get("sample_index")): r for r in _rows(root/"eval_requests.jsonl")}
    return {"root": str(root.resolve()), "summary": summary, "identity": identity,
            "manifest": summary.get("evaluation_manifest") or _json(root/"evaluation_manifest.json"),
            "samples": samples, "requests": requests, "issues": issues}


def compare_runs(left, right):
    left, right = _load(left), _load(right)
    checks, issues = {}, []
    def check(field, a, b, required=True):
        known = a is not None and b is not None and a != "" and b != ""
        if required and field in {"engine", "tokenizer", "packages"}:
            known = known and bool(a) and bool(b)
        checks[field] = {"left": a, "right": b, "equal": a == b if known else None}
        if (required and not known) or (known and a != b):
            issues.append({"field": field, "reason": "different" if known else "missing"})
    for field, a in left["identity"].items():
        required = field in {"actor_state", "actor_hash_scope", "checkpoint_step", "seed", "backend", "prompt_max_tokens"}
        if left["identity"].get("backend") == "gpu_verl":
            required = field not in {"loaded_hf_lora_sha256", "hf_numerics"}
        check(field, a, right["identity"].get(field), required=required)
    # File hash is optional for old artifacts, but a known mismatch must be shown.
    check("checkpoint_sha256", left["summary"].get("checkpoint_sha256"), right["summary"].get("checkpoint_sha256"), required=False)
    for field in ("ordered_records_sha256", "n_problems", "reward_protocol_version", "samples_per_problem",
                  "temperature", "top_p", "max_new_tokens", "sample_seed_start", "sample_batch_size"):
        check(field, left["manifest"].get(field), right["manifest"].get(field))
    keys = sorted(set(left["samples"]) | set(right["samples"]))
    rows = []
    for key in keys:
        a, b = left["samples"].get(key, {}), right["samples"].get(key, {})
        la, lb = a.get("tokens"), b.get("tokens")
        first = None
        if la is not None and lb is not None:
            for i in range(max(len(la), len(lb))):
                va, vb = la[i] if i < len(la) else None, lb[i] if i < len(lb) else None
                if va != vb:
                    first = {"response_index": i, "left_token": va, "right_token": vb}
                    break
        ra, rb = left["requests"].get(key, {}), right["requests"].get(key, {})
        request_diff = {field: {"left": ra.get(field), "right": rb.get(field)} for field in
            ("request_seed", "prompt_token_sha256", "chunk_index", "requested_batch_size", "execution_batch_size",
             "serial_fallback", "serial_fallback_reason") if ra.get(field) != rb.get(field)}
        for label, sample, request in (("left", a, ra), ("right", b, rb)):
            if request.get("request_seed") is not None and request["request_seed"] != sample.get("seed"):
                issues.append({"field": label + "_request_seed:" + str(key), "reason": "trace_sample_mismatch"})
            if request.get("response_token_sha256") and request["response_token_sha256"] != _token_hash(sample.get("tokens")):
                issues.append({"field": label + "_response_hash:" + str(key), "reason": "trace_sample_mismatch"})
        rows.append({"problem_id": key[0], "sample_index": key[1], "left_seed": a.get("seed"), "right_seed": b.get("seed"),
            "left_token_sha256": _token_hash(la), "right_token_sha256": _token_hash(lb),
            "left_n_tokens": None if la is None else len(la), "right_n_tokens": None if lb is None else len(lb),
            "tokens_equal": None if la is None or lb is None else la == lb, "first_difference": first,
            "text_equal": a.get("answer") == b.get("answer") if a and b else None,
            "finish_equal": (a.get("finish_reason"), a.get("truncated")) == (b.get("finish_reason"), b.get("truncated")),
            "request_trace_available": bool(ra) and bool(rb), "request_differences": request_diff})
        if a.get("seed") is None or a.get("seed") != b.get("seed"):
            issues.append({"field": "sample_seed:" + str(key), "reason": "missing_or_different"})
        if ra.get("prompt_token_sha256") != rb.get("prompt_token_sha256"):
            issues.append({"field": "prompt_tokens:" + str(key), "reason": "missing_or_different"})
    complete = bool(rows) and set(left["samples"]) == set(right["samples"]) and not left["issues"] and not right["issues"]
    for side in (left, right):
        expected = (side["manifest"].get("n_problems") or 0) * (side["manifest"].get("samples_per_problem") or 0)
        complete &= len(side["samples"]) == expected
    if not complete:
        issues.append({"field": "sample_inventory", "reason": "incomplete_or_duplicate", "left": left["issues"], "right": right["issues"]})
    different = [item for item in issues if item["reason"] == "different"]
    classification = ("same_protocol_repeat" if not issues else
                      "batch_size_single_factor" if len(issues) == 1 and different and different[0]["field"] == "sample_batch_size" else
                      "confounded" if different else "unverified")
    exact = all(row["tokens_equal"] for row in rows) if complete and all(row["tokens_equal"] is not None for row in rows) else None
    return {"left": left["root"], "right": right["root"], "classification": classification,
            "identity_checks": checks, "identity_issues": issues, "exact_tokens_equal": exact,
            "n_samples": len(rows), "samples": rows,
            "weight_identity_scope": "Checkpoint actor-state or loaded HF LoRA hash; not full base-model or vLLM worker tensor identity unless separately measured. Matching paths/config hashes alone do not verify base weights.",
            "scope": "Observed token traces for these runs only; no GPU determinism guarantee, significance threshold, or acceptance gate. Batch-size controls are not same-protocol repetitions."}


def repeat_evaluations(cfg, run_dir, repeats=2, batch_sizes=None, split="eval"):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not cfg.get("checkpoint") or not cfg.get("data_path"):
        raise ValueError("repeat requires checkpoint and data_path")
    batches = batch_sizes or [int((cfg.get("eval") or {}).get("sample_batch_size", 1))]
    if any(size < 1 for size in batches):
        raise ValueError("batch sizes must be positive")
    run = RunDirectory(resolve_run_dir(run_dir))
    result = {"status": "running", "runs": [], "comparisons": [], "run_dir": str(run.root.resolve()),
              "design": "Fresh evaluate.py subprocess and engine per repetition. Only requested eval.sample_batch_size changes between batch-size groups."}
    run.write_json("summary.json", result)
    anchors = []
    for batch in batches:
        group = []
        for repeat in range(repeats):
            label = f"batch-{batch}-repeat-{repeat}"
            config = copy.deepcopy(cfg)
            config.setdefault("eval", {})["sample_batch_size"] = batch
            config_path = run.write_yaml(label + ".yaml", config).resolve()
            target, stdout = run.root/label, run.root/(label + ".stdout")
            command = [sys.executable, str(ROOT/"scripts/measure_command.py"), "--log", str((run.root/"command_timing.jsonl").resolve()),
                       "--stage", label, "--", sys.executable, str(ROOT/"scripts/evaluate.py"), "--generate",
                       "--config", str(config_path), "--checkpoint", str(cfg["checkpoint"]), "--data-path", str(cfg["data_path"]),
                       "--split", split, "--run-dir", str(target.resolve())]
            with stdout.open("w", encoding="utf-8") as stream:
                child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
            paths = [line[len("run_dir "):].strip() for line in stdout.read_text(encoding="utf-8").splitlines() if line.startswith("run_dir ")]
            actual = paths[-1] if paths else str(target.resolve())
            result["runs"].append({"batch_size": batch, "repeat": repeat, "run_dir": actual, "command": command,
                                   "stdout": str(stdout.resolve()), "exit_code": child.returncode})
            if child.returncode:
                result.update(status="failed", exit_code=child.returncode)
                run.write_json("summary.json", result)
                return result
            group.append(actual)
            run.write_json("summary.json", result)
        anchors.append(group[0])
        result["comparisons"].extend(compare_runs(group[0], other) for other in group[1:])
    result["comparisons"].extend(compare_runs(anchors[0], other) for other in anchors[1:])
    result.update(status="completed", exit_code=0)
    run.write_json("summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    compare = modes.add_parser("compare")
    compare.add_argument("left"); compare.add_argument("right")
    repeat = modes.add_parser("repeat")
    repeat.add_argument("--config", action="append", default=[])
    repeat.add_argument("--checkpoint", required=True); repeat.add_argument("--data-path", required=True)
    repeat.add_argument("--model-path"); repeat.add_argument("--backend"); repeat.add_argument("--seed", type=int)
    repeat.add_argument("--split", default="eval"); repeat.add_argument("--repeats", type=int, default=2)
    repeat.add_argument("--batch-sizes", type=int, nargs="+")
    for command in (compare, repeat):
        command.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)
    target = args.run_dir or default_run_dir("eval-repeatability")
    if args.mode == "compare":
        run = RunDirectory(resolve_run_dir(target))
        result = compare_runs(args.left, args.right)
        run.write_json("summary.json", result)
        print(result["classification"], "exact_tokens_equal", result["exact_tokens_equal"])
        print("run_dir", run.root)
        return 0
    from grace_gc.trainer.loop import build_run_config
    cfg = build_run_config(args.config, {key: getattr(args, key) for key in
        ("checkpoint", "data_path", "model_path", "backend", "seed") if getattr(args, key) is not None})
    result = repeat_evaluations(cfg, target, args.repeats, args.batch_sizes, args.split)
    print(result["status"], "run_dir", result["run_dir"])
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
