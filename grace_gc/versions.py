"""Collect available software versions. Missing optional metadata is recorded."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


_PACKAGES = (
    "numpy",
    "torch",
    "yaml",
    "vllm",
    "verl",
    "math_verify",
    "peft",
    "transformers",
    "datasets",
    "huggingface_hub",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(arr) -> str:
    data = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(data.tobytes()).hexdigest()


def sha256_named(named) -> str:
    digest = hashlib.sha256()
    for name, param in named:
        digest.update(str(name).encode("utf-8"))
        if hasattr(param, "detach"):
            arr = param.detach().float().cpu().numpy()
        else:
            arr = np.asarray(param)
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def _update_digest(digest, value) -> None:
    if hasattr(value, "state_dict"):
        value = value.state_dict()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, dict):
        for key in sorted(value):
            digest.update(str(key).encode("utf-8"))
            _update_digest(digest, value[key])
        return
    if hasattr(value, "tobytes"):
        digest.update(np.ascontiguousarray(np.asarray(value)).tobytes())
        return
    digest.update(repr(value).encode("utf-8"))


def sha256_mapping(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    digest = hashlib.sha256()
    _update_digest(digest, payload)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _read_json_subset(path: Path, keys: tuple[str, ...]) -> dict[str, Any] | None:
    raw = _read_json(path)
    if raw is None:
        return None
    return {key: raw.get(key) for key in keys if key in raw}


def _nvidia_smi_text() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=8,
        ).strip()
    except Exception:
        return None


def _nvidia_processes() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=8,
        ).strip()
    except Exception:
        return None


def _nvidia_power_clocks() -> list[dict[str, Any]] | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=8,
        ).strip()
        rows = []
        for line in output.splitlines():
            fields = [value.strip() for value in line.split(",")]
            if len(fields) != 5:
                continue
            row = {"index": fields[0], "uuid": fields[1]}
            for name, value in zip(("power_draw_watts", "sm_clock_mhz", "memory_clock_mhz"), fields[2:]):
                try:
                    row[name] = float(value)
                except ValueError:
                    row[name] = None
            rows.append(row)
        return rows
    except Exception:
        return None


def _disk_usage() -> dict[str, Any] | None:
    try:
        usage = shutil.disk_usage(str(Path.cwd()))
        return {"free_bytes": int(usage.free), "total_bytes": int(usage.total), "cwd": str(Path.cwd())}
    except Exception:
        return None


def resource_snapshot() -> dict[str, Any]:
    """Cheap live facts. Written again each step so a mid-run GPU hog or full disk is visible."""
    info: dict[str, Any] = {
        "pid": os.getpid(),
        "nvidia_smi": _nvidia_smi_text(),
        "nvidia_compute_processes": _nvidia_processes(),
        "nvidia_power_clocks": _nvidia_power_clocks(),
        "disk": _disk_usage(),
        "torch_cuda_allocated_bytes": None,
        "torch_cuda_reserved_bytes": None,
        "torch_cuda_peak_allocated_bytes": None,
        "torch_cuda_peak_reserved_bytes": None,
        "peak_memory_scope": "current_process_since_last_peak_reset",
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["torch_cuda_allocated_bytes"] = int(torch.cuda.memory_allocated(0))
            info["torch_cuda_reserved_bytes"] = int(torch.cuda.memory_reserved(0))
            info["torch_cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
            info["torch_cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))
    except Exception:
        pass
    return info


def describe_path(path_str: str | None) -> dict[str, Any] | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_file():
        return {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    if path.is_dir():
        names = ("config.json", "tokenizer_config.json", "generation_config.json", "adapter_config.json")
        files = {}
        extras: dict[str, Any] = {}
        for name in names:
            child = path / name
            if child.is_file():
                files[name] = {"sha256": sha256_file(child), "bytes": child.stat().st_size}
                if name == "config.json":
                    extras["config"] = _read_json_subset(
                        child, ("model_type", "architectures", "vocab_size", "torch_dtype", "_name_or_path")
                    )
                if name == "generation_config.json":
                    extras["generation_config"] = _read_json_subset(
                        child, ("max_new_tokens", "temperature", "top_p", "eos_token_id")
                    )
        refs = path / "refs" / "main"
        if refs.is_file():
            extras["hf_ref_main"] = refs.read_text(encoding="utf-8").strip()
        downloaded = _read_json(path / "download_meta.json")
        if downloaded is not None:
            extras["download_meta"] = downloaded
        listing = []
        for child in sorted(path.iterdir(), key=lambda p: p.name)[:64]:
            listing.append({"name": child.name, "is_dir": child.is_dir(), "bytes": None if child.is_dir() else child.stat().st_size})
        return {"path": str(path.resolve()), "files": files, "listing": listing, **extras}
    return {"path": str(path_str), "note": "not_a_local_path"}


def cuda_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "cuda": None,
        "gpu_name": None,
        "gpu_count": 0,
        "gpu_memory_bytes": None,
    }
    try:
        import torch

        info["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpu_count"] = int(torch.cuda.device_count())
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    info["nvidia_smi"] = _nvidia_smi_text()
    info["nvidia_compute_processes"] = _nvidia_processes()
    info["nvidia_power_clocks"] = _nvidia_power_clocks()
    info["disk"] = _disk_usage()
    return info


def git_info() -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return {"head": head, "dirty": bool(dirty.strip()),
                "tracked_dirty": any(line and not line.startswith("??") for line in dirty.splitlines()),
                "untracked_entries": sum(line.startswith("??") for line in dirty.splitlines())}
    except Exception:
        return {"head": None, "dirty": None}


def collect_versions(
    extra_modules: tuple[str, ...] = _PACKAGES,
    files_to_hash: dict[str, str] | None = None,
    include_source: bool = False,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {},
        "file_hashes": {},
        "missing": [],
    }
    info.update(cuda_environment())
    info["git"] = git_info()
    if info["git"]["head"] is None:
        info["missing"].append("git")
    for name in extra_modules:
        try:
            mod = __import__(name)
            info["packages"][name] = getattr(mod, "__version__", "present-no-version")
        except Exception:
            info["packages"][name] = None
            info["missing"].append(name)
    for label, path in (files_to_hash or {}).items():
        p = Path(path)
        if not p.is_file():
            info["file_hashes"][label] = None
            info["missing"].append(label)
            continue
        try:
            is_source = include_source and label.split("/", 1)[0] in {"grace_gc", "scripts", "configs"} and p.name != "env.sh"
            if is_source:
                raw = p.read_bytes()
                info["file_hashes"][label] = sha256_bytes(raw)
                info.setdefault("source_files", {})[label] = raw.decode("utf-8")
            else:
                info["file_hashes"][label] = sha256_file(p)
        except (OSError, UnicodeError):
            info["missing"].append(label)
    return info


def collect_environment(cfg: dict[str, Any] | None = None, files_to_hash: dict[str, str] | None = None) -> dict[str, Any]:
    hashes = dict(files_to_hash or {})
    repo = Path(__file__).resolve().parents[1]
    for folder in ("grace_gc", "scripts", "configs"):
        for path in sorted((repo / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".yml"} and path.name != "env.sh":
                hashes.setdefault(str(path.relative_to(repo)).replace("\\", "/"), str(path))
    paper = repo / "GRACE_ICLR\u8bba\u6587\u6846\u67b6_v3.md"
    if paper.is_file() and "paper" not in hashes:
        hashes["paper"] = str(paper)
    info = collect_versions(files_to_hash=hashes or None, include_source=True)
    info["process"] = {
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
    }
    info["env"] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        "HF_HOME": os.environ.get("HF_HOME"),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
        "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND"),
        "VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
        "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }
    if not cfg:
        return info
    info["model_path"] = cfg.get("model_path")
    info["data_path"] = cfg.get("data_path")
    info["model"] = describe_path(cfg.get("model_path"))
    info["data"] = describe_path(cfg.get("data_path"))
    from grace_gc.data.reward import math_verify_fns

    pred = cfg.get("predictor") or {}
    info["run_knobs"] = {
        "method": cfg.get("method"),
        "backend": cfg.get("backend"),
        "seed": cfg.get("seed"),
        "split_seed": cfg.get("split_seed"),
        "num_steps": cfg.get("num_steps"),
        "n_start": cfg.get("n_start"),
        "n_prompts": cfg.get("n_prompts"),
        "temperature": cfg.get("temperature"),
        "decision_tokens": cfg.get("decision_tokens"),
        "max_new_tokens": cfg.get("max_new_tokens"),
        "prompt_max_tokens": cfg.get("prompt_max_tokens"),
        "warmup_steps": pred.get("warmup_steps"),
        "audit_s": pred.get("audit_s"),
        "k": pred.get("k"),
        "coord_kind": pred.get("coord_kind", "mlp"),
        "use_affine": pred.get("use_affine", True),
        "shrink_m": pred.get("shrink_m", True),
        "align_basis": pred.get("align_basis", True),
        "math_verify": math_verify_fns() is not None,
        "grpo_advantage": "group_mean_no_std",
    }
    if info["data"] is None and cfg.get("data_path"):
        info["missing"].append("data_path")
    if info["model"] is None and cfg.get("model_path"):
        info["missing"].append("model_path")
    return info
