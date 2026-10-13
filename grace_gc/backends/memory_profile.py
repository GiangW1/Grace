"""Opt-in capacity diagnostics, separate from scientific training results."""

from contextlib import contextmanager
import csv
import os
import shutil
import subprocess
from time import perf_counter

import numpy as np


class ActorMemoryProfile:
    def __init__(self, run):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("memory_profile requires a real CUDA actor")
        self.run = run
        self.rows = []
        self.current = None
        self.monitor = self.stream = None
        self.report = {
            "status": "running", "actor_logical_device": 0,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "actor_name": torch.cuda.get_device_name(0),
            "actor_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "phases": self.rows,
            "scope": "Actor allocator peaks include live base weights, not other processes. "
                     "nvidia-smi is sampled board-wide memory, not an exact peak or per-process measurement. "
                     "Profiling synchronizes CUDA and is not a throughput benchmark.",
        }
        torch.cuda.reset_peak_memory_stats(0)
        executable = shutil.which("nvidia-smi")
        self.report["board_monitor"] = "unavailable"
        if executable:
            self.stream = (run.root / "nvidia_smi.csv").open("w")
            try:
                self.monitor = subprocess.Popen([
                    executable,
                    "--query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits", "-lms", "500",
                ], stdout=self.stream, stderr=subprocess.DEVNULL)
                self.report["board_monitor"] = "sampling_all_boards_every_500ms"
            except OSError as exc:
                self.report["board_monitor"] = str(exc)
                self.stream.close()
                self.stream = None
        self.save()

    def save(self):
        self.run.write_json("memory_profile.json", self.report)

    def observe_sequence(self, ids, prompt_len):
        if self.current is not None:
            self.current["logprob_calls"] = self.current.get("logprob_calls", 0) + 1
            self.current["max_sequence_tokens"] = max(self.current.get("max_sequence_tokens", 0), len(ids))
            self.current["max_response_tokens"] = max(self.current.get("max_response_tokens", 0),
                                                       len(ids) - int(prompt_len))

    def snapshot(self, row):
        import torch

        try:
            row.update(peak_allocated_gib=torch.cuda.max_memory_allocated(0) / 2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved(0) / 2**30,
                       allocated_gib=torch.cuda.memory_allocated(0) / 2**30,
                       reserved_gib=torch.cuda.memory_reserved(0) / 2**30)
        except Exception as exc:
            row["measurement_error"] = str(exc)

    @contextmanager
    def phase(self, name, **metadata):
        import torch

        if not self.rows:
            setup = {"phase": "setup_before_first_sync"}
            self.snapshot(setup)
            self.rows.append(setup)
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        row = {"phase": name, **metadata, "status": "running"}
        self.current = row
        self.rows.append(row)
        self.save()
        start = perf_counter()
        try:
            yield row
            torch.cuda.synchronize(0)
            row["status"] = "completed"
        except BaseException as exc:
            row.update(status="out_of_memory" if isinstance(exc, torch.cuda.OutOfMemoryError) else "failed",
                       error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            row["wall_seconds"] = perf_counter() - start
            self.snapshot(row)
            self.current = None
            self.save()

    def close(self, status):
        if self.monitor is not None:
            self.monitor.terminate()
            try:
                self.monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.monitor.kill()
                self.monitor.wait(timeout=5)
        if self.stream is not None:
            self.stream.close()
        path = self.run.root / "nvidia_smi.csv"
        boards = {}
        if path.exists():
            for fields in csv.reader(path.read_text().splitlines()):
                if len(fields) != 7:
                    continue
                stamp, index, uuid, name, used, total, utilization = [x.strip() for x in fields]
                try:
                    used, total = float(used), float(total)
                except ValueError:
                    continue
                board = boards.setdefault(uuid, {"index": index, "name": name,
                                                 "total_mib": total, "sampled_peak_used_mib": 0., "samples": 0})
                board["sampled_peak_used_mib"] = max(board["sampled_peak_used_mib"], used)
                board["samples"] += 1
        self.report.update(status=status, boards=boards)
        terminal = {"phase": "terminal"}
        self.snapshot(terminal)
        self.rows.append(terminal)
        self.save()


def full_length_pg_probe(actor, tokenizer, cfg):
    """One fixed-length synthetic trajectory through the real Full-PG gradient path.

    No reward claims, optimizer step, actor update, or training RNG consumption.
    Real rollout/optimizer memory is measured separately in subsequent steps.
    """
    import torch
    from grace_gc.backends.hf_actor import logprob_one, named_lora_params
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.trainer.actor_update import real_stream_backward_and_audit

    if cfg.get("method") != "full_pg":
        raise ValueError("full_length_probe is restricted to Full-PG capacity tests")
    prompt_len = int(cfg.get("prompt_max_tokens", 1024))
    response_len = int(cfg["max_new_tokens"])
    if prompt_len <= 0 or response_len <= 0:
        raise ValueError("capacity probe requires positive prompt and response lengths")
    special = set(getattr(tokenizer, "all_special_ids", []))
    pattern = [int(t) for t in tokenizer.encode("Let us solve this problem step by step. 1 + 1 = 2. ",
                                               add_special_tokens=False) if int(t) not in special]
    if not pattern:
        raise ValueError("tokenizer produced no ordinary tokens for the capacity probe")
    total = prompt_len + response_len
    ids = (pattern * ((total + len(pattern) - 1) // len(pattern)))[:total]
    named = named_lora_params(actor)
    saved = [p.grad for _, p in named]
    device = next(actor.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        for _, param in named:
            param.grad = None
        with torch.random.fork_rng(devices=devices):
            loss, _ = real_stream_backward_and_audit(
                lambda i: logprob_one(actor, ids, prompt_len, int(tokenizer.pad_token_id or 0), eos_id=None),
                np.array([0.5]), np.ones(1), np.ones(1), 1, np.ones(1, dtype=bool),
                named, collect_lora_layout(named), np.zeros(1, dtype=bool),
            )
        finite = bool(torch.isfinite(loss).item()) and all(
            p.grad is None or bool(torch.isfinite(p.grad).all().item()) for _, p in named)
        result = {"synthetic": True, "prompt_tokens": prompt_len, "response_tokens": response_len,
                  "sequence_tokens": len(ids), "all_finite": finite,
                  "trainable_dim": sum(p.numel() for _, p in named), "optimizer_updated": False}
        if not finite:
            raise FloatingPointError("nonfinite Full-PG capacity probe loss/gradient")
        return result
    finally:
        for (_, param), grad in zip(named, saved):
            param.grad = grad
