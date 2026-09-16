"""GPU backends. Importing this package does not import vLLM/verl."""

from __future__ import annotations


def require_gpu_stack() -> dict:
    missing = []
    versions = {}
    try:
        import torch

        versions["torch"] = torch.__version__
        if not torch.cuda.is_available():
            missing.append("cuda")
    except Exception:
        missing.append("torch")
    try:
        import vllm

        versions["vllm"] = getattr(vllm, "__version__", "present")
    except Exception:
        missing.append("vllm")
    try:
        import verl

        versions["verl"] = getattr(verl, "__version__", "present")
    except Exception:
        versions["verl"] = None
    if missing:
        raise ImportError(
            "GPU stack is unavailable: missing "
            + ", ".join(missing)
            + ". Install the same vLLM/PyTorch revision planned for the server."
        )
    return versions
