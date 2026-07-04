import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

_DTYPE = {
    "cuda": torch.bfloat16,
    "mps": torch.float16,
    "cpu": torch.float32,
}


def available(name: str) -> bool:
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "mps":
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    if name == "cpu":
        return True
    return False


def resolve_device(name: str = "auto") -> tuple[str, torch.dtype]:
    """Resolve "auto"/"cuda"/"mps"/"cpu" to a concrete (device, dtype)."""
    if name == "auto":
        for candidate in ("cuda", "mps", "cpu"):
            if available(candidate):
                return candidate, _DTYPE[candidate]

    if name not in _DTYPE:
        raise SystemError(f"Unknown device '{name}'. Choose from auto, cuda, mps, cpu.")

    if not available(name):
        raise SystemError(f"Requested device '{name}' is not available on this machine.")

    return name, _DTYPE[name]
