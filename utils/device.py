"""Select CUDA devices and autocast precision for PitchDepth entry points."""

from __future__ import annotations

import contextlib

import torch

# Map the public precision names to CUDA autocast data types.
PRECISION_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

_NO_CUDA_MSG = (
    "PitchDepth is configured for GPU-only execution but no CUDA device is "
    "available (torch.cuda.is_available() is False). Run on a machine with a "
    "CUDA GPU and a CUDA-enabled PyTorch build."
)


def require_cuda():
    """Stop before model setup when CUDA is unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError(_NO_CUDA_MSG)


def resolve_device(pref=None):
    """Resolve the default GPU or an explicit ``cuda:N`` device."""
    require_cuda()
    if isinstance(pref, torch.device):
        if pref.type != "cuda":
            raise RuntimeError(_NO_CUDA_MSG)
        return pref
    if pref is None or pref in ("auto", "cuda"):
        return torch.device("cuda")
    if str(pref).startswith("cuda"):
        return torch.device(pref)
    raise RuntimeError(_NO_CUDA_MSG)


def get_device():
    """Return the default CUDA device."""
    return resolve_device(None)


def amp_enabled(device, precision):
    """Return whether the selected precision needs CUDA autocast."""
    return precision in ("float16", "bfloat16")


def amp_autocast(device=None, precision="float32", enabled=None):
    """Use CUDA autocast for 16-bit runs and a no-op context for float32."""
    if enabled is None:
        enabled = amp_enabled(device, precision)
    if enabled:
        dtype = PRECISION_DTYPE.get(precision, torch.float16)
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()
