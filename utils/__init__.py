"""Expose device and mixed-precision helpers used by project entry points."""

from utils.device import (
    resolve_device,
    get_device,
    require_cuda,
    amp_autocast,
    amp_enabled,
    PRECISION_DTYPE,
)

__all__ = ["resolve_device", "get_device", "require_cuda", "amp_autocast", "amp_enabled", "PRECISION_DTYPE"]
