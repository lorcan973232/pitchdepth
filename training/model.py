"""Build the RGB baseline or the four-channel PitchDepth model."""

from __future__ import annotations

import logging
import os

import torch
from torch.utils.checkpoint import checkpoint

from data.ground_prior import load_ground_prior, load_ground_prior_meta
from models.DAv2.depth_anything_v2.dpt import DepthAnythingV2, inflate_patch_embed_weight

logger = logging.getLogger("pitchdepth")

# Decoder sizes published with each Depth Anything V2 encoder.
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def _state_dict_from(obj) -> dict:
    """Accept a bare state dict or one wrapped under 'model' or 'state_dict'."""
    for key in ("model", "state_dict"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            return obj[key]
    return obj


def _strip_wrapper_prefixes(state: dict) -> dict:
    # DistributedDataParallel and torch.compile prefix every key.
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def attach_prior(model: DepthAnythingV2, prior_path: str, norm: str = "standardize",
                 scale: float = 1.0) -> None:
    """Give a four-channel model the saved prior and its normalisation statistics."""
    if not prior_path:
        raise ValueError("A four-channel model needs prior_path to point at a generated prior (.npy).")
    meta = load_ground_prior_meta(prior_path)
    stats = None if meta is None else (meta.valid_mean, meta.valid_std, meta.valid_min, meta.valid_max)
    model.set_ground_prior(load_ground_prior(prior_path), norm=norm, scale=scale, stats=stats)


def build_model(encoder: str = "vitl", in_chans: int = 3, base_weights: str | None = None,
                load_head: bool = True, prior_init: str = "zero", prior_path: str | None = None,
                prior_norm: str = "standardize", prior_scale: float = 1.0) -> DepthAnythingV2:
    """Create the model and optionally start it from pretrained Depth Anything V2 weights.

    With ``load_head=False`` only the encoder weights are loaded. For four
    input channels the pretrained patch projection gains one kernel,
    initialised to zero (``prior_init="zero"``) or to the mean RGB kernel
    (``"avg"``), and the prior is attached.
    """
    model = DepthAnythingV2(in_chans=in_chans, **MODEL_CONFIGS[encoder])

    if base_weights is not None:
        if not os.path.exists(base_weights):
            raise FileNotFoundError(
                f"Pretrained weights not found at {base_weights}. Run scripts/get_weights.py first.")
        state = _state_dict_from(torch.load(base_weights, map_location="cpu"))
        if not load_head:
            state = {k: v for k, v in state.items() if k.startswith("pretrained.")}
        state = inflate_patch_embed_weight(state, in_chans, init=prior_init)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Loaded {base_weights}: {len(missing)} missing and {len(unexpected)} unexpected keys.")

    if in_chans >= 4:
        attach_prior(model, prior_path, norm=prior_norm, scale=prior_scale)
    return model


def load_trained_weights(model: DepthAnythingV2, path: str) -> None:
    """Load a checkpoint written by ``train.py`` (or a bare state dict)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = _strip_wrapper_prefixes(_state_dict_from(torch.load(path, map_location="cpu")))
    model.load_state_dict(state, strict=True)


def enable_gradient_checkpointing(model: DepthAnythingV2) -> int:
    """Recompute each encoder block's activations during the backward pass.

    Each block's ``forward`` is wrapped in place, so parameter names and
    checkpoints are unchanged. Returns the number of blocks wrapped.
    """
    blocks = model.pretrained.blocks
    for block in blocks:
        plain_forward = block.forward

        def checkpointed_forward(x, _forward=plain_forward):
            if torch.is_grad_enabled():
                return checkpoint(_forward, x, use_reentrant=False)
            return _forward(x)

        block.forward = checkpointed_forward
    return len(blocks)
