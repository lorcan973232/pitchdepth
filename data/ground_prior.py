"""Build and load PitchDepth's fixed fourth input channel.

The prior is the mean training depth map with its upper third set to zero.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data.discovery import find_rgb_depth

# The generated prior is small and retained.
DEFAULT_PRIOR_DIR = "priors"
DEFAULT_PRIOR_NAME = "ground_prior_foot"

# Supported transformations from the stored prior to a model input channel.
VALID_PRIOR_NORMS = ("none", "minmax", "standardize")


@dataclass
class PriorMeta:
    """Values needed to reproduce the prior channel at inference time."""

    sport: str
    split: str
    depth_space: str
    crop_fraction: float
    num_frames: int
    height: int
    width: int
    # Statistics describe the retained lower region before runtime normalisation.
    valid_mean: float
    valid_std: float
    valid_min: float
    valid_max: float
    data_dir: str

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "PriorMeta":
        with open(path) as f:
            return PriorMeta(**json.load(f))


def _depth_to_model_space(depth_uint16: np.ndarray, depth_space: str) -> np.ndarray:
    """Convert a 16-bit label to raw or inverted model depth."""
    d = depth_uint16.astype(np.float64) / 65535.0
    if depth_space == "model":
        return 1.0 - d
    if depth_space == "raw":
        return d
    raise ValueError(f"Unknown depth_space '{depth_space}' (expected 'model' or 'raw').")


def generate_ground_prior(
    data_dir: str,
    split: str = "Train",
    sport: str = "foot",
    crop_fraction: float = 2.0 / 3.0,
    depth_space: str = "model",
    out_dir: str = DEFAULT_PRIOR_DIR,
    name: Optional[str] = None,
    max_frames: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, PriorMeta, str]:
    """Average one labelled split and save the resulting fixed prior."""
    if name is None:
        name = f"ground_prior_{sport}"

    # Training labels are the intended source; other splits are useful for tests.
    pairs = find_rgb_depth(data_dir, split=split, sport=sport)
    if len(pairs) == 0:
        raise FileNotFoundError(
            f"No depth maps found under '{data_dir}' for split='{split}', sport='{sport}'. "
            "Check the dataset path / structure."
        )
    if max_frames is not None:
        pairs = pairs[:max_frames]

    # Use the first label's resolution throughout the average.
    first_depth = np.array(Image.open(pairs[0][1]))
    height, width = first_depth.shape[:2]

    accum = np.zeros((height, width), dtype=np.float64)
    count = 0
    for i, (_rgb_path, depth_path) in enumerate(pairs):
        depth = np.array(Image.open(depth_path))
        if depth.ndim != 2:
            depth = depth[..., 0]
        if depth.shape[:2] != (height, width):
            depth_t = torch.from_numpy(depth.astype(np.float32))[None, None]
            depth_t = F.interpolate(depth_t, size=(height, width), mode="nearest")
            depth = depth_t[0, 0].numpy()
        accum += _depth_to_model_space(depth, depth_space)
        count += 1
        if verbose and (i + 1) % 250 == 0:
            print(f"  [ground_prior] processed {i + 1}/{len(pairs)} depth maps")

    mean_full = (accum / count).astype(np.float32)

    # The lower portion contains the pitch in the training broadcasts.
    crop_fraction = float(np.clip(crop_fraction, 0.0, 1.0))
    top_row = int(round(height * (1.0 - crop_fraction)))
    prior = mean_full.copy()
    if top_row > 0:
        prior[:top_row, :] = 0.0

    valid = prior[top_row:, :]
    meta = PriorMeta(
        sport=sport,
        split=split,
        depth_space=depth_space,
        crop_fraction=crop_fraction,
        num_frames=count,
        height=height,
        width=width,
        valid_mean=float(valid.mean()),
        valid_std=float(valid.std() + 1e-12),
        valid_min=float(valid.min()),
        valid_max=float(valid.max()),
        # Recorded as given, so a relative path keeps machine paths out of the metadata.
        data_dir=str(data_dir),
    )

    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, f"{name}.npy")
    png_path = os.path.join(out_dir, f"{name}.png")
    meta_path = os.path.join(out_dir, f"{name}.json")

    np.save(npy_path, prior)
    meta.to_json(meta_path)
    _save_prior_preview(prior, png_path)

    if verbose:
        print(
            f"[ground_prior] saved prior to {npy_path}\n"
            f"               averaged {count} frames, shape={prior.shape}, "
            f"crop_fraction={crop_fraction:.3f}, depth_space={depth_space}\n"
            f"               valid region mean={meta.valid_mean:.5f} std={meta.valid_std:.5f}"
        )
    return prior, meta, npy_path


def _save_prior_preview(prior: np.ndarray, png_path: str) -> None:
    """Save the colour preview distributed with the number value prior."""
    try:
        import matplotlib

        matplotlib.use("Agg")

        p = prior.astype(np.float32)
        valid = p[p != 0]
        if valid.size == 0:
            norm = np.zeros_like(p)
        else:
            lo, hi = float(valid.min()), float(valid.max())
            norm = (p - lo) / (hi - lo + 1e-8)
            norm = np.clip(norm, 0.0, 1.0)
        rgba = matplotlib.colormaps["Spectral_r"](norm)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        Image.fromarray(rgb).save(png_path)
    except Exception as exc:
        print(f"[ground_prior] could not write preview PNG: {exc}")


def load_ground_prior(path: str) -> torch.Tensor:
    """Load a saved prior as a ``(1, H, W)`` float tensor."""
    npy_path = _resolve_prior_path(path)
    arr = np.load(npy_path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    return torch.from_numpy(arr)


def load_ground_prior_meta(path: str) -> Optional[PriorMeta]:
    npy_path = _resolve_prior_path(path)
    meta_path = os.path.splitext(npy_path)[0] + ".json"
    if os.path.exists(meta_path):
        return PriorMeta.from_json(meta_path)
    return None


def _resolve_prior_path(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, f"{DEFAULT_PRIOR_NAME}.npy")
    if path.endswith(".npy"):
        return path
    if os.path.exists(path + ".npy"):
        return path + ".npy"
    return path


def prepare_prior_channel(
    prior: torch.Tensor,
    size: Tuple[int, int],
    norm: str = "standardize",
    scale: float = 1.0,
    meta: Optional[PriorMeta] = None,
) -> torch.Tensor:
    """Resize the saved prior and apply the configured input scaling."""
    if norm not in VALID_PRIOR_NORMS:
        raise ValueError(f"Unknown prior norm '{norm}'. Expected one of {VALID_PRIOR_NORMS}.")

    if prior.ndim == 2:
        prior = prior.unsqueeze(0)
    prior = prior.float()

    # The prior is a smooth average, so it is resized using nearby values rather than as a label.
    resized = F.interpolate(prior.unsqueeze(0), size=size, mode="bilinear", align_corners=False)
    resized = resized.squeeze(0)  # (1, h, w)

    if norm == "none":
        out = resized
    elif norm == "minmax":
        # Saved training statistics keep this transform fixed across all frames.
        if meta is not None:
            lo, hi = meta.valid_min, meta.valid_max
        else:
            valid = resized[resized != 0]
            lo = float(valid.min()) if valid.numel() else 0.0
            hi = float(valid.max()) if valid.numel() else 1.0
        out = (resized - lo) / (hi - lo + 1e-8)
    else:
        if meta is not None:
            mean, std = meta.valid_mean, meta.valid_std
        else:
            valid = resized[resized != 0]
            mean = float(valid.mean()) if valid.numel() else 0.0
            std = float(valid.std() + 1e-12) if valid.numel() else 1.0
        out = (resized - mean) / std

    return out * float(scale)


def main():
    import argparse
    from training.defaults import DEFAULT_DATA_DIR

    parser = argparse.ArgumentParser(description="Build the PitchDepth ground prior.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", type=str, default="Train")
    parser.add_argument("--sport", type=str, default="foot", choices=["foot", "basket"])
    parser.add_argument("--crop_fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--depth_space", type=str, default="model", choices=["model", "raw"])
    parser.add_argument("--out_dir", type=str, default=DEFAULT_PRIOR_DIR)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    generate_ground_prior(
        data_dir=args.data_dir,
        split=args.split,
        sport=args.sport,
        crop_fraction=args.crop_fraction,
        depth_space=args.depth_space,
        out_dir=args.out_dir,
        name=args.name,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
