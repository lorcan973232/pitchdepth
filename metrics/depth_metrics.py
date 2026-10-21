"""Least-squares alignment and the five depth errors used to score predictions.

Relative-depth predictions have no fixed scale or offset, so each prediction is
aligned to its target by a per-image least-squares scale and shift before it is
scored. The five errors and the overlay exclusions follow the SoccerNet-Depth
evaluation protocol (Leduc et al., 2024). Every error is lower-is-better.
"""

from __future__ import annotations

import numpy as np
import torch

METRIC_NAMES = ("abs_rel", "sq_rel", "rmse", "rmse_log", "silog")

# Pixel boxes covered by broadcast overlays: the football scoreboard is
# optional, the basketball overlay is always excluded.
FOOTBALL_SCOREBOARD = (slice(70, 122), slice(95, 612))
BASKETBALL_OVERLAY = (slice(870, 1016), slice(1570, 1829))


def fit_scale_shift(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Per-image scale ``s`` and shift ``t`` minimising sum(mask * (s * pred + t - target)^2).

    All inputs have shape (B, H, W). Returns two (B,) tensors. An image whose
    normal equations are singular, for example one with no valid pixels, gets
    s = t = 0.
    """
    dims = (1, 2)
    n = mask.sum(dims)
    sum_p = (mask * pred).sum(dims)
    sum_pp = (mask * pred * pred).sum(dims)
    sum_t = (mask * target).sum(dims)
    sum_pt = (mask * pred * target).sum(dims)

    det = sum_pp * n - sum_p * sum_p
    solvable = det > 0
    safe_det = torch.where(solvable, det, torch.ones_like(det))
    scale = torch.where(solvable, (n * sum_pt - sum_p * sum_t) / safe_det, torch.zeros_like(det))
    shift = torch.where(solvable, (sum_pp * sum_t - sum_p * sum_pt) / safe_det, torch.zeros_like(det))
    return scale, shift


def align(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply the least-squares scale and shift to ``pred``."""
    scale, shift = fit_scale_shift(pred, target, mask)
    return scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)


def _errors(gt: np.ndarray, pred: np.ndarray) -> dict:
    """The five errors over the valid pixels of one image (1-D arrays)."""
    diff = gt - pred
    log_ratio = np.log(pred) - np.log(gt)
    return {
        "abs_rel": np.mean(np.abs(diff) / gt),
        "sq_rel": np.mean(diff ** 2 / gt),
        "rmse": np.sqrt(np.mean(diff ** 2)),
        "rmse_log": np.sqrt(np.mean(log_ratio ** 2)),
        # Scale-invariant log error, reported multiplied by 100.
        "silog": 100 * np.sqrt(np.mean(log_ratio ** 2) - np.mean(log_ratio) ** 2),
    }


def depth_metrics(gt: torch.Tensor, pred: torch.Tensor, sport: str = "foot",
                  mask_scoreboard: bool = False) -> dict:
    """Score a (B, H, W) batch; each error is averaged over the images.

    Pixels where either map is non-positive or non-finite are skipped, as are
    the overlay boxes for the sport.
    """
    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    per_image = []
    for g, p in zip(gt_np, pred_np):
        valid = np.isfinite(g) & (g > 0) & np.isfinite(p) & (p > 0)
        if sport == "basket":
            valid[BASKETBALL_OVERLAY] = False
        elif sport == "foot" and mask_scoreboard:
            valid[FOOTBALL_SCOREBOARD] = False
        per_image.append(_errors(g[valid], p[valid]))
    return {name: np.mean([errors[name] for errors in per_image]) for name in METRIC_NAMES}


def metric_average(metrics: dict) -> float:
    """Unweighted mean of the five errors, used to select checkpoints."""
    return sum(metrics[name] for name in METRIC_NAMES) / len(METRIC_NAMES)


class MetricMeans:
    """Average metric dictionaries over many batches."""

    def __init__(self):
        self.totals: dict = {}
        self.count = 0

    def add(self, metrics: dict) -> None:
        for name, value in metrics.items():
            self.totals[name] = self.totals.get(name, 0.0) + value
        self.count += 1

    def means(self) -> dict:
        return {name: total / self.count for name, total in self.totals.items()}
