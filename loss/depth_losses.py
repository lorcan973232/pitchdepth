"""Training losses for relative depth.

All losses take ``(pred, target, mask)`` with shape (B, H, W) and return a
scalar. The default, ``ssitrim``, is the MiDaS loss (Ranftl et al., 2022): the
scale-and-shift-invariant trimmed mean absolute error plus 0.5 times a
multi-scale gradient-matching term. A batch with no valid pixels yields a zero
loss that still back-propagates, so a bad frame cannot break training.
"""

from __future__ import annotations

import torch

from metrics.depth_metrics import align

# MiDaS settings: discard the largest 20% of residuals; four gradient scales.
TRIM_FRACTION = 0.2
GRADIENT_SCALES = 4
GRADIENT_WEIGHT = 0.5


def _zero(like: torch.Tensor) -> torch.Tensor:
    # A zero that stays attached to the graph.
    return like.sum() * 0.0


def normalise_median_mad(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per image, subtract the median and divide by the mean absolute deviation.

    Statistics use valid pixels only, which makes the result invariant to any
    positive scale and any shift of ``x``.
    """
    out = torch.zeros_like(x)
    for i in range(x.shape[0]):
        valid = mask[i] > 0
        if valid.sum() == 0:
            continue
        values = x[i][valid]
        centred = x[i] - values.median()
        spread = centred[valid].abs().mean().clamp(min=1e-6)
        out[i] = centred / spread
    return out


def gradient_matching(residual: torch.Tensor, mask: torch.Tensor, scales: int = GRADIENT_SCALES) -> torch.Tensor:
    """Multi-scale gradient matching of a residual map, normalised by valid pixels.

    At each scale the residual is subsampled by 2^k and the absolute horizontal
    and vertical differences between valid neighbours are summed. The total is
    divided by the number of valid full-resolution pixels.
    """
    count = mask.sum()
    if count == 0:
        return _zero(residual)
    total = residual.new_zeros(())
    for level in range(scales):
        step = 2 ** level
        r = (residual * mask)[:, ::step, ::step]
        m = mask[:, ::step, ::step]
        total = total + ((r[:, :, 1:] - r[:, :, :-1]).abs() * m[:, :, 1:] * m[:, :, :-1]).sum()
        total = total + ((r[:, 1:, :] - r[:, :-1, :]).abs() * m[:, 1:, :] * m[:, :-1, :]).sum()
    return total / count


def ssi_trimmed_loss(pred, target, mask, trim: float = TRIM_FRACTION,
                     gradient_weight: float = GRADIENT_WEIGHT) -> torch.Tensor:
    """MiDaS scale-and-shift-invariant trimmed MAE plus gradient matching.

    Prediction and target are each median/MAD-normalised. Per image, the
    largest ``trim`` fraction of absolute residuals is discarded and the rest
    are summed over twice the number of valid pixels; images are then averaged.
    """
    mask = mask.float()
    if mask.sum() == 0:
        return _zero(pred)
    pred_n = normalise_median_mad(pred, mask)
    target_n = normalise_median_mad(target, mask)

    per_image = []
    for i in range(pred.shape[0]):
        valid = mask[i] > 0
        n = int(valid.sum())
        if n == 0:
            continue
        residuals = (pred_n[i][valid] - target_n[i][valid]).abs()
        kept = torch.sort(residuals).values[: max(1, int(n * (1.0 - trim)))]
        per_image.append(kept.sum() / (2 * n))
    data_term = torch.stack(per_image).mean()
    return data_term + gradient_weight * gradient_matching(pred_n - target_n, mask)


def ssi_mse_loss(pred, target, mask, gradient_weight: float = GRADIENT_WEIGHT) -> torch.Tensor:
    """MiDaS scale-and-shift-invariant MSE (least-squares alignment) plus gradient matching."""
    mask = mask.float()
    if mask.sum() == 0:
        return _zero(pred)
    residual = align(pred, target, mask) - target
    data_term = (mask * residual ** 2).sum() / (2 * mask.sum())
    return data_term + gradient_weight * gradient_matching(residual, mask)


def silog_loss(pred, target, mask, variance_focus: float = 0.5) -> torch.Tensor:
    """Scale-invariant log loss (Eigen et al., 2014), in the form Depth Anything V2 uses."""
    valid = mask.detach().bool() & (target > 0) & (pred > 0)
    if valid.sum() == 0:
        return _zero(pred)
    log_diff = torch.log(target[valid]) - torch.log(pred[valid])
    return torch.sqrt((log_diff ** 2).mean() - variance_focus * log_diff.mean() ** 2 + 1e-12)


LOSSES = {
    "ssitrim": ssi_trimmed_loss,
    "ssimse": ssi_mse_loss,
    "silog": silog_loss,
}
TRAINING_LOSSES = tuple(LOSSES)


def training_loss(name: str):
    """Return the objective that ``train.py --loss_fn name`` optimises."""
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Expected one of {TRAINING_LOSSES}.")
    return LOSSES[name]
