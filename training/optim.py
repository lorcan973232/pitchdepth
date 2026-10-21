"""Optimiser and learning-rate schedule for fine-tuning.

The defaults follow the Depth Anything V2 fine-tuning setup (Yang et al., 2024):
the pretrained encoder at the base learning rate, the decoder at ten times it,
and polynomial decay with power 0.9 over all optimiser steps.
"""

from __future__ import annotations

import math

import torch

OPTIMIZERS = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}
SCHEDULERS = ("poly", "constant", "cosine_warmup")
DECODER_LR_MULTIPLIER = 10.0
POLY_POWER = 0.9


def make_optimizer(model: torch.nn.Module, name: str, learning_rate: float, weight_decay: float,
                   decoder_lr_multiplier: float = DECODER_LR_MULTIPLIER) -> torch.optim.Optimizer:
    """Two parameter groups: the pretrained encoder at ``learning_rate`` and the
    decoder at ``decoder_lr_multiplier`` times that rate."""
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Expected one of {sorted(OPTIMIZERS)}.")
    encoder, decoder = [], []
    for param_name, param in model.named_parameters():
        (encoder if "pretrained" in param_name else decoder).append(param)
    groups = [
        {"params": encoder, "lr": learning_rate},
        {"params": decoder, "lr": learning_rate * decoder_lr_multiplier},
    ]
    return OPTIMIZERS[name](groups, lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))


def make_scheduler(optimizer: torch.optim.Optimizer, name: str, total_steps: int,
                   warmup_fraction: float = 0.1) -> torch.optim.lr_scheduler.LambdaLR:
    """Learning-rate multiplier per optimiser step.

    ``poly`` decays each group from its base rate as (1 - step / total) ** 0.9.
    ``constant`` keeps the base rates. ``cosine_warmup`` rises linearly over the
    first ``warmup_fraction`` of the steps, then follows a half cosine to zero.
    """
    total_steps = max(1, total_steps)
    if name == "poly":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0.0, 1.0 - step / total_steps) ** POLY_POWER)
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    if name == "cosine_warmup":
        warmup = max(1, int(warmup_fraction * total_steps))

        def multiplier(step):
            if step < warmup:
                return step / warmup
            progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    raise ValueError(f"Unknown scheduler '{name}'. Expected one of {SCHEDULERS}.")
