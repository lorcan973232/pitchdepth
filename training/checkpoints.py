"""Save and restore training state and model weights."""

from __future__ import annotations

import os

import torch


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The underlying model inside a DistributedDataParallel wrapper."""
    return model.module if hasattr(model, "module") else model


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def save_weights(path: str, model, **info) -> str:
    """Write the model weights plus any extra fields; enough for evaluation."""
    _ensure_parent(path)
    torch.save({"model": unwrap(model).state_dict(), **info}, path)
    return path


def save_training_state(path: str, model, optimizer, scheduler, epoch: int, **info) -> str:
    """Write model, optimiser and scheduler state plus any extra fields, for resuming."""
    _ensure_parent(path)
    torch.save({
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        **info,
    }, path)
    return path


def load_training_state(path: str, model, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    """Restore what ``save_training_state`` wrote and return the full record."""
    state = torch.load(path, map_location=map_location)
    unwrap(model).load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state
