"""Default paths shared by the training, evaluation and prior-generation scripts."""

import os

# Set PITCHDEPTH_DATA_DIR to use a dataset outside the repository.
DEFAULT_DATA_DIR = os.environ.get("PITCHDEPTH_DATA_DIR", "SoccerNet/SN-Depth-2025")
DEFAULT_PRIOR_PATH = "priors/ground_prior_foot.npy"
BASE_WEIGHTS_DIR = "checkpoints/DepthAnythingV2"


def base_weights_path(encoder: str) -> str:
    """Where ``scripts/get_weights.py`` saves the pretrained Depth Anything V2 weights."""
    return os.path.join(BASE_WEIGHTS_DIR, f"depth_anything_v2_{encoder}.pth")
