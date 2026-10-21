"""Create a tiny SoccerNet-style dataset for CPU smoke tests.\n\nSynthetic depth increases down the frame so prior tests have a known gradient.\n"""

from __future__ import annotations

import os
import numpy as np
from PIL import Image


def make_synthetic_dataset(root, splits=("Train", "Test"), n_per_split=4, h=140, w=252, seed=0,
                           layout="category"):
    """Write either the sport-folder or flat SoccerNet layout."""
    rng = np.random.default_rng(seed)
    for split in splits:
        if layout == "flat":
            vdir = os.path.join(root, split, "game_1", "video_1")
        else:
            vdir = os.path.join(root, "depth-football", split, "game_1", "video_1")
        # Match the held-out challenge split by leaving out depth labels.
        has_depth = split.lower() != "challenge"
        color_dir = os.path.join(vdir, "color")
        depth_dir = os.path.join(vdir, "depth_r")
        os.makedirs(color_dir, exist_ok=True)
        if has_depth:
            os.makedirs(depth_dir, exist_ok=True)

        # A vertical gradient provides predictable far-to-near depth values.
        grad = np.linspace(0.2, 0.9, h, dtype=np.float32)[:, None].repeat(w, axis=1)
        for i in range(n_per_split):
            # Add light frame noise to a green pitch-like RGB image.
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            rgb[..., 1] = 0.5 + 0.1 * rng.standard_normal((h, w))
            rgb[..., 0] = 0.3 + 0.1 * rng.standard_normal((h, w))
            rgb[..., 2] = 0.2 + 0.1 * rng.standard_normal((h, w))
            rgb = np.clip(rgb, 0, 1)
            Image.fromarray((rgb * 255).astype(np.uint8)).save(
                os.path.join(color_dir, f"color_{i:04d}.png"))

            if has_depth:
                depth = grad + 0.03 * rng.standard_normal((h, w)).astype(np.float32)
                depth = np.clip(depth, 0, 1)
                depth_u16 = (depth * 65535).astype(np.uint16)
                Image.fromarray(depth_u16).save(os.path.join(depth_dir, f"color_{i:04d}.png"))
    return root


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_data"
    make_synthetic_dataset(out)
    print(f"Synthetic dataset written to {out}")
