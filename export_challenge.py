"""Write 16-bit depth predictions for unlabelled SoccerNet-Depth challenge frames.

Each frame under ``game_<g>/video_<v>/color/`` produces
``foot_game_<g>_video_<v>_depth_r_<frame>.png`` in the output folder, the file
name the challenge server expects. Predictions are rescaled to the full 16-bit
range and inverted back to depth_r space. ``--flip`` averages each prediction
with that of the mirrored frame.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from training.defaults import DEFAULT_PRIOR_PATH
from training.model import MODEL_CONFIGS, build_model, load_trained_weights
from utils.device import amp_autocast, resolve_device


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Export challenge predictions as 16-bit PNGs.")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--input_dir", required=True, help="Challenge split folder, or a folder above it.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--encoder", default="vitl", choices=sorted(MODEL_CONFIGS))
    p.add_argument("--in_chans", type=int, default=3, choices=[3, 4])
    p.add_argument("--prior_path", default=DEFAULT_PRIOR_PATH)
    p.add_argument("--prior_norm", default="standardize", choices=["none", "minmax", "standardize"])
    p.add_argument("--prior_scale", type=float, default=1.0)
    p.add_argument("--infer_size", default=None, help="Network resolution as height,width (default: largest patch-aligned size).")
    p.add_argument("--flip", action="store_true", help="Average with the prediction for the mirrored frame.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--precision", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def challenge_frames(input_dir: str) -> list[tuple[str, str]]:
    """List (rgb_path, output_name) for every colour frame in a game/video tree."""
    frames = []
    for folder, subfolders, files in os.walk(input_dir):
        subfolders.sort()
        if os.path.basename(folder).lower() != "color":
            continue
        video = re.search(r"video_(\d+)", os.path.basename(os.path.dirname(folder)))
        game = re.search(r"game_(\d+)", os.path.basename(os.path.dirname(os.path.dirname(folder))))
        if not (video and game):
            continue
        for name in sorted(files):
            if name.startswith("._") or not name.lower().endswith(".png"):
                continue
            number = re.findall(r"\d+", os.path.splitext(name)[0])
            if number:
                out_name = f"foot_game_{game.group(1)}_video_{video.group(1)}_depth_r_{int(number[-1])}.png"
                frames.append((os.path.join(folder, name), out_name))
    if not frames:
        raise SystemExit(f"No game_*/video_*/color/*.png frames found under {input_dir}.")
    return frames


def to_depth_r_png(pred: np.ndarray) -> Image.Image:
    """Rescale a relative-depth map to [0, 1], invert it and store it as 16 bits."""
    span = pred.max() - pred.min()
    unit = (pred - pred.min()) / span if span > 0 else np.zeros_like(pred)
    return Image.fromarray(((1.0 - unit) * 65535.0).astype(np.uint16))


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    infer_size = tuple(int(v) for v in args.infer_size.split(",")) if args.infer_size else None

    model = build_model(args.encoder, args.in_chans, prior_path=args.prior_path,
                        prior_norm=args.prior_norm, prior_scale=args.prior_scale)
    load_trained_weights(model, args.checkpoint_path)
    model.to(device).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    frames = challenge_frames(args.input_dir)
    for start in tqdm(range(0, len(frames), args.batch_size), desc="Exporting"):
        chunk = frames[start:start + args.batch_size]
        rgb = torch.stack([
            torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
            for path, _ in chunk
        ]).to(device)
        with amp_autocast(device, args.precision):
            depth = model.infer_image(rgb, infer_size=infer_size)
            if args.flip:
                mirrored = model.infer_image(torch.flip(rgb, dims=[-1]), infer_size=infer_size)
                depth = (depth + torch.flip(mirrored, dims=[-1])) / 2
        for (_, out_name), pred in zip(chunk, depth.float().cpu().numpy()):
            to_depth_r_png(pred).save(os.path.join(args.output_dir, out_name))
    print(f"Wrote {len(frames)} predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
