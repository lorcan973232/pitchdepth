"""Evaluate PitchDepth, its RGB baseline or ZoeDepth on one SoccerNet split.\n\nOutputs include run-level, per-frame and per-clip metric tables.\n"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from argparse import Namespace
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SoccerNetDepth
from metrics.depth_metrics import depth_metrics, fit_scale_shift
from utils.device import resolve_device, amp_autocast
from training.defaults import DEFAULT_DATA_DIR

METRIC_KEYS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "silog"]
META_KEYS   = ["frame", "clip_id", "sequence_id", "frame_filename", "depth_filename"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a depth model on SoccerNet Test.")
    p.add_argument("--model_type", type=str, default="pitchdepth",
                   choices=["pitchdepth", "depthanything", "zoedepth"])
    p.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    p.add_argument("--checkpoint", type=str, default=None, help="Trained checkpoint (.pth/.ckpt) for DA/PitchDepth")
    p.add_argument("--pretrained_weights", type=str, default=None,
                   help="Optional Depth Anything V2 weights (only used if --checkpoint is absent)")

    # Ground prior used only by PitchDepth.
    p.add_argument("--prior_path", type=str, default="priors/ground_prior_foot.npy")
    p.add_argument("--prior_norm", type=str, default="standardize", choices=["none", "minmax", "standardize"])
    p.add_argument("--prior_scale", type=float, default=1.0)
    p.add_argument("--prior_init", type=str, default="zero", choices=["zero", "avg"])

    # Dataset and runtime.
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--split", type=str, default="Test")
    p.add_argument("--sport", type=str, default="foot", choices=["foot", "basket"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers. 0 recommended on Windows to avoid spawn overhead.")
    p.add_argument("--max_frames", type=int, default=None, help="Cap #frames (debug / smoke tests)")
    p.add_argument("--infer_size", type=str, default=None,
                   help="Internal inference resolution 'H,W' (default 1078,1918). Smaller = faster, for tests only.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda"], help="GPU-only")
    p.add_argument("--precision", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                   help="Inference precision; float16/bfloat16 enable autocast on CUDA for speed.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default=None, help="Short label for the output files")
    p.add_argument("--out_dir", type=str, default="evaluation_results")
    return p.parse_args()


def _device(arg):
    return resolve_device(arg)


def _extract_clip_meta(rgb_path: str) -> dict:
    """Read clip and sequence fields from either supported dataset layout."""
    parts = os.path.normpath(rgb_path).split(os.sep)
    frame_filename = parts[-1]

    # Use the colour directory as a stable anchor in both supported layouts.
    try:
        color_idx = next(i for i, p in enumerate(parts) if p.lower() == "color")
        # Parents before colour identify the clip and optional sequence.
        above = parts[:color_idx]
        if len(above) >= 2:
            # Two parent levels represent clip and sequence identifiers.
            sequence_id = above[-1]
            clip_id     = above[-2]
        elif len(above) == 1:
            clip_id     = above[-1]
            sequence_id = ""
        else:
            clip_id = sequence_id = ""
    except StopIteration:
        # Unusual layouts fall back to the last two parent directories.
        clip_id     = parts[-3] if len(parts) >= 3 else ""
        sequence_id = parts[-2] if len(parts) >= 2 else ""

    return {
        "clip_id":        clip_id,
        "sequence_id":    sequence_id,
        "frame_filename": frame_filename,
    }


def _collect_frame_metadata(dataset) -> list[dict]:
    """Build the metadata rows that accompany per-frame metrics."""
    meta = []
    for rgb_path, depth_path in dataset.image_pairs:
        m = _extract_clip_meta(str(rgb_path))
        m["depth_filename"] = os.path.basename(str(depth_path))
        meta.append(m)
    return meta


def build_test_dataset(args):
    in_chans = 4 if args.model_type == "pitchdepth" else 3
    dataset = SoccerNetDepth(
        data_dir_path=args.data_dir,
        crop_size=(546, 966),       # Evaluation returns raw full-resolution RGB.
        sport=args.sport,
        split=args.split,
        in_chans=in_chans,
        prior_path=args.prior_path if in_chans == 4 else None,
        prior_norm=args.prior_norm,
        prior_scale=args.prior_scale,
        eval_mode=True,
    )
    if args.max_frames is not None:
        dataset.image_pairs = dataset.image_pairs[: args.max_frames]
    return dataset


def load_depth_model(args, device):
    """Build a DA/PitchDepth model and load a trained checkpoint."""
    from training.defaults import base_weights_path
    from training.model import build_model, load_trained_weights

    # Without a fine-tuned checkpoint, evaluate the pretrained weights instead.
    base_weights = None if args.checkpoint else (args.pretrained_weights or base_weights_path(args.encoder))
    model = build_model(
        args.encoder,
        in_chans=4 if args.model_type == "pitchdepth" else 3,
        base_weights=base_weights,
        prior_init=args.prior_init,
        prior_path=args.prior_path,
        prior_norm=args.prior_norm,
        prior_scale=args.prior_scale,
    )

    if args.checkpoint:
        load_trained_weights(model, args.checkpoint)
        print(f"Loaded checkpoint {args.checkpoint}")
    else:
        print("WARNING: no --checkpoint provided; evaluating un-fine-tuned weights.")

    model.to(device).eval()
    return model


@torch.no_grad()
def predict_depth_model(model, rgb, device, infer_size=None):
    """Return relative-depth predictions at the input frame size."""
    rgb = rgb.to(device, non_blocking=True)
    preds = model.infer_image(rgb, infer_size=infer_size)   # prior appended internally for 4-channel
    if preds.ndim == 2:
        preds = preds.unsqueeze(0)
    return preds


def _patch_zoedepth_model_io():
    """Relax state-dict loading in the cached ZoeDepth release."""
    hub_dir = torch.hub.get_dir()
    candidates = [d for d in os.listdir(hub_dir) if d.startswith("isl-org_ZoeDepth")]
    if not candidates:
        return False

    for repo_dir in candidates:
        model_io_path = os.path.join(hub_dir, repo_dir, "zoedepth", "models", "model_io.py")
        if not os.path.exists(model_io_path):
            continue
        # The released checkpoint contains keys omitted by newer dependencies.
        with open(model_io_path, "r") as f:
            src = f.read()
        if "strict=False" in src:
            return True
        patched = re.sub(r'(model\.load_state_dict\([^)]+)\)', r'\1, strict=False)', src)
        if patched == src:
            print(f"[ZoeDepth patch] Could not find load_state_dict call in {model_io_path} — skipping patch.")
            continue
        with open(model_io_path, "w") as f:
            f.write(patched)
        print(f"[ZoeDepth patch] Applied strict=False patch to {model_io_path}")
        return True
    return False


def load_zoedepth(device):
    """Load the zero-shot NYU ZoeDepth model through ``torch.hub``."""
    print("Loading ZoeDepth via torch.hub (isl-org/ZoeDepth)...")
    _patch_zoedepth_model_io()
    try:
        model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True, trust_repo=True)
    except RuntimeError as e:
        if "load_state_dict" in str(e) or "relative_position_index" in str(e):
            print(f"[ZoeDepth] Initial load failed ({e}). Applying patch and retrying...")
            patched = _patch_zoedepth_model_io()
            if not patched:
                raise RuntimeError(
                    "Could not patch ZoeDepth's model_io.py automatically. "
                    "Manually add strict=False to model.load_state_dict() in "
                    f"{torch.hub.get_dir()}/isl-org_ZoeDepth_*/zoedepth/models/model_io.py"
                ) from e
            model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True,
                                   trust_repo=True, force_reload=False)
        else:
            raise
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_zoedepth(model, rgb, device):
    """Run ZoeDepth on RGB and drop its singleton channel axis."""
    rgb = rgb.to(device, non_blocking=True)
    out = model.infer(rgb)
    if out.ndim == 4:
        out = out.squeeze(1)
    return out


def evaluate(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.device)
    print(f"Device: {device} | model_type: {args.model_type}")

    dataset = build_test_dataset(args)
    print(f"Evaluating on {len(dataset)} {args.sport} {args.split} frames.")
    if len(dataset) == 0:
        raise RuntimeError("No frames found. Check --data_dir / dataset structure.")

    # Capture metadata before batching and index it by absolute frame position.
    frame_meta = _collect_frame_metadata(dataset)
    print(f"[metadata] Extracted clip/sequence IDs for {len(frame_meta)} frames.")

    # Show one parsed path so layout mistakes are visible before a long run.
    if frame_meta:
        s = frame_meta[0]
        print(f"  e.g. frame 0 → clip_id='{s['clip_id']}', "
              f"sequence_id='{s['sequence_id']}', "
              f"file='{s['frame_filename']}'")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    infer_size = None
    if args.infer_size:
        infer_size = tuple(int(x) for x in args.infer_size.split(","))

    if args.model_type == "zoedepth":
        model = load_zoedepth(device)

        def predict(rgb):
            return predict_zoedepth(model, rgb, device)
    else:
        model = load_depth_model(args, device)

        def predict(rgb):
            return predict_depth_model(model, rgb, device, infer_size=infer_size)

    per_frame = []
    global_idx = 0   # tracks absolute frame index across batches

    for rgb, depth_gt, _mask in tqdm(loader, desc=f"Eval {args.model_type}"):
        depth_gt = depth_gt.to(device).squeeze(1) if depth_gt.ndim == 4 else depth_gt.to(device)
        if depth_gt.ndim == 2:
            depth_gt = depth_gt.unsqueeze(0)

        with amp_autocast(device, args.precision):
            preds = predict(rgb)
        preds = preds.float()
        if preds.shape[-2:] != depth_gt.shape[-2:]:
            preds = torch.nn.functional.interpolate(
                preds.unsqueeze(1), size=depth_gt.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)

        mask = (depth_gt > 0).float()
        scale, shift = fit_scale_shift(preds, depth_gt, mask)
        scaled = scale.view(-1, 1, 1) * preds + shift.view(-1, 1, 1)

        # Match training validation by evaluating in depth_r/65535 space.
        gt_eval   = 1 - depth_gt
        pred_eval = 1 - scaled
        batch_metrics = _per_sample_metrics(gt_eval, pred_eval, args.sport)

        # Reattach path metadata to each item in the batch.
        for i, m in enumerate(batch_metrics):
            abs_idx = global_idx + i
            meta = frame_meta[abs_idx] if abs_idx < len(frame_meta) else {}
            per_frame.append({
                "frame":          abs_idx,
                "clip_id":        meta.get("clip_id", ""),
                "sequence_id":    meta.get("sequence_id", ""),
                "frame_filename": meta.get("frame_filename", ""),
                "depth_filename": meta.get("depth_filename", ""),
                **m,
            })

        global_idx += len(batch_metrics)

    averages = {k: float(np.mean([m[k] for m in per_frame])) for k in METRIC_KEYS}
    stds     = {k: float(np.std( [m[k] for m in per_frame])) for k in METRIC_KEYS}

    print("\n=== Average metrics ===")
    for k in METRIC_KEYS:
        print(f"  {k:10s}: {averages[k]:.6f}  (+/- {stds[k]:.6f})")

    # A clip-level view helps reveal sequence-specific failures.
    _print_clip_summary(per_frame)

    _write_results(args, averages, stds, per_frame)
    return averages


def _per_sample_metrics(gt_eval, pred_eval, sport):
    """Calculate separate metrics for every frame in a batch."""
    out = []
    B = gt_eval.shape[0]
    for i in range(B):
        m = depth_metrics(gt_eval[i:i + 1], pred_eval[i:i + 1], sport=sport)
        out.append({k: float(m[k]) for k in METRIC_KEYS})
    return out


def _print_clip_summary(per_frame: list[dict]) -> None:
    """Print mean absolute-relative error for each source clip."""
    from collections import defaultdict
    clip_data: dict[str, list[float]] = defaultdict(list)
    for m in per_frame:
        key = m["clip_id"] or "unknown"
        clip_data[key].append(m["abs_rel"])

    print(f"\n=== Per-clip AbsRel summary ({len(clip_data)} clips) ===")
    for clip_id, vals in sorted(clip_data.items()):
        print(f"  {clip_id:40s}  n={len(vals):4d}  abs_rel={np.mean(vals):.6f}")


def _write_results(args, averages, stds, per_frame):
    os.makedirs(args.out_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag  = args.tag or args.model_type
    base = os.path.join(args.out_dir, f"eval_{tag}_{ts}")

    payload = {
        "timestamp":       ts,
        "config":          vars(args),
        "num_frames":      len(per_frame),
        "average_metrics": averages,
        "metric_std":      stds,
    }
    with open(base + ".json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # Store path metadata beside each frame's five metrics.
    fieldnames = META_KEYS + METRIC_KEYS
    with open(base + "_perframe.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_frame)

    # Aggregate a second table by clip identifier.
    _write_clip_csv(base + "_perclip.csv", per_frame)

    # Append one row so repeated experiments remain comparable.
    summary_path = os.path.join(args.out_dir, "summary.csv")
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "tag", "model_type", "encoder", "checkpoint"] + METRIC_KEYS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": ts, "tag": tag, "model_type": args.model_type,
            "encoder": args.encoder, "checkpoint": args.checkpoint or "",
            **averages,
        })

    print("\nResults written to:")
    print(f"  {base}.json")
    print(f"  {base}_perframe.csv  (frame-level metrics + clip/sequence IDs)")
    print(f"  {base}_perclip.csv   (clip-level aggregated metrics)")
    print(f"  {summary_path}       (cumulative summary)")


def _write_clip_csv(path: str, per_frame: list[dict]) -> None:
    """Aggregate the frame rows into one row per clip."""
    from collections import defaultdict
    clip_rows: dict[str, dict] = defaultdict(lambda: {k: [] for k in METRIC_KEYS})
    clip_seq:  dict[str, set]  = defaultdict(set)
    clip_n:    dict[str, int]  = defaultdict(int)

    for m in per_frame:
        cid = m["clip_id"] or "unknown"
        for k in METRIC_KEYS:
            clip_rows[cid][k].append(m[k])
        clip_seq[cid].add(m["sequence_id"])
        clip_n[cid] += 1

    fieldnames = ["clip_id", "sequence_ids", "n_frames"] + \
                 [f"{k}_mean" for k in METRIC_KEYS] + \
                 [f"{k}_std"  for k in METRIC_KEYS]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cid in sorted(clip_rows):
            row = {
                "clip_id":      cid,
                "sequence_ids": "|".join(sorted(clip_seq[cid])),
                "n_frames":     clip_n[cid],
            }
            for k in METRIC_KEYS:
                vals = clip_rows[cid][k]
                row[f"{k}_mean"] = float(np.mean(vals))
                row[f"{k}_std"]  = float(np.std(vals))
            writer.writerow(row)


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
