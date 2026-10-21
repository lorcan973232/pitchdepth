#!/usr/bin/env python3
"""Generate the individual image files used for Chapter 5 Figures.\n"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from argparse import Namespace

# Check dependencies before loading large checkpoints.
try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy is required (pip install numpy).")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("ERROR: matplotlib is required (pip install matplotlib).")

try:
    import torch
except ImportError:
    sys.exit("ERROR: torch is required. Run this script on the training "
             "machine environment (see repository requirements.txt).")

# Use the same dataset and alignment code as the recorded evaluation.
try:
    from data.dataset import SoccerNetDepth
    from metrics.depth_metrics import fit_scale_shift
    from utils.device import resolve_device
except ImportError as exc:
    sys.exit(f"ERROR: run from the repository root (import failed: {exc}).")

DEPTH_CMAP = "magma"
ERROR_CMAP = "seismic"
MODEL_LABEL = {
    "zoedepth": "ZoeDepth (zero-shot)",
    "depthanything": "Depth Anything (RGB-only, fine-tuned)",
    "pitchdepth": "PitchDepth (RGB + ground prior)",
}


def parse_args():
    p = argparse.ArgumentParser(description="Generate Chapter 5 Figures 5.9-5.13 "
                                            "from already trained checkpoints.")
    p.add_argument("--data_dir", type=str, default="SoccerNet/SN-Depth-2025")
    p.add_argument("--sport", type=str, default="foot", choices=["foot", "basket"])
    p.add_argument("--encoder", type=str, default="vitl")
    p.add_argument("--baseline_ckpt", type=str, required=True)
    p.add_argument("--pitchdepth_ckpt", type=str, required=True)
    p.add_argument("--zoedepth", action="store_true",
                   help="Also run zero-shot ZoeDepth if it can be loaded.")
    p.add_argument("--prior_path", type=str, default="priors/ground_prior_foot.npy")
    p.add_argument("--prior_norm", type=str, default="standardize")
    p.add_argument("--prior_scale", type=float, default=1.0)
    p.add_argument("--infer_size", type=str, default="392,672",
                   help="Internal inference resolution 'H,W'; 392,672 matches "
                        "the archived evaluation exactly.")
    p.add_argument("--frame_indices", type=int, nargs="+",
                   default=[347, 1327, 1408, 1228, 186, 51])
    p.add_argument("--profile_frames", type=int, nargs="+", default=[347, 1228],
                   help="Frames for the Figure 5.13 depth profiles.")
    p.add_argument("--out_dir", type=str, default="outputs/chapter5_figures")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scatter_points", type=int, default=40000)
    return p.parse_args()


def _load_dataset(args):
    """Load raw Test frames in order."""
    if not os.path.isdir(args.data_dir):
        sys.exit(f"ERROR: dataset directory not found: {args.data_dir}\n"
                 "Point --data_dir at the SoccerNet SN-Depth-2025 root.")
    ds = SoccerNetDepth(args.data_dir, crop_size=(546, 966), sport=args.sport,
                   split="Test", in_chans=3)
    if len(ds) == 0:
        sys.exit("ERROR: no Test frames found under the dataset directory.")
    print(f"[ok] dataset loaded: {len(ds)} Test frames")
    return ds


def _load_model(model_type, ckpt, args, device):
    """Load a trained checkpoint through the evaluation model builder."""
    from evaluate_pitchdepth import load_depth_model
    if ckpt and not os.path.exists(ckpt):
        sys.exit(f"ERROR: checkpoint not found: {ckpt}")
    ev_args = Namespace(model_type=model_type, encoder=args.encoder,
                        checkpoint=ckpt, pretrained_weights=None,
                        prior_path=args.prior_path, prior_norm=args.prior_norm,
                        prior_scale=args.prior_scale, prior_init="zero")
    model = load_depth_model(ev_args, device)
    print(f"[ok] loaded trained checkpoint for {model_type}: {ckpt}")
    return model


@torch.no_grad()
def _predict_aligned(model, model_type, rgb, gt, device, infer_size):
    """Align one prediction and return it in stored ``depth_r`` space."""
    rgb = rgb.to(device)
    if model_type == "zoedepth":
        from evaluate_pitchdepth import predict_zoedepth
        pred = predict_zoedepth(model, rgb, device)
    else:
        pred = model.infer_image(rgb, infer_size=infer_size)
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=gt.shape[-2:], mode="bilinear",
            align_corners=False).squeeze(1)
    mask = (gt > 0).float()
    scale, shift = fit_scale_shift(pred, gt, mask)
    aligned = scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)
    return (1 - aligned)[0].detach().cpu().numpy()


# The thesis imports each panel separately, so every call writes one image.
MANIFEST = []


def _record(fname, figure, frame, model, purpose):
    MANIFEST.append({"file": fname, "figure": figure, "frame": frame,
                     "model": model, "purpose": purpose})


def _save_image(arr, out_dir, fname, title, cmap=None, vmin=None, vmax=None,
                colorbar=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def _rgb_hwc(rgb_tensor):
    img = rgb_tensor.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (3, 4):
        img = np.transpose(img[:3], (1, 2, 0))
    return np.clip(img, 0, 1)


def _gt_range(gt):
    valid = gt[gt > 0]
    return (float(np.nanpercentile(valid, 2)), float(np.nanpercentile(valid, 98))) \
        if valid.size else (0.0, 1.0)


def fig_5_9(sample, model_order, out_dir):
    """Write the input, target and model panels for Figure 5.9."""
    fid = sample["frame"]
    vmin, vmax = _gt_range(sample["gt"])
    _save_image(_rgb_hwc(sample["rgb"]), out_dir,
                f"fig5_9_frame{fid:04d}_01_rgb.png",
                f"Figure 5.9 (frame {fid}): RGB input")
    _record(f"fig5_9_frame{fid:04d}_01_rgb.png", "5.9", fid, "-",
            "RGB input for qualitative comparison")
    _save_image(sample["gt"], out_dir, f"fig5_9_frame{fid:04d}_02_groundtruth.png",
                f"Figure 5.9 (frame {fid}): ground-truth depth_r",
                cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax, colorbar=True)
    _record(f"fig5_9_frame{fid:04d}_02_groundtruth.png", "5.9", fid, "ground truth",
            "Reference depth for the frame")
    order_index = {"zoedepth": 3, "depthanything": 4, "pitchdepth": 5}
    for m in model_order:
        fname = f"fig5_9_frame{fid:04d}_{order_index[m]:02d}_{m}.png"
        _save_image(sample["preds"][m], out_dir, fname,
                    f"Figure 5.9 (frame {fid}): {MODEL_LABEL[m]}",
                    cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax, colorbar=True)
        _record(fname, "5.9", fid, MODEL_LABEL[m],
                "Prediction for qualitative comparison against ground truth")


def fig_5_10(sample, out_dir):
    """Write matched-scale signed error maps for Figure 5.10."""
    fid = sample["frame"]
    gt = sample["gt"]
    valid = gt > 0
    vmax = 1e-3
    for m in ("depthanything", "pitchdepth"):
        if m in sample["preds"]:
            err = np.abs(sample["preds"][m] - gt)[valid]
            if err.size:
                vmax = max(vmax, float(np.nanpercentile(err, 98)))
    for m, short in (("depthanything", "baseline"), ("pitchdepth", "pitchdepth")):
        if m not in sample["preds"]:
            continue
        err = np.where(valid, sample["preds"][m] - gt, np.nan)
        fname = f"fig5_10_frame{fid:04d}_error_{short}.png"
        _save_image(err, out_dir, fname,
                    f"Figure 5.10 (frame {fid}): signed error, {MODEL_LABEL[m]}\n"
                    "(blue = underestimate, red = overestimate)",
                    cmap=ERROR_CMAP, vmin=-vmax, vmax=vmax, colorbar=True)
        _record(fname, "5.10", fid, MODEL_LABEL[m],
                "Signed error map on a shared symmetric scale; tests where "
                "the prior changes error spatially")


def _crop_windows(gt, kind, frac=0.30):
    """Choose the fixed pitch crop or the strongest depth-boundary crop."""
    h, w = gt.shape
    ch, cw = int(h * frac), int(w * frac)
    if kind == "pitch":
        # Match the lower image region from which the ground prior was built.
        r0 = int(h * 2 / 3) - ch // 2
        c0 = (w - cw) // 2
        return max(0, r0), c0
    # Scan coarse windows and keep the largest mean target-depth gradient.
    gy, gx = np.gradient(np.where(gt > 0, gt, np.nan))
    mag = np.nan_to_num(np.hypot(gy, gx))
    best, best_pos = -1.0, (0, 0)
    step = max(1, min(ch, cw) // 4)
    for r0 in range(0, h - ch + 1, step):
        for c0 in range(0, w - cw + 1, step):
            score = float(mag[r0:r0 + ch, c0:c0 + cw].mean())
            if score > best:
                best, best_pos = score, (r0, c0)
    return best_pos


def fig_crops(sample, out_dir, kind, fig_no):
    fid = sample["frame"]
    gt = sample["gt"]
    r0, c0 = _crop_windows(gt, kind)
    h, w = gt.shape
    ch, cw = int(h * 0.30), int(w * 0.30)
    sl = (slice(r0, r0 + ch), slice(c0, c0 + cw))
    vmin, vmax = _gt_range(gt[sl])
    tag = "pitchcrop" if kind == "pitch" else "boundarycrop"
    what = ("pitch-dominant region" if kind == "pitch"
            else "highest depth-discontinuity region (players/occlusions)")
    _save_image(gt[sl], out_dir, f"fig{fig_no}_frame{fid:04d}_{tag}_groundtruth.png",
                f"Figure {fig_no.replace('_', '.')} (frame {fid}): {what}, ground truth "
                f"[rows {r0}-{r0+ch}, cols {c0}-{c0+cw}]",
                cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax, colorbar=True)
    _record(f"fig{fig_no}_frame{fid:04d}_{tag}_groundtruth.png",
            fig_no.replace("_", "."), fid, "ground truth",
            f"Crop reference for the {what}")
    for m, short in (("depthanything", "baseline"), ("pitchdepth", "pitchdepth")):
        if m not in sample["preds"]:
            continue
        fname = f"fig{fig_no}_frame{fid:04d}_{tag}_{short}.png"
        _save_image(sample["preds"][m][sl], out_dir, fname,
                    f"Figure {fig_no.replace('_', '.')} (frame {fid}): {what}, "
                    f"{MODEL_LABEL[m]}",
                    cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax, colorbar=True)
        _record(fname, fig_no.replace("_", "."), fid, MODEL_LABEL[m],
                f"Tests model behaviour in the {what}")


def fig_5_13_profile(sample, model_order, out_dir):
    fid = sample["frame"]
    gt = sample["gt"]
    h, w = gt.shape
    col = w // 2
    rows = np.arange(h)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(rows, gt[:, col], "k-", lw=2.0, label="Ground truth")
    for m in model_order:
        ax.plot(rows, sample["preds"][m][:, col], lw=1.2, alpha=0.85,
                label=MODEL_LABEL[m])
    ax.set_xlabel("Image row (top to bottom)")
    ax.set_ylabel("Depth (aligned evaluation space)")
    ax.set_title(f"Figure 5.13 (frame {fid}): depth profile along vertical "
                 f"scanline (column {col})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f"fig5_13_frame{fid:04d}_depth_profile.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {os.path.join(out_dir, fname)}")
    _record(fname, "5.13", fid, "all models",
            "Colour-map-independent ground-plane fidelity along a scanline")


def fig_5_13_scatter(samples, model_order, out_dir, max_points, seed):
    rng = np.random.default_rng(seed)
    gt_all = np.concatenate([s["gt"][s["gt"] > 0].ravel() for s in samples])
    for m in model_order:
        pred_all = np.concatenate(
            [s["preds"][m][s["gt"] > 0].ravel() for s in samples])
        idx = rng.choice(len(gt_all), size=min(max_points, len(gt_all)),
                         replace=False)
        g, pr = gt_all[idx], pred_all[idx]
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        ax.scatter(g, pr, s=1, alpha=0.15, edgecolors="none")
        lo, hi = float(min(g.min(), pr.min())), float(max(g.max(), pr.max()))
        ax.plot([lo, hi], [lo, hi], "r-", lw=1.2, label="Identity (perfect)")
        ax.set_xlabel("Ground-truth depth (aligned space)")
        ax.set_ylabel("Predicted depth (aligned space)")
        ax.set_title(f"Figure 5.13: predicted vs ground truth, {MODEL_LABEL[m]}",
                     fontsize=9)
        ax.set_aspect("equal", "box")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fname = f"fig5_13_scatter_{m}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {os.path.join(out_dir, fname)}")
        _record(fname, "5.13", "all selected", MODEL_LABEL[m],
                "Pixel-level tightness around the identity diagonal; "
                "outlier fringe corresponds to squared-metric deterioration")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[ok] device: {device} (CUDA available: {torch.cuda.is_available()})")
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device detected; inference will be slow or "
              "may fail if the repository loader is GPU-only.")

    dataset = _load_dataset(args)
    bad = [i for i in args.frame_indices if i < 0 or i >= len(dataset)]
    if bad:
        sys.exit(f"ERROR: frame indices out of range for this split: {bad}")

    infer_size = tuple(int(x) for x in args.infer_size.split(","))

    model_order, loaded = [], {}
    if args.zoedepth:
        try:
            from evaluate_pitchdepth import load_zoedepth
            loaded["zoedepth"] = load_zoedepth(device)
            model_order.append("zoedepth")
            print("[ok] ZoeDepth loaded (zero-shot reference)")
        except Exception as exc:
            print(f"[warn] ZoeDepth unavailable ({exc}); continuing without it.")
    loaded["depthanything"] = _load_model("depthanything", args.baseline_ckpt,
                                          args, device)
    model_order.append("depthanything")
    loaded["pitchdepth"] = _load_model("pitchdepth", args.pitchdepth_ckpt,
                                       args, device)
    model_order.append("pitchdepth")

    samples = []
    for i, idx in enumerate(args.frame_indices, 1):
        print(f"[{i}/{len(args.frame_indices)}] frame {idx}: running inference...")
        rgb, depth_gt, _mask = dataset[idx]
        gt_t = depth_gt.squeeze().unsqueeze(0).to(device)
        preds = {m: _predict_aligned(loaded[m], m, rgb.unsqueeze(0), gt_t,
                                     device, infer_size) for m in model_order}
        samples.append({"frame": idx, "rgb": rgb,
                        "gt": (1 - gt_t[0]).cpu().numpy(), "preds": preds})

    for s in samples:
        fig_5_9(s, model_order, args.out_dir)
        fig_5_10(s, args.out_dir)
        fig_crops(s, args.out_dir, "pitch", "5_11")
        fig_crops(s, args.out_dir, "boundary", "5_12")
        if s["frame"] in args.profile_frames:
            fig_5_13_profile(s, model_order, args.out_dir)
    fig_5_13_scatter(samples, model_order, args.out_dir,
                     args.scatter_points, args.seed)

    manifest_path = os.path.join(args.out_dir, "figure_manifest.csv")
    with open(manifest_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["file", "figure", "frame",
                                            "model", "purpose"])
        wr.writeheader()
        wr.writerows(MANIFEST)
    print(f"[saved] {manifest_path}")
    print(f"\nDone: {len(MANIFEST)} figure files written to {args.out_dir}. "
          "No model was trained or modified by this script.")


if __name__ == "__main__":
    main()
