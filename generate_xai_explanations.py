#!/usr/bin/env python3
"""Generate the Chapter 5 Grad-CAM, SmoothGrad and LIME figures.\n\nThe script explains existing baseline and PitchDepth checkpoints; it never trains them.\n"""
from __future__ import annotations

import argparse
import csv
import os
import sys

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    sys.exit(f"ERROR: numpy/matplotlib required ({exc}).")

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("ERROR: torch is required. Run on the training machine "
             "environment (see repository requirements.txt).")

try:
    from data.dataset import SoccerNetDepth
    from utils.device import resolve_device
except ImportError as exc:
    sys.exit(f"ERROR: run from the repository root (import failed: {exc}).")

MODEL_LABEL = {"depthanything": "Depth Anything (RGB-only, fine-tuned)",
               "pitchdepth": "PitchDepth (RGB + ground prior)"}
MANIFEST = []


def parse_args():
    p = argparse.ArgumentParser(description="XAI evidence (Grad-CAM style, "
                                            "SmoothGrad, optional LIME) from "
                                            "trained checkpoints.")
    p.add_argument("--data_dir", type=str, default="SoccerNet/SN-Depth-2025")
    p.add_argument("--sport", type=str, default="foot")
    p.add_argument("--encoder", type=str, default="vitl")
    p.add_argument("--baseline_ckpt", type=str, required=True)
    p.add_argument("--pitchdepth_ckpt", type=str, required=True)
    p.add_argument("--prior_path", type=str, default="priors/ground_prior_foot.npy")
    p.add_argument("--prior_norm", type=str, default="standardize")
    p.add_argument("--prior_scale", type=float, default=1.0)
    p.add_argument("--infer_size", type=str, default="392,672")
    p.add_argument("--frame_indices", type=int, nargs="+", default=[347, 1228])
    p.add_argument("--sg_samples", type=int, default=16)
    p.add_argument("--sg_sigma", type=float, default=0.08)
    p.add_argument("--out_dir", type=str, default="outputs/chapter5_figures/xai")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lime_samples", type=int, default=512,
                   help="Number of LIME perturbation samples. More = smoother but slower.")
    p.add_argument("--lime_features", type=int, default=10,
                   help="Max superpixel features to highlight in the LIME explanation.")
    return p.parse_args()


def _load_model(model_type, ckpt, args, device):
    """Load a trained model through the same path as the evaluator."""
    from argparse import Namespace
    from evaluate_pitchdepth import load_depth_model
    if not os.path.exists(ckpt):
        sys.exit(f"ERROR: checkpoint not found: {ckpt}")
    ev_args = Namespace(model_type=model_type, encoder=args.encoder,
                        checkpoint=ckpt, pretrained_weights=None,
                        prior_path=args.prior_path, prior_norm=args.prior_norm,
                        prior_scale=args.prior_scale, prior_init="zero")
    m = load_depth_model(ev_args, device)
    m.eval()
    return m


def _load_prior(prior_path, prior_norm, prior_scale, device):
    """Prepare the stored prior as a batched model input."""
    if not os.path.exists(prior_path):
        sys.exit(f"ERROR: prior not found: {prior_path}")
    prior = np.load(prior_path).astype(np.float32)
    if prior_norm == "standardize":
        mu, sigma = prior.mean(), prior.std()
        prior = (prior - mu) / (sigma + 1e-8)
    elif prior_norm == "minmax":
        lo, hi = prior.min(), prior.max()
        prior = (prior - lo) / (hi - lo + 1e-8)
    prior = prior * prior_scale
    return torch.from_numpy(prior).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)


def _make_input(x_rgb, mname, prior_tensor, infer_size):
    """Resize an RGB frame and append the prior for PitchDepth."""
    x = F.interpolate(x_rgb, size=infer_size, mode="bilinear", align_corners=False)
    if mname == "pitchdepth":
        # Resize the stored prior to the exact model input geometry.
        prior = F.interpolate(prior_tensor, size=infer_size,
                              mode="bilinear", align_corners=False)
        x = torch.cat([x, prior], dim=1)   # (1, 4, H, W)
    return x


def _regions(h, w):
    """Return the pitch window also used for Figures 5.24–5.29."""
    ch, cw = int(h * 0.30), int(w * 0.30)
    pitch = (slice(int(h * 2 / 3) - ch // 2, int(h * 2 / 3) + ch // 2),
             slice((w - cw) // 2, (w + cw) // 2))
    return {"pitchregion": pitch}


def _forward_depth(model, x_prepared):
    """Keep the model call explicit for the gradient-based methods."""
    return model(x_prepared)


def _save_heat(heat, rgb_hwc, out_dir, fname, title):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.imshow(rgb_hwc)
    im = ax.imshow(heat, cmap="inferno", alpha=0.55)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", fname)


def smoothgrad(model, x_rgb, mname, prior_tensor, region, infer_size, n, sigma, rng):
    """Average RGB input gradients over noisy copies of one frame."""
    # Explanations differentiate with respect to RGB, not the fixed prior.
    grads = torch.zeros_like(x_rgb)   # (1, 3, H, W)
    for _ in range(n):
        noisy_rgb = (x_rgb + sigma * torch.randn_like(x_rgb)).requires_grad_(True)
        x_in = _make_input(noisy_rgb, mname, prior_tensor, infer_size)
        depth = _forward_depth(model, x_in)
        target = depth[..., region[0], region[1]].mean()
        model.zero_grad(set_to_none=True)
        target.backward()
        grads += noisy_rgb.grad.abs()
    sal = grads.sum(dim=1)[0]        # aggregate over RGB channels -> (H, W)
    sal = sal / (sal.max() + 1e-12)
    return sal.detach().cpu().numpy()


def gradcam_vit(model, x_rgb, mname, prior_tensor, region, infer_size):
    """Derive a Grad-CAM map from the final ViT block's patch tokens."""
    feats = {}
    def hook(_m, _i, o):
        feats["tokens"] = o[0] if isinstance(o, tuple) else o

    # The final transformer block provides the spatial feature response.
    block = model.pretrained.blocks[-1]
    h = block.register_forward_hook(hook)

    x_rgb = x_rgb.requires_grad_(True)
    x_in = _make_input(x_rgb, mname, prior_tensor, infer_size)

    depth = _forward_depth(model, x_in)
    target = depth[..., region[0], region[1]].mean()
    model.zero_grad(set_to_none=True)
    tokens = feats["tokens"]
    tokens.retain_grad()
    target.backward()
    h.remove()

    g = tokens.grad[0]                      # (num_tokens, channels)
    t = tokens[0]
    weights = g.mean(dim=0, keepdim=True)   # channel weights
    cam = torch.relu((t * weights).sum(dim=1))
    side_h = infer_size[0] // 14
    side_w = infer_size[1] // 14
    cam = cam[-side_h * side_w:].reshape(side_h, side_w)
    cam = cam / (cam.max() + 1e-12)
    return cam.detach().cpu().numpy()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = resolve_device(args.device)
    infer_size = tuple(int(v) for v in args.infer_size.split(","))

    # Reuse one prior tensor for every PitchDepth explanation.
    prior_tensor = _load_prior(args.prior_path, args.prior_norm,
                               args.prior_scale, device)
    print(f"[info] Ground prior loaded: shape={tuple(prior_tensor.shape)}, "
          f"norm={args.prior_norm}, scale={args.prior_scale}")

    # Load RGB only because this script controls prior attachment explicitly.
    ds = SoccerNetDepth(args.data_dir, crop_size=(546, 966), sport=args.sport,
                   split="Test", in_chans=3)
    if len(ds) == 0:
        sys.exit("ERROR: no Test frames found under --data_dir.")

    models = {
        "depthanything": _load_model("depthanything", args.baseline_ckpt, args, device),
        "pitchdepth":    _load_model("pitchdepth",    args.pitchdepth_ckpt, args, device),
    }

    try:
        from lime import lime_image      # noqa: F401
        have_lime = True
    except ImportError:
        have_lime = False
        print("[warn] `lime` not installed - LIME figures (fig5_37_*) skipped. "
              "Install with: pip install lime scikit-image")

    for idx in args.frame_indices:
        rgb, _gt, _mask = ds[idx]
        # Keep RGB separate here; _make_input appends the prior for PitchDepth.
        x_rgb = rgb.unsqueeze(0).to(device)          # (1, 3, H, W)
        rgb_hwc = np.clip(np.transpose(rgb[:3].numpy(), (1, 2, 0)), 0, 1)

        # Region coordinates refer to the dataset's 546 x 966 crop.
        for rname, region in _regions(x_rgb.shape[-2], x_rgb.shape[-1]).items():
            for mname, model in models.items():
                # Explain the region through the last transformer block.
                cam = gradcam_vit(model, x_rgb.clone(), mname, prior_tensor,
                                  region, infer_size)
                fn = f"fig5_35_gradcam_{mname}_frame{idx:04d}_{rname}.png"
                cam_up = np.kron(cam, np.ones((14, 14)))[:infer_size[0], :infer_size[1]]
                rgb_vis = (rgb_hwc[:infer_size[0], :infer_size[1]]
                           if rgb_hwc.shape[0] >= infer_size[0] else rgb_hwc)
                _save_heat(cam_up, rgb_vis, args.out_dir, fn,
                           f"Grad-CAM-style explanation, {MODEL_LABEL[mname]}, "
                           f"frame {idx}, target: mean depth of {rname}")
                MANIFEST.append({"file": fn, "figure": "5.35",
                                 "method": "Grad-CAM (ViT adaptation)",
                                 "model": MODEL_LABEL[mname], "frame": idx,
                                 "target": f"mean predicted depth, {rname}"})

                # Explain the same region through averaged RGB gradients.
                sal = smoothgrad(model, x_rgb.clone(), mname, prior_tensor,
                                 region, infer_size,
                                 args.sg_samples, args.sg_sigma, rng)
                fn = f"fig5_36_smoothgrad_{mname}_frame{idx:04d}_{rname}.png"
                _save_heat(sal, rgb_hwc, args.out_dir, fn,
                           f"SmoothGrad saliency, {MODEL_LABEL[mname]}, "
                           f"frame {idx}, target: mean depth of {rname}")
                MANIFEST.append({"file": fn, "figure": "5.36",
                                 "method": "SmoothGrad",
                                 "model": MODEL_LABEL[mname], "frame": idx,
                                 "target": f"mean predicted depth, {rname}"})

        # LIME perturbs superpixels at the model's inference resolution.
        if have_lime:
            from lime import lime_image
            from skimage.segmentation import mark_boundaries

            # Scale the selected region to the model input size.
            scale_h = infer_size[0] / x_rgb.shape[-2]
            scale_w = infer_size[1] / x_rgb.shape[-1]
            region_scaled = (
                slice(int(region[0].start * scale_h), int(region[0].stop * scale_h)),
                slice(int(region[1].start * scale_w), int(region[1].stop * scale_w)),
            )

            # LIME expects an unsigned eight-bit, channel-last image.
            rgb_lime = np.clip(
                np.transpose(
                    torch.nn.functional.interpolate(
                        x_rgb, size=infer_size, mode="bilinear", align_corners=False
                    )[0].cpu().numpy(), (1, 2, 0)
                ), 0, 1
            )
            rgb_lime_uint8 = (rgb_lime * 255).astype(np.uint8)

            for mname, model in models.items():
                print(f"[info] Running LIME for {mname}, frame {idx} "
                      f"({args.lime_samples} samples) ...")

                _mname_capture = mname
                _model_capture = model

                def _predict_fn(perturbed_images,
                                _mn=_mname_capture, _mo=_model_capture):
                    results = []
                    with torch.no_grad():
                        for img in perturbed_images:
                            t = torch.from_numpy(img.astype(np.float32) / 255.0)
                            t = t.permute(2, 0, 1).unsqueeze(0).to(device)
                            if _mn == "pitchdepth":
                                prior_rs = torch.nn.functional.interpolate(
                                    prior_tensor, size=infer_size,
                                    mode="bilinear", align_corners=False)
                                t = torch.cat([t, prior_rs], dim=1)
                            depth = _mo(t)
                            # Both model wrappers return one depth map per image.
                            d = depth[0] if depth.dim() == 3 else depth[0, 0]
                            score = d[region_scaled[0], region_scaled[1]].mean().item()
                            results.append(score)
                    return np.array(results).reshape(-1, 1)

                explainer = lime_image.LimeImageExplainer()
                explanation = explainer.explain_instance(
                    rgb_lime_uint8,
                    _predict_fn,
                    top_labels=1,       # regression: treat single output as 1 label
                    labels=(0,),        # explain output index 0
                    hide_color=0,
                    num_samples=args.lime_samples,
                    random_seed=args.seed,
                )

                temp, mask = explanation.get_image_and_mask(
                    label=0,
                    positive_only=True,
                    num_features=args.lime_features,
                    hide_rest=False,
                )

                seg = explanation.segments
                local_exp = dict(explanation.local_exp.get(0, []))
                heat_lime = np.zeros(seg.shape, dtype=np.float32)
                for seg_id, weight in local_exp.items():
                    heat_lime[seg == seg_id] = weight
                amax = np.abs(heat_lime).max()
                if amax > 0:
                    heat_lime = heat_lime / amax

                fn = f"fig5_37_lime_{mname}_frame{idx:04d}_{rname}.png"
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
                axes[0].imshow(mark_boundaries(temp / 255.0, mask))
                axes[0].set_title(
                    f"LIME superpixels, {MODEL_LABEL[mname]}\n"
                    f"frame {idx}, target: mean depth of {rname}", fontsize=8)
                axes[0].axis("off")
                axes[1].imshow(rgb_lime)
                im2 = axes[1].imshow(heat_lime, cmap="RdBu_r", alpha=0.55,
                                     vmin=-1, vmax=1)
                axes[1].set_title("Superpixel weights (red=positive influence)",
                                  fontsize=8)
                axes[1].axis("off")
                fig.colorbar(im2, ax=axes[1], fraction=0.035, pad=0.02)
                fig.tight_layout()
                fig.savefig(os.path.join(args.out_dir, fn), dpi=300,
                            bbox_inches="tight")
                plt.close(fig)
                print("[saved]", fn)
                MANIFEST.append({"file": fn, "figure": "5.37",
                                 "method": "LIME (superpixel regression)",
                                 "model": MODEL_LABEL[mname], "frame": idx,
                                 "target": f"mean predicted depth, {rname}"})

    with open(os.path.join(args.out_dir, "xai_manifest.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["file", "figure", "method",
                                            "model", "frame", "target"])
        wr.writeheader()
        wr.writerows(MANIFEST)
    print(f"Done: {len(MANIFEST)} XAI figures written to {args.out_dir}. "
          "No model was trained or modified.")


if __name__ == "__main__":
    main()
