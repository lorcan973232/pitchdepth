"""Exercise PitchDepth data, model, loss and metric paths on synthetic frames.\n\nCUDA-only entry-point checks run when a GPU is available.\n"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

# Direct script execution needs the repository root on the import path.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests.make_synthetic_dataset import make_synthetic_dataset

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    if cond:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}  {detail}")


def section(title):
    print(f"\n=== {title} ===")


def main():
    work = tempfile.mkdtemp(prefix="pitchdepth_smoke_")
    data_dir = os.path.join(work, "synthetic_data")
    prior_dir = os.path.join(work, "priors")
    out_dir = os.path.join(work, "out")
    fig_dir = os.path.join(work, "figs")
    H, W = 140, 252
    try:
        make_synthetic_dataset(data_dir, splits=("Train", "Test"), n_per_split=4, h=H, w=W)

        # Ground-prior generation and metadata.
        section("1. Ground-prior generation")
        from data.ground_prior import (generate_ground_prior, load_ground_prior,
                                        load_ground_prior_meta, prepare_prior_channel)
        prior, meta, npy_path = generate_ground_prior(
            data_dir, split="Train", sport="foot", crop_fraction=2 / 3,
            out_dir=prior_dir, verbose=False)
        top = int(round(H * (1 - 2 / 3)))
        check("prior shape", prior.shape == (H, W), str(prior.shape))
        check("prior top third zeroed", np.allclose(prior[:top], 0.0))
        check("prior bottom non-zero", prior[top:].mean() > 0)
        check("prior frame count", meta.num_frames == 4, str(meta.num_frames))
        check("prior files saved", os.path.exists(npy_path)
              and os.path.exists(npy_path.replace(".npy", ".json")))

        # Prior normalisation.
        section("2. Prior load / normalise")
        ploaded = load_ground_prior(npy_path)
        pmeta = load_ground_prior_meta(npy_path)
        check("loaded prior shape", tuple(ploaded.shape) == (1, H, W), str(ploaded.shape))
        ch = prepare_prior_channel(ploaded, size=(70, 126), norm="standardize", meta=pmeta)
        check("prepared channel shape", tuple(ch.shape) == (1, 70, 126), str(ch.shape))
        check("prepared channel finite", torch.isfinite(ch).all().item())

        # Pretrained patch-weight channel inflation.
        section("3. Patch-embed weight inflation")
        from models.DAv2.depth_anything_v2.dpt import inflate_patch_embed_weight, PATCH_EMBED_KEY
        w3 = torch.randn(8, 3, 14, 14)
        sd = {PATCH_EMBED_KEY: w3.clone()}
        sd4 = inflate_patch_embed_weight(dict(sd), 4, init="zero")
        check("inflate shape 3->4", tuple(sd4[PATCH_EMBED_KEY].shape) == (8, 4, 14, 14))
        check("inflate keeps RGB", torch.allclose(sd4[PATCH_EMBED_KEY][:, :3], w3))
        check("inflate zero 4th", torch.allclose(sd4[PATCH_EMBED_KEY][:, 3], torch.zeros(8, 14, 14)))
        sd4a = inflate_patch_embed_weight(dict(sd), 4, init="avg")
        check("inflate avg 4th", torch.allclose(sd4a[PATCH_EMBED_KEY][:, 3], w3.mean(1)))

        # Four-channel model forward pass.
        section("4. 4-channel forward + backward")
        from models.DAv2.depth_anything_v2.dpt import DepthAnythingV2
        vits = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
        model4 = DepthAnythingV2(in_chans=4, **vits)
        x = torch.randn(1, 4, 70, 126, requires_grad=True)
        out = model4(x)
        check("forward output shape", tuple(out.shape) == (1, 70, 126), str(out.shape))
        out.mean().backward()
        gw = model4.pretrained.patch_embed.proj.weight.grad
        check("patch-embed grad shape", gw is not None and tuple(gw.shape) == (384, 4, 14, 14))

        # Inference preprocessing and output shapes.
        section("5. Prior-aware infer_image")
        model4.set_ground_prior(ploaded, norm="standardize",
                                stats=(pmeta.valid_mean, pmeta.valid_std, pmeta.valid_min, pmeta.valid_max))
        rgb = torch.rand(1, 3, 140, 252)
        img4 = model4.prepare_input(rgb, (70, 126))
        check("prepare_input appends prior", tuple(img4.shape) == (1, 4, 70, 126), str(tuple(img4.shape)))
        d4 = model4.infer_image(rgb, infer_size=(70, 126))
        check("infer_image 4ch shape", tuple(d4.shape) == (1, 140, 252), str(tuple(d4.shape)))
        model3 = DepthAnythingV2(in_chans=3, **vits)
        d3 = model3.infer_image(rgb, infer_size=(70, 126))
        check("infer_image 3ch regression", tuple(d3.shape) == (1, 140, 252), str(tuple(d3.shape)))

        # Dataset tensors for training and evaluation.
        section("6. Dataset (3ch / 4ch, alignment)")
        from data.dataset import SoccerNetDepth
        ds_tr4 = SoccerNetDepth(data_dir, crop_size=(70, 126), split="Train", in_chans=4,
                                prior_path=npy_path)
        rgb_tr, depth_tr, mask_tr = ds_tr4[0]
        check("train 4ch tensor", tuple(rgb_tr.shape) == (4, 70, 126), str(tuple(rgb_tr.shape)))
        check("train depth shape", depth_tr.shape[-2:] == (70, 126))
        ds_te4 = SoccerNetDepth(data_dir, crop_size=(70, 126), split="Test", in_chans=4,
                                prior_path=npy_path)
        rgb_te, _, _ = ds_te4[0]
        check("test returns 3ch raw", rgb_te.shape[0] == 3, str(tuple(rgb_te.shape)))
        ds_tr3 = SoccerNetDepth(data_dir, crop_size=(70, 126), split="Train", in_chans=3)
        check("train 3ch tensor", ds_tr3[0][0].shape[0] == 3)

        # Loss and metric functions.
        section("7. Losses + metrics")
        from loss.depth_losses import LOSSES as losses, TRAINING_LOSSES, training_loss
        pred = torch.rand(2, 70, 126) * 0.8 + 0.1
        target = torch.rand(2, 70, 126) * 0.8 + 0.1
        m = (target > 0).float()
        for lname in TRAINING_LOSSES:
            try:
                lv = losses[lname](pred.clone().requires_grad_(True), target, m)
                ok = torch.isfinite(lv).all().item()
            except Exception as exc:
                ok, lv = False, exc
            check(f"loss {lname} finite", ok, str(lv))

        # The MiDaS losses ignore any positive scale and any shift of the prediction.
        trim = training_loss("ssitrim")
        base_loss = trim(pred, target, m)
        check("ssitrim is scale and shift invariant",
              torch.allclose(trim(3.0 * pred + 0.7, target, m), base_loss, rtol=1e-4),
              f"{trim(3.0 * pred + 0.7, target, m).item()} vs {base_loss.item()}")
        check("ssitrim is zero for an affine copy of the target",
              trim(2.0 * target - 0.5, target, m).item() < 1e-5, str(trim(2.0 * target - 0.5, target, m).item()))
        check("ssimse is scale and shift invariant",
              torch.allclose(losses["ssimse"](3.0 * pred + 0.7, target, m), losses["ssimse"](pred, target, m), rtol=1e-3))
        # Trimming should discard a larger share of the data term when 10% of rows are outliers.
        noisy = target + 0.05 * torch.randn_like(target)
        outlier = noisy.clone()
        outlier[:, :7, :] += 5.0
        share = {name: trim(p, target, m, gradient_weight=0.0) / trim(p, target, m, trim=0.0, gradient_weight=0.0)
                 for name, p in (("clean", noisy), ("outliers", outlier))}
        check("ssitrim trims outlier residuals", share["outliers"] < share["clean"] < 1.0,
              f"kept share clean {share['clean'].item():.3f}, with outliers {share['outliers'].item():.3f}")

        # Invalid masks must skip cleanly without breaking autograd.
        for lname in TRAINING_LOSSES:
            sparse_pred = torch.rand(1, 4, 4, requires_grad=True)
            sparse_target = torch.rand(1, 4, 4)
            sparse_mask = torch.zeros(1, 4, 4)
            try:
                sparse_loss = losses[lname](sparse_pred, sparse_target, sparse_mask)
                sparse_loss.backward()
                ok = (
                    torch.is_tensor(sparse_loss)
                    and sparse_loss.item() == 0.0
                    and sparse_pred.grad is not None
                    and torch.count_nonzero(sparse_pred.grad).item() == 0
                )
            except Exception as exc:
                ok, sparse_loss = False, exc
            check(f"loss {lname} empty-mask autograd", ok, str(sparse_loss))

        from metrics.depth_metrics import depth_metrics, fit_scale_shift
        scale, shift = fit_scale_shift(pred, target, m)
        check("scale/shift shape", tuple(scale.shape) == (2,), str(tuple(scale.shape)))
        exact = torch.rand(2, 70, 126) + 0.1
        s_true, t_true = fit_scale_shift(exact, 2.5 * exact - 0.3, torch.ones_like(exact))
        check("scale/shift recovers a known fit", torch.allclose(s_true, torch.full((2,), 2.5), atol=1e-4)
              and torch.allclose(t_true, torch.full((2,), -0.3), atol=1e-4), f"{s_true} {t_true}")
        metrics = depth_metrics(target, pred, sport="foot")
        keys = {"abs_rel", "sq_rel", "rmse", "rmse_log", "silog"}
        check("metrics keys", set(metrics) == keys, str(set(metrics)))
        check("metrics finite", all(np.isfinite(v) for v in metrics.values()))

        # Alternative flat dataset layout.
        section("7b. SN-Depth-2025 flat layout (lowercase splits, no depth-football parent)")
        flat_dir = os.path.join(work, "SN-Depth-2025")
        make_synthetic_dataset(flat_dir, splits=("train", "valid", "test", "challenge"),
                               n_per_split=3, h=70, w=126, layout="flat")
        from data.discovery import find_rgb_depth as _frd, resolve_split_dirs as _rsd
        check("flat Train resolves", len(_frd(flat_dir, "Train", sport="foot")) == 3,
              str(len(_frd(flat_dir, "Train", sport="foot"))))
        check("flat Validation resolves (valid->Validation)", len(_frd(flat_dir, "Validation", sport="foot")) == 3)
        check("flat Test resolves", len(_frd(flat_dir, "Test", sport="foot")) == 3)
        check("flat Challenge has no pairs (no depth_r)", len(_frd(flat_dir, "Challenge", sport="foot")) == 0)
        check("flat Complete = train+valid+test", len(_frd(flat_dir, "Complete", sport="foot")) == 9,
              str(len(_frd(flat_dir, "Complete", sport="foot"))))
        check("flat no case-dup double count", len(_rsd(flat_dir, "Train")) == 1, str(_rsd(flat_dir, "Train")))
        ds_flat = SoccerNetDepth(flat_dir, crop_size=(70, 126), split="Train", in_chans=4, prior_path=npy_path)
        check("flat dataset loads 4ch", ds_flat[0][0].shape[0] == 4 and len(ds_flat) == 3)

        # Training CLI options must expose the implementations they configure.
        section("7c. Training interfaces")
        from train import parse_args as _parse_train_args
        from training.optim import SCHEDULERS, make_optimizer, make_scheduler
        parsed_schedulers = {_parse_train_args(["--scheduler", name]).scheduler for name in SCHEDULERS}
        check("scheduler CLI choices", parsed_schedulers == set(SCHEDULERS), str(parsed_schedulers))
        sizes = _parse_train_args(["--crop_size", "392,672"])
        check("crop and validation sizes are height,width",
              sizes.crop_size == (392, 672) and sizes.val_infer_size is None, str(sizes.crop_size))

        defaults = _parse_train_args([])
        check("training defaults: ssitrim, poly, 10 epochs, no early stop, seed 0",
              (defaults.loss_fn, defaults.scheduler, defaults.epochs, defaults.early_stop_patience, defaults.seed)
              == ("ssitrim", "poly", 10, 0, 0), str(vars(defaults)))

        optimizer = make_optimizer(model3, "adamw", learning_rate=1e-5, weight_decay=1e-3)
        check("decoder learning rate is 10x encoder",
              [g["lr"] for g in optimizer.param_groups] == [1e-5, 1e-4],
              str([g["lr"] for g in optimizer.param_groups]))
        poly = make_scheduler(optimizer, "poly", total_steps=100)
        poly_multipliers = []
        for _ in range(100):
            poly_multipliers.append(poly.get_last_lr()[0] / 1e-5)
            optimizer.step()
            poly.step()
        check("poly schedule decays as (1 - t)^0.9",
              poly_multipliers[0] == 1.0 and abs(poly_multipliers[50] - 0.5 ** 0.9) < 1e-6
              and poly_multipliers[-1] < 0.02 and poly.get_last_lr()[1] == 0.0,
              f"{poly_multipliers[0]} {poly_multipliers[50]} {poly_multipliers[-1]}")
        optimizer = make_optimizer(model3, "adamw", learning_rate=1e-5, weight_decay=1e-3)
        cosine = make_scheduler(optimizer, "cosine_warmup", total_steps=100)
        multipliers = []
        for _ in range(100):
            multipliers.append(cosine.get_last_lr()[0] / 1e-5)
            optimizer.step()
            cosine.step()
        check("cosine schedule warms up then decays",
              multipliers[0] == 0.0 and abs(multipliers[10] - 1.0) < 1e-6 and multipliers[-1] < 0.01,
              f"{multipliers[0]} {multipliers[10]} {multipliers[-1]}")

        from training.model import enable_gradient_checkpointing
        model_ck = DepthAnythingV2(in_chans=3, **vits)
        model_ck.load_state_dict(model3.state_dict())
        x3 = torch.rand(1, 3, 70, 126)
        plain = model3(x3)
        plain.mean().backward()
        enable_gradient_checkpointing(model_ck)
        ck_out = model_ck(x3)
        ck_out.mean().backward()
        same_grad = torch.allclose(model3.pretrained.patch_embed.proj.weight.grad,
                                   model_ck.pretrained.patch_embed.proj.weight.grad, atol=1e-6)
        check("gradient checkpointing keeps outputs and gradients",
              torch.allclose(plain, ck_out, atol=1e-6) and same_grad)

        from data.dataset import make_loader
        loader_args = SimpleNamespace(
            data_dir=data_dir, crop_size=(70, 126), sport="basket",
            batch_size=1, num_workers=0, in_chans=3,
            prior_path=None, prior_norm="standardize", prior_scale=1.0,
        )
        loader, _ = make_loader(loader_args, rank=0, world_size=1, split="Train")
        check("dataloader honours sport", loader.dataset.sport == "basket")
        check("dataloader honours num_workers", loader.num_workers == 0)

        # Experiment registry and multi-seed summary.
        section("7d. Seeds and summaries")
        import run_experiment
        registry = run_experiment.load_config()
        merged, flags = run_experiment.resolve("pitchdepth", registry, seed=2)
        check("per-seed checkpoint folder", merged["seed"] == 2
              and merged["checkpoint_dir"].replace("\\", "/") == "checkpoints/pitchdepth/seed2", str(merged))
        evals = run_experiment.build_eval_commands("pitchdepth", merged)
        check("best and final evaluation commands", [c[c.index("--tag") + 1] for c in evals]
              == ["pitchdepth_seed2_best", "pitchdepth_seed2_final"])
        check("registry seeds are 0, 1, 2", registry.get("seeds") == [0, 1, 2], str(registry.get("seeds")))
        known = set(vars(_parse_train_args([])))
        unknown = {k for k in registry["base"] if k != "flags" and k not in known}
        unknown |= {k for exp in registry["experiments"].values() for k in exp["args"] if k not in known}
        check("registry uses only real train.py options", not unknown, str(unknown))

        from scripts.summarise_seeds import load as load_summary
        summary_csv = os.path.join(work, "summary.csv")
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            f.write("timestamp,tag,model_type,encoder,checkpoint,abs_rel,sq_rel,rmse,rmse_log,silog\n")
            for seed, value in enumerate((0.010, 0.012, 0.014)):
                f.write(f"2026-11-0{seed + 1}_00-00-00,baseline_seed{seed}_final,depthanything,vitl,x,{value},1,1,1,1\n")
            # A later evaluation of the same tag replaces the earlier one.
            f.write("2026-11-09_00-00-00,baseline_seed0_final,depthanything,vitl,x,0.011,1,1,1,1\n")
        rows = load_summary(summary_csv)
        check("summary keeps the latest row per tag and groups seeds",
              sorted(rows[("baseline", "final")]) == [0, 1, 2]
              and rows[("baseline", "final")][0]["abs_rel"] == 0.011, str(dict(rows)))

        # CUDA-only entry-point checks.
        section("8. Scripts: evaluate + visualize (subprocess)")
        if not torch.cuda.is_available():
            print("  SKIP  GPU-only build: evaluate/visualize scripts require CUDA "
                  "(they fail fast without a GPU). Run this section on the GPU box.")
        else:
            os.makedirs(out_dir, exist_ok=True)
            # Small random ViT-S checkpoints are sufficient for interface checks.
            ck3 = os.path.join(work, "ck3.pth")
            ck4 = os.path.join(work, "ck4.pth")
            torch.save({"model": model3.state_dict()}, ck3)
            torch.save({"model": model4.state_dict()}, ck4)

            env = dict(os.environ, PYTHONPATH=REPO_ROOT, PYTHONUTF8="1")
            common = [sys.executable]
            # Three-channel Depth Anything V2.
            r1 = subprocess.run(common + [
                os.path.join(REPO_ROOT, "evaluate_pitchdepth.py"),
                "--model_type", "depthanything", "--encoder", "vits",
                "--checkpoint", ck3, "--data_dir", data_dir, "--max_frames", "2",
                "--infer_size", "70,126", "--batch_size", "1", "--num_workers", "0",
                "--out_dir", out_dir, "--tag", "smoke_da"],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True)
            check("evaluate 3ch runs", r1.returncode == 0, r1.stderr[-800:])
            # Four-channel PitchDepth.
            r2 = subprocess.run(common + [
                os.path.join(REPO_ROOT, "evaluate_pitchdepth.py"),
                "--model_type", "pitchdepth", "--encoder", "vits",
                "--checkpoint", ck4, "--prior_path", npy_path, "--data_dir", data_dir,
                "--max_frames", "2", "--infer_size", "70,126", "--batch_size", "1",
                "--num_workers", "0", "--out_dir", out_dir, "--tag", "smoke_pd"],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True)
            check("evaluate 4ch runs", r2.returncode == 0, r2.stderr[-800:])
            check("summary.csv written", os.path.exists(os.path.join(out_dir, "summary.csv")))

            r3 = subprocess.run(common + [
                os.path.join(REPO_ROOT, "visualize_pitchdepth_figures_5_9_to_5_13.py"),
                "--data_dir", data_dir, "--encoder", "vits",
                "--baseline_ckpt", ck3, "--pitchdepth_ckpt", ck4, "--prior_path", npy_path,
                "--frame_indices", "0", "--profile_frames", "0",
                "--infer_size", "70,126",
                "--out_dir", fig_dir],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True)
            check("visualize runs", r3.returncode == 0, r3.stderr[-800:])
            for f in ["fig5_9_frame0000_01_rgb.png",
                      "fig5_10_frame0000_error_baseline.png",
                      "fig5_11_frame0000_pitchcrop_pitchdepth.png",
                      "fig5_12_frame0000_boundarycrop_groundtruth.png",
                      "fig5_13_frame0000_depth_profile.png",
                      "fig5_13_scatter_pitchdepth.png", "figure_manifest.csv"]:
                check(f"figure {f}", os.path.exists(os.path.join(fig_dir, f)))

    finally:
        shutil.rmtree(work, ignore_errors=True)

    # Temporary files are removed even when a check fails.
    print(f"\n{'=' * 50}")
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("All smoke tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
