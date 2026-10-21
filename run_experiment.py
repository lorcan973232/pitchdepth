"""Print or launch a named training run from ``configs/experiments.json``.

Each run writes to ``<checkpoint_dir>/seed<N>``. Use ``--seed N`` for one seed
or ``--all-seeds`` for every seed listed in the registry.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "experiments.json")


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def merge_args(base, overrides):
    merged = {k: v for k, v in base.items() if k != "flags"}
    flags = list(base.get("flags", []))
    for k, v in overrides.items():
        merged[k] = v
    return merged, flags


def resolve(name, cfg, seed=None):
    """Merged arguments for one experiment and seed, with a per-seed checkpoint folder."""
    exps = cfg["experiments"]
    if name not in exps:
        raise SystemExit(f"Unknown experiment '{name}'. Use --list to see options.")
    merged, flags = merge_args(cfg["base"], exps[name]["args"])
    merged["seed"] = int(merged.get("seed", 0) if seed is None else seed)
    base_dir = merged.get("checkpoint_dir", f"checkpoints/{name}")
    merged["checkpoint_dir"] = os.path.join(base_dir, f"seed{merged['seed']}")
    return merged, flags


def build_train_command(merged, flags, gpus):
    cmd = [
        "torchrun", "--nnodes=1", f"--nproc_per_node={gpus}",
        "--node_rank=0", "--master_addr=127.0.0.1", "--master_port=29500",
        "train.py",
    ]
    for k, v in merged.items():
        cmd += [f"--{k}", str(v)]
    for fl in flags:
        cmd += [f"--{fl}"]
    return cmd


def build_eval_commands(name, merged):
    """Evaluation commands for the best-validation and final-epoch weights."""
    in_chans = int(merged.get("in_chans", 3))
    model_type = "pitchdepth" if in_chans == 4 else "depthanything"
    commands = []
    for which in ("best", "final"):
        cmd = [
            sys.executable, "evaluate_pitchdepth.py",
            "--model_type", model_type,
            "--encoder", str(merged.get("encoder", "vitl")),
            "--checkpoint", os.path.join(merged["checkpoint_dir"], f"{which}_model.pth"),
            "--data_dir", str(merged.get("data_dir", "SoccerNet/SN-Depth-2025")),
            # Evaluation runs at the training crop, given as H,W.
            "--infer_size", str(merged.get("crop_size", "392,672")),
            "--tag", f"{name}_seed{merged['seed']}_{which}",
        ]
        if in_chans == 4:
            cmd += [
                "--prior_path", str(merged.get("prior_path", "priors/ground_prior_foot.npy")),
                "--prior_norm", str(merged.get("prior_norm", "standardize")),
                "--prior_scale", str(merged.get("prior_scale", 1.0)),
            ]
        commands.append(cmd)
    return commands


def main():
    p = argparse.ArgumentParser(description="PitchDepth experiment runner")
    p.add_argument("experiment", nargs="?", help="Experiment name from configs/experiments.json")
    p.add_argument("--list", action="store_true", help="List available experiments")
    p.add_argument("--gpus", type=int, default=1, help="nproc_per_node for torchrun")
    seeds = p.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, default=None, help="Run one seed (default: the base seed)")
    seeds.add_argument("--all-seeds", action="store_true", help="Run every seed in the registry, one after another")
    p.add_argument("--execute", action="store_true", help="Actually run (default: print only)")
    p.add_argument("--print-eval", action="store_true", help="Also print the matching evaluation commands")
    args = p.parse_args()

    cfg = load_config()

    if args.list or not args.experiment:
        print("Available experiments:\n")
        for name, exp in cfg["experiments"].items():
            print(f"  {name:22s} {exp['description']}")
        print(f"\nSeeds for --all-seeds: {cfg.get('seeds', [cfg['base'].get('seed', 0)])}")
        return

    run_seeds = cfg.get("seeds", [0]) if args.all_seeds else [args.seed]
    print(f"# Experiment: {args.experiment}")
    print(f"# {cfg['experiments'][args.experiment]['description']}")
    for seed in run_seeds:
        merged, flags = resolve(args.experiment, cfg, seed)
        cmd = build_train_command(merged, flags, args.gpus)
        print(f"\n# Seed {merged['seed']}")
        print(" ".join(cmd))
        if args.print_eval:
            print("# Matching evaluation commands:")
            for eval_cmd in build_eval_commands(args.experiment, merged):
                print(" ".join(eval_cmd))
        if args.execute:
            os.makedirs(merged["checkpoint_dir"], exist_ok=True)
            print("\nExecuting...\n")
            code = subprocess.call(cmd)
            if code != 0:
                raise SystemExit(code)


if __name__ == "__main__":
    main()
