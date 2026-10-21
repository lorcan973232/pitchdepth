"""Fine-tune the RGB baseline or four-channel PitchDepth on SoccerNet-Depth.

Runs on one GPU with ``python train.py`` or on several with ``torchrun``.
``run_experiment.py`` builds the commands for the registered experiments and
seeds.

Each run trains for a fixed number of epochs, validates after every epoch and
writes to ``--checkpoint_dir``:

    run_config.json     the resolved arguments
    metrics_log.jsonl   training loss and validation metrics per epoch
    best_model.pth      weights at the epoch with the lowest validation AbsRel
    final_model.pth     weights after the last epoch
    last.ckpt           full training state after the latest epoch, for --resume
    epoch_XX.pth        weights every --save_every epochs (off by default)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from data.dataset import make_loader
from loss.depth_losses import TRAINING_LOSSES, training_loss
from metrics.depth_metrics import MetricMeans, align, depth_metrics, metric_average
from training.checkpoints import load_training_state, save_training_state, save_weights, unwrap
from training.defaults import DEFAULT_DATA_DIR, DEFAULT_PRIOR_PATH, base_weights_path
from training.model import MODEL_CONFIGS, build_model, enable_gradient_checkpointing
from training.optim import DECODER_LR_MULTIPLIER, OPTIMIZERS, SCHEDULERS, make_optimizer, make_scheduler
from utils.device import amp_autocast, amp_enabled, require_cuda

try:
    import wandb
except ImportError:
    wandb = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pitchdepth")

# Checkpoints are selected on this validation error.
SELECTION_METRIC = "abs_rel"


def size_pair(text: str) -> tuple[int, int]:
    """Parse 'H,W' into (H, W)."""
    values = tuple(int(v) for v in text.strip("()").split(","))
    if len(values) != 2:
        raise argparse.ArgumentTypeError(f"Expected 'height,width', got '{text}'.")
    return values


def parse_args(argv=None) -> argparse.Namespace:
    """Defaults reproduce the RGB baseline run; see configs/experiments.json."""
    p = argparse.ArgumentParser(description="Fine-tune Depth Anything V2 or PitchDepth on SoccerNet-Depth.")

    data = p.add_argument_group("data")
    data.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    data.add_argument("--sport", default="foot", choices=["foot", "basket"])
    data.add_argument("--crop_size", type=size_pair, default=(392, 672),
                      help="Training resolution as height,width.")
    data.add_argument("--val_infer_size", type=size_pair, default=None,
                      help="Validation inference resolution as height,width (default: --crop_size).")
    data.add_argument("--val_split", default="Validation")
    data.add_argument("--num_workers", type=int, default=0)
    data.add_argument("--mask", action="store_true",
                      help="Restrict the loss to the dataset masks as well as positive-depth pixels.")

    model = p.add_argument_group("model")
    model.add_argument("--encoder", default="vitl", choices=sorted(MODEL_CONFIGS))
    model.add_argument("--pretrained_weights", default=None,
                       help="Depth Anything V2 weights (default: checkpoints/DepthAnythingV2/depth_anything_v2_<encoder>.pth).")
    model.add_argument("--load_head", action="store_true", help="Also load the pretrained decoder.")
    model.add_argument("--in_chans", type=int, default=3, choices=[3, 4],
                       help="3 for the RGB baseline, 4 for PitchDepth (RGB + prior).")
    model.add_argument("--prior_path", default=DEFAULT_PRIOR_PATH)
    model.add_argument("--prior_norm", default="standardize", choices=["none", "minmax", "standardize"])
    model.add_argument("--prior_scale", type=float, default=1.0)
    model.add_argument("--prior_init", default="zero", choices=["zero", "avg"],
                       help="Initial patch-projection kernel for the prior channel.")

    optim = p.add_argument_group("optimisation")
    optim.add_argument("--loss_fn", default="ssitrim", choices=TRAINING_LOSSES)
    optim.add_argument("--optimizer", default="adamw", choices=sorted(OPTIMIZERS))
    optim.add_argument("--learning_rate", type=float, default=5e-6, help="Encoder learning rate.")
    optim.add_argument("--decoder_lr_multiplier", type=float, default=DECODER_LR_MULTIPLIER,
                       help="Decoder learning rate as a multiple of the encoder's.")
    optim.add_argument("--weight_decay", type=float, default=0.01)
    optim.add_argument("--scheduler", default="poly", choices=SCHEDULERS)
    optim.add_argument("--epochs", type=int, default=10)
    optim.add_argument("--batch_size", type=int, default=1)
    optim.add_argument("--accumulation_steps", type=int, default=4,
                       help="Batches per optimiser step; the effective batch is batch_size times this.")
    optim.add_argument("--precision", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    optim.add_argument("--grad_checkpoint", action="store_true",
                       help="Recompute encoder activations in the backward pass to save memory.")
    optim.add_argument("--seed", type=int, default=0)

    run = p.add_argument_group("run")
    run.add_argument("--checkpoint_dir", default="checkpoints/run")
    run.add_argument("--validate", action="store_true",
                     help="Validate after each epoch and keep the weights with the lowest validation AbsRel.")
    run.add_argument("--val_freq", type=int, default=1, help="Validate every N epochs.")
    run.add_argument("--early_stop_patience", type=int, default=0,
                     help="Stop after this many validations without improvement; 0 trains every epoch.")
    run.add_argument("--save_every", type=int, default=0,
                     help="Also keep the weights every N epochs as epoch_XX.pth; 0 keeps none.")
    run.add_argument("--log_freq", type=int, default=200, help="Log the running loss every N batches.")
    run.add_argument("--resume", action="store_true", help="Continue from last.ckpt in --checkpoint_dir.")
    run.add_argument("--wandb", action="store_true", help="Also log to Weights & Biases.")
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def start_processes():
    """Return (rank, world_size, device); one process needs no process group."""
    require_cuda()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        if os.name == "nt":
            # Windows has no NCCL, and a file store avoids torchrun's TCP store there.
            store = dist.FileStore(os.path.join(tempfile.gettempdir(), "pitchdepth_store"), world_size)
            dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
        else:
            dist.init_process_group("nccl")
    return rank, world_size, torch.device("cuda", local_rank)


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@torch.no_grad()
def validate(model, loader, device, infer_size, sport) -> dict:
    """Mean metrics over the loader, in stored depth_r space after alignment."""
    model.eval()
    net = unwrap(model)
    means = MetricMeans()
    for image, target, _ in tqdm(loader, desc="Validation", leave=False):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).squeeze(1)
        pred = net.infer_image(image, infer_size=infer_size)
        aligned = align(pred, target, (target > 0).float())
        # Targets are 1 - depth_r; the metrics use depth_r itself.
        means.add(depth_metrics(1 - target, 1 - aligned, sport=sport))
    return means.means()


def train(args) -> None:
    rank, world_size, device = start_processes()
    seed_everything(args.seed)
    main_process = rank == 0
    if args.val_infer_size is None:
        args.val_infer_size = args.crop_size
    if args.pretrained_weights is None:
        args.pretrained_weights = base_weights_path(args.encoder)
    if args.wandb and wandb is None:
        logger.warning("wandb is not installed; logging locally only.")
        args.wandb = False

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    out = lambda name: os.path.join(args.checkpoint_dir, name)
    if main_process:
        with open(out("run_config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
        if args.wandb:
            wandb.init(project="PitchDepth", config=vars(args),
                       name=f"{args.encoder}_ch{args.in_chans}_{args.loss_fn}_seed{args.seed}")

    train_loader, n_train = make_loader(args, rank, world_size, "Train" if args.validate else "Complete")
    val_loader = make_loader(args, rank, world_size, args.val_split)[0] if args.validate else None

    model = build_model(args.encoder, args.in_chans, base_weights=args.pretrained_weights,
                        load_head=args.load_head, prior_init=args.prior_init, prior_path=args.prior_path,
                        prior_norm=args.prior_norm, prior_scale=args.prior_scale).to(device)
    if args.grad_checkpoint:
        wrapped = enable_gradient_checkpointing(model)
        logger.info(f"Gradient checkpointing on {wrapped} encoder blocks.")
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)

    accumulation = max(1, args.accumulation_steps)
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    optimizer = make_optimizer(model, args.optimizer, args.learning_rate, args.weight_decay,
                               decoder_lr_multiplier=args.decoder_lr_multiplier)
    scheduler = make_scheduler(optimizer, args.scheduler, total_steps=steps_per_epoch * args.epochs)
    objective = training_loss(args.loss_fn)
    use_amp = amp_enabled(device, args.precision)
    # Loss scaling is needed only for float16; bfloat16 has float32's range.
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "float16")

    start_epoch, best_score, stale_validations = 0, float("inf"), 0
    if args.resume:
        state = load_training_state(out("last.ckpt"), model, optimizer, scheduler, map_location=device)
        start_epoch = state["epoch"] + 1
        best_score = state.get("best_score", best_score)
        stale_validations = state.get("stale_validations", 0)
        logger.info(f"Resuming at epoch {start_epoch + 1}.")

    if main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"{args.encoder}, {args.in_chans} input channels, {trainable / 1e6:.1f}M trainable parameters; "
                    f"{n_train} training frames; effective batch {args.batch_size * accumulation}; seed {args.seed}.")

    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        running_loss, batches = 0.0, len(train_loader)
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=not main_process)
        for index, (image, target, mask) in enumerate(progress):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).squeeze(1)
            valid = (target > 1e-5).float() * mask.to(device).squeeze(1) if args.mask else (target > 0).float()

            if index % accumulation == 0:
                optimizer.zero_grad()
            with amp_autocast(device, args.precision, enabled=use_amp):
                loss = objective(model(image), target, valid)
            # Dividing by the cycle length makes the summed gradient an average.
            scaler.scale(loss / accumulation).backward()

            if (index + 1) % accumulation == 0 or index + 1 == batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            running_loss += loss.item()
            if main_process and (index + 1) % args.log_freq == 0:
                progress.set_postfix(loss=f"{running_loss / (index + 1):.5f}",
                                     lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                if args.wandb:
                    wandb.log({"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"]})

        epoch_loss = torch.tensor([running_loss, batches], dtype=torch.float64, device=device)
        if world_size > 1:
            dist.all_reduce(epoch_loss)
        record = {"epoch": epoch + 1, "train_loss": (epoch_loss[0] / epoch_loss[1]).item()}

        stop = torch.zeros(1, device=device)
        if args.validate and (epoch + 1) % args.val_freq == 0:
            metrics = validate(model, val_loader, device, args.val_infer_size, args.sport)
            score = float(metrics[SELECTION_METRIC])
            record.update({k: float(v) for k, v in metrics.items()}, metric_average=float(metric_average(metrics)))
            if main_process:
                if args.wandb:
                    wandb.log({f"val/{k}": v for k, v in record.items() if k != "epoch"})
                if score < best_score:
                    logger.info(f"Epoch {epoch + 1}: validation {SELECTION_METRIC} {score:.6f} is the best so far.")
                    best_score, stale_validations = score, 0
                    save_weights(out("best_model.pth"), model, epoch=epoch, **{f"val_{SELECTION_METRIC}": score})
                else:
                    stale_validations += 1
                    logger.info(f"Epoch {epoch + 1}: validation {SELECTION_METRIC} {score:.6f}; "
                                f"best remains {best_score:.6f}.")
                    if 0 < args.early_stop_patience <= stale_validations:
                        stop += 1

        if main_process:
            append_jsonl(out("metrics_log.jsonl"), record)
            if args.save_every and (epoch + 1) % args.save_every == 0:
                save_weights(out(f"epoch_{epoch + 1:02d}.pth"), model, epoch=epoch)
            save_training_state(out("last.ckpt"), model, optimizer, scheduler, epoch,
                                best_score=best_score, stale_validations=stale_validations)

        # Every process must leave the loop together.
        if world_size > 1:
            dist.all_reduce(stop)
        if stop.item() > 0:
            logger.info("Early stopping: validation has stopped improving.")
            break

    if main_process:
        save_weights(out("final_model.pth"), model, epoch=epoch)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train(parse_args())
