"""SoccerNet-Depth frames as PyTorch samples for training and evaluation.

Each sample is ``(image, target, mask)``:

- ``target`` is relative depth in model space, ``1 - depth_r / 65535``, with
  shape (1, H, W);
- ``mask`` marks valid pixels with shape (1, H, W); without a mask file every
  pixel is valid;
- ``image`` depends on the mode. Training samples are resized to the crop
  size, augmented, ImageNet-normalised and, for PitchDepth, given the fixed
  prior as a fourth channel. Evaluation samples keep the raw RGB frame in
  [0, 1] at full resolution, because ``DepthAnythingV2.infer_image`` does its
  own resizing, normalisation and prior attachment.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import ColorJitter
from torchvision.transforms.v2 import functional as TF

from data.discovery import find_rgb_depth
from data.ground_prior import load_ground_prior, load_ground_prior_meta, prepare_prior_channel

logger = logging.getLogger("pitchdepth")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TRAINING_SPLITS = ("Train", "Complete")
# Evaluation batches stay small whatever the training batch size.
EVAL_BATCH_SIZE = 4


# Standard photometric jitter for an RGB image in [0, 1]. The hue range is
# small so the synthetic pitch keeps a plausible colour.
COLOUR_JITTER = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
AUGMENT_PROBABILITY = 0.5


class SoccerNetDepth(Dataset):
    """Labelled SoccerNet-Depth frames from one split."""

    def __init__(self, data_dir_path, crop_size=(392, 672), sport="foot", split="Train",
                 in_chans=3, prior_path=None, prior_norm="standardize", prior_scale=1.0,
                 eval_mode=False):
        super().__init__()
        self.split = split
        self.crop_size = tuple(crop_size)
        self.sport = sport
        # The test split is always served raw for evaluation.
        self.eval_mode = bool(eval_mode) or split == "Test"
        self.augment = split in TRAINING_SPLITS

        self.in_chans = int(in_chans)
        self.prior_norm = prior_norm
        self.prior_scale = prior_scale
        self.prior = None
        self.prior_meta = None
        if self.in_chans >= 4:
            if not prior_path:
                raise ValueError("A four-channel dataset needs prior_path to point at a generated prior (.npy).")
            self.prior = load_ground_prior(prior_path)
            self.prior_meta = load_ground_prior_meta(prior_path)

        self.image_pairs = find_rgb_depth(data_dir_path, split, sport=sport)

    def __len__(self):
        return len(self.image_pairs)

    @staticmethod
    def _mask_path(rgb_path: str) -> str:
        # Optional masks sit in a "masks" folder beside "color", as .pt files.
        frame_folder, name = os.path.split(rgb_path)
        stem = os.path.splitext(name)[0]
        return os.path.join(os.path.dirname(frame_folder), "masks", stem + ".pt")

    def _read(self, index):
        rgb_path, depth_path = self.image_pairs[index]
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        depth_r = np.asarray(Image.open(depth_path), dtype=np.float32) / 65535.0

        image = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        target = torch.from_numpy(1.0 - depth_r).unsqueeze(0)

        mask_path = self._mask_path(rgb_path)
        if os.path.exists(mask_path):
            mask = torch.load(mask_path)
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"{mask_path} does not contain a tensor.")
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()
        else:
            mask = torch.ones_like(target)
        return image, target, mask

    def _prior_channel(self, height, width, flipped):
        channel = prepare_prior_channel(self.prior, size=(height, width), norm=self.prior_norm,
                                        scale=self.prior_scale, meta=self.prior_meta)
        return TF.horizontal_flip(channel) if flipped else channel

    def __getitem__(self, index):
        image, target, mask = self._read(index)

        if not self.eval_mode:
            image = TF.resize(image, list(self.crop_size), interpolation=InterpolationMode.BILINEAR, antialias=True)
            target = TF.resize(target, list(self.crop_size), interpolation=InterpolationMode.NEAREST_EXACT)
            mask = TF.resize(mask, list(self.crop_size), interpolation=InterpolationMode.NEAREST_EXACT)

        flipped = False
        if self.augment:
            # Colour jitter changes RGB only; a flip moves image, target, mask and prior together.
            if torch.rand(()) < AUGMENT_PROBABILITY:
                image = COLOUR_JITTER(image).clamp(0.0, 1.0)
            if torch.rand(()) < AUGMENT_PROBABILITY:
                flipped = True
                image = TF.horizontal_flip(image)
                target = TF.horizontal_flip(target)
                mask = TF.horizontal_flip(mask)

        if not self.eval_mode:
            image = TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            if self.in_chans >= 4:
                prior = self._prior_channel(image.shape[-2], image.shape[-1], flipped)
                image = torch.cat([image, prior.to(image.dtype)], dim=0)

        return image, target, mask


def make_loader(args, rank: int = 0, world_size: int = 1, split: str = "Train"):
    """Build the dataset and loader for ``split``; returns (loader, number of frames)."""
    training = split in TRAINING_SPLITS
    dataset = SoccerNetDepth(
        data_dir_path=args.data_dir,
        crop_size=args.crop_size,
        sport=args.sport,
        split=split,
        in_chans=int(getattr(args, "in_chans", 3)),
        prior_path=getattr(args, "prior_path", None),
        prior_norm=getattr(args, "prior_norm", "standardize"),
        prior_scale=getattr(args, "prior_scale", 1.0),
        eval_mode=not training,
    )
    # A distributed sampler is used even on one GPU so the shuffle order is
    # the same whatever the number of processes.
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=training)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size if training else EVAL_BATCH_SIZE,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    if rank == 0:
        logger.info(f"{split}: {len(dataset)} frames.")
    return loader, len(dataset)
