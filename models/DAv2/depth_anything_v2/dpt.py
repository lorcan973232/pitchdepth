"""Depth Anything V2 (DPT decoder on a DINOv2 encoder) with an optional prior channel.

The encoder, decoder and their weights follow the official Depth Anything V2
release (Yang et al., 2024; Apache-2.0, see ../LICENSE). PitchDepth adds two
things: a configurable input-channel count, so a fixed ground prior can enter as
a fourth channel, and a batched tensor inference path used for training-time
validation, evaluation and prediction export.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        return out


# DINOv2 takes ImageNet-normalised RGB.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 14


class DepthAnythingV2(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        in_chans=3,
    ):
        super(DepthAnythingV2, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }

        self.encoder = encoder
        # Three channels select the baseline; four include the fixed prior.
        self.in_chans = in_chans
        self.pretrained = DINOv2(model_name=encoder, in_chans=in_chans)

        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

        self.register_buffer("rgb_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        # The fixed prior stays outside checkpoints because its own files record it.
        self.prior_norm = "standardize"
        self.prior_scale = 1.0
        # Non-persistent buffers still follow the model to its selected device.
        self.register_buffer("ground_prior", None, persistent=False)
        self.register_buffer("prior_stats", torch.tensor([0.0, 1.0, 0.0, 1.0]), persistent=False)

    def set_ground_prior(self, prior, norm="standardize", scale=1.0, stats=None):
        """Attach the saved prior and the statistics used to normalise it."""
        if prior is None:
            self.ground_prior = None
            return
        prior = torch.as_tensor(prior).float()
        if prior.ndim == 2:
            prior = prior.unsqueeze(0)            # (1, H, W)
        prior = prior.unsqueeze(0)                # (1, 1, H, W)
        self.ground_prior = prior                 # routes to the registered buffer
        if stats is None:
            # Older files do not have a JSON sidecar, so calculations for their statistics are done here.
            valid = prior[prior != 0]
            mean = float(valid.mean()) if valid.numel() else 0.0
            std = float(valid.std() + 1e-12) if valid.numel() else 1.0
            lo = float(valid.min()) if valid.numel() else 0.0
            hi = float(valid.max()) if valid.numel() else 1.0
            stats = (mean, std, lo, hi)
        self.prior_stats = torch.tensor([float(s) for s in stats])
        self.prior_norm = norm
        self.prior_scale = float(scale)

    def _append_prior(self, image):
        """Append the resized prior to an ImageNet-normalised RGB batch."""
        if self.in_chans < 4 or self.ground_prior is None:
            return image
        b, _, h, w = image.shape
        prior = F.interpolate(self.ground_prior, size=(h, w), mode="bilinear", align_corners=False)
        mean, std, lo, hi = [float(v) for v in self.prior_stats]
        if self.prior_norm == "none":
            p = prior
        elif self.prior_norm == "minmax":
            p = (prior - lo) / (hi - lo + 1e-8)
        else:
            p = (prior - mean) / (std + 1e-12)
        p = (p * self.prior_scale).to(image.dtype).to(image.device)
        p = p.expand(b, 1, h, w)
        return torch.cat([image, p], dim=1)

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)

        return depth.squeeze(1)

    def prepare_input(self, rgb, size):
        """Turn an RGB batch in [0, 1] into model input at ``size`` (height, width).

        The frames are resized bicubically, clipped back to [0, 1] after the
        resize overshoots, ImageNet-normalised and, for a four-channel model,
        given the prior as the last channel. Both sides of ``size`` must be
        multiples of the 14-pixel patch.
        """
        height, width = (int(v) for v in size)
        if height % PATCH_SIZE or width % PATCH_SIZE:
            raise ValueError(f"Inference size {height}x{width} is not a multiple of {PATCH_SIZE}.")
        resized = F.interpolate(rgb, size=(height, width), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        normalised = (resized - self.rgb_mean.to(resized.dtype)) / self.rgb_std.to(resized.dtype)
        return self._append_prior(normalised)

    @torch.no_grad()
    def infer_image(self, rgb, infer_size=None):
        """Predict relative depth for an RGB batch in [0, 1].

        ``rgb`` has shape (B, 3, H, W), or (3, H, W) for one frame. The network
        runs at ``infer_size`` (height, width); by default that is the largest
        patch-aligned size that fits the frame, 1078 x 1918 for 1080p. As in the
        official Depth Anything V2 inference, the prediction is mapped back to
        H x W bilinearly; the result has shape (B, H, W).
        """
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        height, width = rgb.shape[-2:]
        if infer_size is None:
            infer_size = (height // PATCH_SIZE * PATCH_SIZE, width // PATCH_SIZE * PATCH_SIZE)
        depth = self.forward(self.prepare_input(rgb, infer_size))
        return F.interpolate(depth.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=True).squeeze(1)


PATCH_EMBED_KEY = "pretrained.patch_embed.proj.weight"


def inflate_patch_embed_weight(state_dict, in_chans, key=PATCH_EMBED_KEY, init="zero"):
    """Resize a pretrained patch projection without changing its RGB kernels."""
    if key not in state_dict:
        return state_dict
    w = state_dict[key]
    cur = w.shape[1]
    if cur == in_chans:
        return state_dict
    if in_chans < cur:
        # Reducing channels retains the leading pretrained kernels.
        state_dict[key] = w[:, :in_chans].clone()
        return state_dict
    # Expanding channels copies existing kernels and initialises only new ones.
    extra = in_chans - cur
    if init == "avg":
        # The sensitivity run starts the prior kernel at the mean RGB response.
        fill = w.mean(dim=1, keepdim=True).repeat(1, extra, 1, 1)
    else:
        # Zero keeps the initial four-channel RGB response equal to the baseline.
        fill = torch.zeros((w.shape[0], extra, w.shape[2], w.shape[3]), dtype=w.dtype, device=w.device)
    state_dict[key] = torch.cat([w, fill], dim=1)
    return state_dict
