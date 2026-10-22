# PitchDepth: Geometric Prior Conditioning of a Pretrained Depth Foundation Model for Football Broadcast Video

PitchDepth investigates whether geometric information from the football pitch
can improve monocular depth estimation in broadcast video. It adapts a
pretrained Depth Anything V2 model by adding a fourth input channel derived
from the average ground-depth structure of the SoccerNet training data.

The project compares two models through the same data, training and evaluation
pipeline:

- **Depth Anything baseline:** fine-tuned Depth Anything V2 with RGB input.
- **PitchDepth:** the same encoder, decoder and training procedure with the
  ground prior appended to RGB.

This repository revisits the experiment from my MSc dissertation in Computing
in Big Data Analytics and Artificial Intelligence at Atlantic Technological
University (ATU) Donegal (2026), supervised by Dr Shagufta Henna. It is a new
implementation with a revised protocol: three seeds per model, fixed-length
training and checkpoint selection on validation AbsRel.

## Results

Results, run logs and figures will be added once both models have been trained
and evaluated with this code.

Each model is trained with three seeds, and every comparison is paired by
seed. The evaluation compares the two models twice:

- **Validation-selected checkpoints:** each run keeps the epoch with the
  lowest validation AbsRel.
- **Final-epoch checkpoints:** every run trains for the same ten epochs, so the
  final weights are epoch-matched by construction. A difference here cannot come
  from one run happening to be selected at a better point in training.

Predictions are aligned to each target with a least-squares scale and shift
and scored with the five SoccerNet-Depth errors on the 1,423-frame test split,
per frame and per clip (20 clips). Results are reported as the mean and
standard deviation over seeds.

## Method

SoccerNet depth labels are stored as 16-bit `depth_r` PNG files. The data loader
converts each label to the model target space:

```text
target = 1 - depth_r / 65535
```

`data/ground_prior.py` averages the training targets pixel by pixel. It retains
the lower two thirds of the average map, where the pitch normally appears, and
sets the upper third to zero. The resulting array is stored in
`priors/ground_prior_foot.npy`. Its JSON sidecar records the source split,
dimensions, frame count and fixed normalisation statistics; the PNG is a visual
preview. The prior is empirical and does not use camera calibration or a
homography.

For PitchDepth, the DINOv2 patch-embedding convolution accepts four channels.
When three-channel weights initialise the model, the RGB kernels are copied and
the new kernel is zero-initialised by default, so training starts numerically
identical to the RGB baseline. The `avg` sensitivity setting instead
initialises it from the mean RGB kernel. The DINOv2 encoder and DPT decoder are
otherwise unchanged.

During training, RGB and depth are resized to the configured size. Half of the
training frames receive standard colour jitter (brightness, contrast and
saturation within ±20%, hue within ±0.02) on RGB only, and half are flipped
horizontally, with depth, mask and prior flipped together. RGB is
ImageNet-normalised before the separately normalised prior is appended.
Evaluation and inference pass raw RGB to the model, which resizes it
bicubically, normalises it, attaches the same saved prior, and maps the
prediction back to the frame size bilinearly.

The training objective (`ssitrim`) is the MiDaS loss: prediction and target are
each normalised by their median and mean absolute deviation, the largest 20% of
residuals in each image are discarded, and the rest form the data term, plus
0.5 times a gradient-matching term over four scales. Optimisation follows the
Depth Anything V2 fine-tuning setup: AdamW with the decoder at ten times the
encoder learning rate, weight decay 0.01, and polynomial decay with power 0.9. Validation after
each epoch scores the model at the training crop, and the epoch with the
lowest validation AbsRel is kept as `best_model.pth`.

Evaluation reports absolute relative error, squared relative error, RMSE,
log RMSE and SILog in `depth_r / 65535` space.

## Repository structure

| Path | Purpose |
| --- | --- |
| `train.py` | Fine-tuning on one GPU or several with `torchrun`. |
| `run_experiment.py` | Prints or runs the registered training configurations, per seed. |
| `evaluate_pitchdepth.py` | Evaluates PitchDepth, the RGB baseline or ZoeDepth per frame and per clip. |
| `export_challenge.py` | Writes 16-bit predictions for the unlabelled challenge frames. |
| `visualize_pitchdepth_figures_5_9_to_5_13.py` | Recreates the dissertation's comparison figures. |
| `generate_xai_explanations.py` | Creates Grad-CAM, SmoothGrad and LIME figures. |
| `data/discovery.py` | Finds the splits and pairs each colour frame with its label. |
| `data/dataset.py` | Training and evaluation samples, augmentation and the data loader. |
| `data/ground_prior.py` | Builds, loads and normalises the ground prior. |
| `loss/depth_losses.py` | Training losses. |
| `metrics/depth_metrics.py` | Least-squares alignment and the five depth errors. |
| `training/` | Model construction, gradient checkpointing, optimiser, schedule and checkpoints. |
| `models/DAv2/depth_anything_v2/` | Depth Anything V2 and DINOv2 model code with the four-channel input. |
| `utils/` | CUDA device and precision handling. |
| `scripts/get_weights.py` | Base-weight download. |
| `scripts/summarise_seeds.py` | Mean and standard deviation over seeds, and paired differences between models. |
| `configs/experiments.json` | Baseline, PitchDepth and sensitivity settings, and the seeds. |
| `tests/` | Synthetic-data smoke checks. |
| `priors/` | The generated football prior, metadata and preview. |

## Requirements

- Python 3.9 or later.
- A CUDA-capable NVIDIA GPU and a compatible CUDA build of PyTorch for training,
  evaluation and inference. The registered settings fit one 8 GB RTX 3070 Ti.
- Internet access for the initial Depth Anything V2 download and optional
  zero-shot ZoeDepth download.
- About 80 GB of free space while the full dataset archives and extracted files
  coexist. The compressed archives total roughly 41 GB.
- About 7 GB per training run for `best_model.pth`, `final_model.pth` and
  `last.ckpt`, so roughly 42 GB for three seeds of both models.

The CPU smoke tests cover data handling, losses, metrics, the training
interfaces and a small model without requiring the SoccerNet dataset. The main
entry points deliberately stop when CUDA is unavailable.

## Installation

Run all commands from the repository root. Install a CUDA-enabled PyTorch build
before the remaining requirements. Replace `cu121` with the wheel index that
matches the machine.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The CUDA check must print `True` before running the main workflows. Weights &
Biases is optional: install `wandb` and pass `--wandb` to `train.py` to log
there as well.

## Dataset setup

The project uses the football portion of SoccerNet SN-Depth-2025:

https://huggingface.co/datasets/SoccerNet/SN-Depth-2025

The labelled `train`, `valid` and `test` splits contain paired RGB and 16-bit
depth images. Training labels supply both the model targets and the fixed prior,
validation data is used during training, and test data is used for evaluation.
The unlabelled `challenge` split is used only for prediction export.

Download the splits from the dataset page and extract them under
`SoccerNet/SN-Depth-2025`, or set `PITCHDEPTH_DATA_DIR` to another location.
The loader searches recursively and accepts either layout:

```text
<data_dir>/{train,valid,test,challenge}/game_*/video_*/{color,depth_r}/
<data_dir>/depth-football/{Train,Validation,Test,Challenge}/game_*/video_*/{color,depth_r}/
```

RGB and depth files are paired by filename. The challenge split has no
`depth_r` folder. If a `masks` folder beside `color` holds a `.pt` mask for a
frame, the loader uses it; otherwise every pixel is valid.

## Ground prior

The checked-in prior was generated from 4,071 football training frames at
1080 x 1920. Rebuild it from the selected dataset revision with:

```bash
python -m data.ground_prior --data_dir "SoccerNet/SN-Depth-2025" --split Train --sport foot --crop_fraction 0.6667 --depth_space model --out_dir priors --name ground_prior_foot
```

This replaces `priors/ground_prior_foot.npy`, its metadata and its preview.
The metadata records the `--data_dir` as given, so pass a relative path to keep
machine-specific paths out of it.

## Base weights and training

Download the Depth Anything V2 ViT-L weights:

```bash
python scripts/get_weights.py --encoder vitl
```

The file is saved as
`checkpoints/DepthAnythingV2/depth_anything_v2_vitl.pth`, where `train.py`
looks for it by default.

List the registered experiments, print the commands for every seed, or run
them:

```bash
python run_experiment.py --list
python run_experiment.py pitchdepth --all-seeds --print-eval
python run_experiment.py baseline --all-seeds --execute
python run_experiment.py pitchdepth --all-seeds --execute
```

`--seed N` runs a single seed instead. Each run writes to
`<checkpoint_dir>/seed<N>`, for example `checkpoints/pitchdepth/seed1`.

The base settings in `configs/experiments.json` are ViT-L, 392 x 672 training
tensors (height x width), bfloat16 with gradient checkpointing on the encoder
blocks, batch size 1 with 4-step gradient accumulation (effective batch 4), the
`ssitrim` loss, AdamW with an encoder learning rate of `5e-6`, the decoder at
ten times that, weight decay `0.01`, polynomial decay, ten epochs and seeds 0,
1 and 2. The baseline and PitchDepth runs differ only in the input channel and
prior settings. The registry also holds the loss, normalisation, prior scale,
initialisation, learning-rate, schedule and duration sensitivity runs.

All sizes on the command line are `height,width`. Each run writes:

```text
run_config.json     the resolved arguments
metrics_log.jsonl   training loss and validation metrics per epoch
best_model.pth      weights at the epoch with the lowest validation AbsRel
final_model.pth     weights after the last epoch
last.ckpt           full training state after the latest epoch
epoch_XX.pth        weights every --save_every epochs (off by default)
```

Add `--resume` to the same command to continue from `last.ckpt`.

## Evaluation

`python run_experiment.py <experiment> --all-seeds --print-eval` prints the
evaluation commands for each seed's `best_model.pth` and `final_model.pth`,
tagged `<experiment>_seed<N>_best` and `<experiment>_seed<N>_final`. They use
an internal inference size of 392 x 672 (`--infer_size` is `height,width`).
For example:

```bash
python evaluate_pitchdepth.py --model_type pitchdepth --encoder vitl --checkpoint checkpoints/pitchdepth/seed0/final_model.pth --prior_path priors/ground_prior_foot.npy --prior_norm standardize --prior_scale 1.0 --data_dir "SoccerNet/SN-Depth-2025" --infer_size 392,672 --tag pitchdepth_seed0_final
```

The zero-shot reference needs no checkpoint:

```bash
python evaluate_pitchdepth.py --model_type zoedepth --data_dir "SoccerNet/SN-Depth-2025" --infer_size 392,672 --tag zoedepth
```

ZoeDepth is loaded through `torch.hub`; its first run downloads the repository
and weights. Each evaluation writes a timestamped JSON file, per-frame CSV,
per-clip CSV and one row in `evaluation_results/summary.csv`. Summarise the
seeds and the paired PitchDepth-minus-baseline differences with:

```bash
python scripts/summarise_seeds.py --compare baseline pitchdepth
```

## Challenge export

```bash
python export_challenge.py --checkpoint_path checkpoints/pitchdepth/seed0/best_model.pth --in_chans 4 --prior_path priors/ground_prior_foot.npy --input_dir "SoccerNet/SN-Depth-2025/challenge" --output_dir predictions/pitchdepth --flip
```

For the RGB baseline, use its checkpoint with `--in_chans 3`. The script writes
one 16-bit PNG per frame named
`foot_game_<game>_video_<video>_depth_r_<frame>.png`. `--flip` averages each
prediction with that of the mirrored frame.

## Dissertation figures and explanations

Recreate Chapter 5 Figures 5.9--5.13 from two trained checkpoints. The
`--zoedepth` flag adds the zero-shot comparison.

```bash
python visualize_pitchdepth_figures_5_9_to_5_13.py --data_dir "SoccerNet/SN-Depth-2025" --encoder vitl --baseline_ckpt checkpoints/baseline/seed0/best_model.pth --pitchdepth_ckpt checkpoints/pitchdepth/seed0/best_model.pth --prior_path priors/ground_prior_foot.npy --zoedepth --out_dir outputs/chapter5_figures
```

Generate the Grad-CAM, SmoothGrad and LIME explanations with:

```bash
python generate_xai_explanations.py --data_dir "SoccerNet/SN-Depth-2025" --encoder vitl --baseline_ckpt checkpoints/baseline/seed0/best_model.pth --pitchdepth_ckpt checkpoints/pitchdepth/seed0/best_model.pth --prior_path priors/ground_prior_foot.npy --out_dir outputs/chapter5_figures/xai
```

Both scripts default to a 392 x 672 inference size and seed 42. The figure
manifests record the model, frame, method and purpose of every output.

## Reproducibility and verification

The experiment registry fixes the training settings and the seeds. Each run
records its resolved arguments and per-epoch measurements, and evaluation JSON
files record their arguments, timestamp, frame count, aggregate metrics and
standard deviations.

Run the lightweight checks with:

```bash
python tests/test_smoke.py
```

They build a temporary SoccerNet-style dataset and check prior generation and
normalisation, both supported dataset layouts, three- and four-channel model
construction, patch-weight inflation, forward and backward passes, inference
shapes, the losses (including scale-and-shift invariance, trimming and empty
masks), alignment and metrics, the training defaults, the optimiser groups,
the learning-rate schedules, gradient checkpointing, the per-seed experiment
commands and the seed summary. Evaluation and figure subprocess checks run only
when CUDA is available.

Exact reproduction also depends on the dataset revision, CUDA and PyTorch
versions, GPU hardware and downloaded base weights. GPU training is not
bit-for-bit deterministic even with a fixed seed, which is one reason each
model is trained with several seeds.

## Practical limitations

- ViT-L training and full-resolution inference require substantial GPU memory.
- The fixed prior represents average training-camera geometry and does not
  adapt to a new viewpoint.
- Three seeds per model give a first estimate of run-to-run variance; more
  seeds would narrow it.
- The dataset is synthetic, so results make no claim about transfer to real
  broadcast footage.

## Acknowledgements and licences

- **Depth Anything V2.** `models/DAv2/` contains the model code from
  [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) by
  Lihe Yang and colleagues ([arXiv:2406.09414](https://arxiv.org/abs/2406.09414)),
  under the Apache-2.0 licence retained in `models/DAv2/LICENSE`, with the
  four-channel input and the inference path added here. The fine-tuning
  optimiser setup (decoder at ten times the encoder learning rate, weight decay
  0.01, polynomial decay) and the bilinear output resizing follow its published
  code. The
  upstream project states that the Small weights are Apache-2.0 and the Base,
  Large and Giant weights are CC-BY-NC-4.0; users must observe the licence of
  the downloaded ViT-L weights.
- **MiDaS losses.** The trimmed and least-squares scale-and-shift-invariant
  losses and the multi-scale gradient-matching term follow Ranftl et al.,
  "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot
  Cross-Dataset Transfer", IEEE TPAMI, 2022.
- **SILog.** The scale-invariant log loss follows Eigen et al., "Depth Map
  Prediction from a Single Image using a Multi-Scale Deep Network", NeurIPS,
  2014.
- **SoccerNet-Depth.** The dataset and the five-error evaluation protocol come
  from Leduc et al., "SoccerNet-Depth: a Scalable Dataset for Monocular Depth
  Estimation in Sports Videos", CVPR Workshops, 2024. SN-Depth-2025 is
  distributed under GPL-3.0, and the ground prior in `priors/` is computed from
  its training labels.
- **ZoeDepth** is loaded unmodified through `torch.hub` as a zero-shot
  reference.

The rest of the code is released under the MIT licence; see [`LICENSE`](LICENSE).
