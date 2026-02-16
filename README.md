# ZS-KAN-Denoising

Official implementation for **ZS-KAN: Zero-shot Image Denoising with Lightweight Kolmogorov-Arnold Networks**.

- Paper: *IEEE Transactions on Biomedical Engineering* (2025)
- DOI: [10.1109/TBME.2025.3640551](https://doi.org/10.1109/TBME.2025.3640551)

This repository provides:
- A unified CLI (`zskan`) for single-image denoising and dataset-level evaluation

## Table of Contents

- [Abstract](#abstract)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [ZS-KAN Network Architecture](#zs-kan-network-architecture)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Citation](#citation)
- [License](#license)

## Abstract

ZS-KAN is a lightweight zero-shot image denoising method designed for data-limited and edge-computing scenarios. It combines the efficiency of convolutional neural networks with the strong function-approximation ability of Kolmogorov-Arnold Networks (KANs).

Across both synthetic and real noisy images, ZS-KAN achieves comparable or better performance than state-of-the-art zero-shot denoisers while using only 1%–25% of their parameters, with better preservation of fine details in many cases.

These results show that KAN-based lightweight denoisers are practical for real-world deployment, especially in medical imaging workflows.

## Installation

Tested environment:
- Python `3.8.19`
- PyTorch `2.2.1`
- CUDA `12.1`

### 1) Clone project

```bash
git clone https://github.com/Jayx-Wang/ZS-KAN.git
cd ZS-KAN
```

### 2) Create environment

```bash
python -m venv zskan
source zskan/bin/activate
pip install -U pip
```

### 3) Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA/CPU targets, use [official PyTorch instructions](https://pytorch.org/get-started/locally/).

### 4) Install project dependencies

Portable install:

```bash
pip install -r requirements.txt
pip install -e .
```

Reproducible install (closest to published setup):

```bash
pip install -r requirements-lock.txt
pip install -e .
```

### 5) Check CLI

```bash
zskan --help
```

## Data Preparation

### A) Kodak24 (for synthetic-noise experiments)

```bash
bash scripts/download_kodak24.sh
```

Expected folder:

```text
data/kodak24/clean
```

### B) Microscopy pairs (for real-noise experiments)

This repo already includes sample microscopy image pairs:

```text
data/microscopy/clean
data/microscopy/noisy
```

Filenames in `clean/` and `noisy/` must match one-to-one and have the same name.
Current built-in pairs:
- `TwoPhoton_BPAE_B_4.png` <-> `TwoPhoton_BPAE_B_4.png`
- `Confocal_MICE_4.png` <-> `Confocal_MICE_4.png`

## ZS-KAN Network Architecture

ZS-KAN uses a lightweight CNN front-end followed by a final `KAN_Convolutional_Layer` (`kernel_size=1x1`, `n_convs=1`).

Two common variants are:

| Variant | Core setting | Approx. parameters | Default image type |
| --- | --- | --- | --- |
| 6k ZS-KAN | `chan_embed=25`, two 3x3 hidden conv blocks | ~6K | `gray` |
| 24k ZS-KAN | `chan_embed=35`, three 3x3 hidden conv blocks | ~24K | `color (and non-gray)` |

To adjust network size, modify `src/zskan_denoising/models/zs_models.py`:
- `chan_embed` in `ZS_KAN.__init__`
- Number of CNN layers (for example, enable/disable `conv3`)

## Quick Start

All outputs are saved to `outputs/...`.
- By default, no cropping is applied. To enable center-crop, add `--crop-size <N>`.

### 1) Single Image Denoising

Kodak synthetic + ZS-KAN:

```bash
zskan denoise-single \
  --method zs \
  --model zs_kan \
  --img-type color \
  --noise-source synthetic \
  --noise-type gauss \
  --noise-level 25 \
  --clean-img-path data/kodak24/clean/kodim01.png \
  --output-dir outputs/quickstart_kodak_zs \
  --device cuda
```

Microscopy real + ZS-KAN:

```bash
zskan denoise-single \
  --method zs \
  --model zs_kan \
  --img-type gray \
  --noise-source real \
  --clean-img-path data/microscopy/clean/TwoPhoton_BPAE_B_4.png \
  --noisy-img-path data/microscopy/noisy/TwoPhoton_BPAE_B_4.png \
  --output-dir outputs/quickstart_micro_zs \
  --device cuda
```

For each run, the directory contains:
- `clean.png`
- `noisy.png`
- `denoised.png`
- `metrics.json` (PSNR/SSIM/MS-SSIM/runtime)

### 2) Evaluation on Datasets

Synthetic noise on Kodak:

```bash
zskan evaluate-dataset \
  --method zs \
  --model zs_kan \
  --img-type color \
  --noise-source synthetic \
  --noise-type poiss \
  --noise-level 50 \
  --data-folder data/kodak24/clean \
  --output-dir outputs/eval_kodak_zs \
  --device cuda
```

For other methods, refer to:
- `zskan denoise-single --help`
- `zskan evaluate-dataset --help`

Optional residual-noise analysis:

```bash
zskan analyze-noise \
  --img-type gray \
  --clean-img-path data/microscopy/clean/TwoPhoton_BPAE_B_4.png \
  --noisy-img-path data/microscopy/noisy/TwoPhoton_BPAE_B_4.png \
  --output-path outputs/noise_analysis/TwoPhoton_BPAE_B_4.png
```

## CLI Reference

### `zskan denoise-single`

Core options:
- `--method {zs,bm3d}`
- `--model {zs_n2n,zs_kan,zs_mkan}` (used when `--method zs`)
  - `zs_mkan` now auto-adapts its intermediate channel width for gray/RGB input.
- `--noise-source {synthetic,real}`
- `--img-type {gray,color}`
- `--crop-size` (optional center-crop size; default is no crop)
- `--clean-img-path ...`
- `--noisy-img-path ...` (required for real-noise)
- `--output-dir ...`

Training control (ZS):
- `--epochs`
- `--lr`
- `--step-size`
- `--gamma`
- `--device` (recommended: `cuda`)

BM3D control:
- `--sigma-bm3d`

### `zskan evaluate-dataset`

Core options:
- `--method {zs,bm3d}`
- `--model {zs_n2n,zs_kan,zs_mkan}`
- `--noise-source {synthetic,real}`
- `--crop-size` (optional center-crop size; default is no crop)
- `--data-folder ...`
- `--max-images ...`

Output report:
- `dataset_metrics.json` with per-image metrics and averages.

### `zskan analyze-noise`

- Computes residual histogram, autocorrelation, and frequency spectrum.
- Saves one summary plot image.

## Citation

```bibtex
@article{wang2025zskan,
  title={ZS-KAN: Zero-shot Image Denoising with Lightweight Kolmogorov-Arnold Networks},
  author={Wang, Jianxu and Wang, Ge},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2025},
  doi={10.1109/TBME.2025.3640551}
}
```

## License

MIT License. See `LICENSE` for details.
