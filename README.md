# ZS-KAN-Denoising

Official implementation for **ZS-KAN: Zero-shot Image Denoising with Lightweight Kolmogorov-Arnold Networks**.

- Paper: *IEEE Transactions on Biomedical Engineering* (2025)
- DOI: [10.1109/TBME.2025.3640551](https://doi.org/10.1109/TBME.2025.3640551)

This repository provides:
- A unified CLI (`zskan`) for single-image denoising and dataset-level evaluation
- Zero-shot denoising models (`ZS-N2N`, `ZS-KAN`, `ZS-MKAN`)
- BM3D baseline integration
- Built-in microscopy sample pairs for out-of-the-box testing

## Table of Contents

- [Abstract](#abstract)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Reproduce Main Results](#reproduce-main-results)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## Abstract

Deep zero-shot denoising avoids paired clean/noisy training data by optimizing a lightweight network directly on the target image. This project implements ZS-KAN and related baselines for synthetic and real-noise settings, with practical scripts for single-image denoising, dataset evaluation, and residual-noise analysis.

## Installation

Tested environment:
- Python `3.8.19`
- PyTorch `2.2.1`
- CUDA `12.1`

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 2) Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA/CPU targets, use [official PyTorch instructions](https://pytorch.org/get-started/locally/).

### 3) Install project dependencies

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

### 4) Check CLI

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

Filenames in `clean/` and `noisy/` must match one-to-one.

## Quick Start

All outputs are saved to `outputs/...`.

### 1) Kodak synthetic + ZS-KAN

```bash
zskan denoise-single \
  --method zs \
  --model zs_kan \
  --img-type color \
  --noise-source synthetic \
  --noise-type poiss \
  --noise-level 80 \
  --clean-img-path data/kodak24/clean/kodim01.png \
  --output-dir outputs/quickstart_kodak_zs
```

### 2) Kodak synthetic + BM3D

```bash
zskan denoise-single \
  --method bm3d \
  --img-type color \
  --noise-source synthetic \
  --noise-type gauss \
  --noise-level 25 \
  --sigma-bm3d 0.1 \
  --clean-img-path data/kodak24/clean/kodim01.png \
  --output-dir outputs/quickstart_kodak_bm3d
```

### 3) Microscopy real + ZS-KAN

```bash
zskan denoise-single \
  --method zs \
  --model zs_kan \
  --img-type gray \
  --noise-source real \
  --clean-img-path data/microscopy/clean/micro_01.png \
  --noisy-img-path data/microscopy/noisy/micro_01.png \
  --output-dir outputs/quickstart_micro_zs
```

### 4) Microscopy real + BM3D

```bash
zskan denoise-single \
  --method bm3d \
  --img-type gray \
  --noise-source real \
  --sigma-bm3d 0.1 \
  --clean-img-path data/microscopy/clean/micro_01.png \
  --noisy-img-path data/microscopy/noisy/micro_01.png \
  --output-dir outputs/quickstart_micro_bm3d
```

For each run, the directory contains:
- `clean.png`
- `noisy.png`
- `denoised.png`
- `metrics.json` (PSNR/SSIM/MS-SSIM/runtime)

## Reproduce Main Results

### Dataset-level evaluation (ZS)

Synthetic noise on Kodak:

```bash
zskan evaluate-dataset \
  --method zs \
  --model zs_kan \
  --img-type color \
  --noise-source synthetic \
  --noise-type poiss \
  --noise-level 80 \
  --data-folder data/kodak24/clean \
  --output-dir outputs/eval_kodak_zs
```

Real microscopy pairs:

```bash
zskan evaluate-dataset \
  --method zs \
  --model zs_kan \
  --img-type gray \
  --noise-source real \
  --data-folder data/microscopy \
  --output-dir outputs/eval_micro_zs
```

### Dataset-level evaluation (BM3D)

```bash
zskan evaluate-dataset \
  --method bm3d \
  --img-type color \
  --noise-source synthetic \
  --noise-type gauss \
  --noise-level 25 \
  --sigma-bm3d 0.1 \
  --data-folder data/kodak24/clean \
  --output-dir outputs/eval_kodak_bm3d
```

### Residual-noise analysis

```bash
zskan analyze-noise \
  --img-type gray \
  --clean-img-path data/microscopy/clean/micro_01.png \
  --noisy-img-path data/microscopy/noisy/micro_01.png \
  --output-path outputs/noise_analysis/micro_01.png
```

## CLI Reference

### `zskan denoise-single`

Core options:
- `--method {zs,bm3d}`
- `--model {zs_n2n,zs_kan,zs_mkan}` (used when `--method zs`)
- `--noise-source {synthetic,real}`
- `--img-type {gray,color}`
- `--clean-img-path ...`
- `--noisy-img-path ...` (required for real-noise)
- `--output-dir ...`

Training control (ZS):
- `--epochs`
- `--lr`
- `--step-size`
- `--gamma`
- `--device`

BM3D control:
- `--sigma-bm3d`

### `zskan evaluate-dataset`

Core options:
- `--method {zs,bm3d}`
- `--model {zs_n2n,zs_kan,zs_mkan}`
- `--noise-source {synthetic,real}`
- `--data-folder ...`
- `--max-images ...`

Output report:
- `dataset_metrics.json` with per-image metrics and averages.

### `zskan analyze-noise`

- Computes residual histogram, autocorrelation, and frequency spectrum.
- Saves one summary plot image.

## Project Structure

```text
.
├── data
│   ├── kodak24
│   │   └── clean
│   └── microscopy
│       ├── clean
│       └── noisy
├── scripts
│   └── download_kodak24.sh
├── src
│   └── zskan_denoising
│       ├── cli
│       ├── core
│       ├── engine
│       ├── metrics
│       ├── models
│       ├── utils
│       ├── layers
│       │   ├── kan_conv_v1
│       │   └── kan_conv_v2
│       └── experimental
│           └── diffusion
├── third_party
│   └── kan_implementations
└── tests
```

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
