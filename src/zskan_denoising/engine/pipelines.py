import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms

from zskan_denoising.engine.training import predict_denoised, train_step
from zskan_denoising.metrics import compute_quality_metrics
from zskan_denoising.models import build_model
from zskan_denoising.utils import add_noise, bm3d_denoise_torch, list_images, load_image, save_image_tensor


def _build_transform(crop_size: int):
    transform_ops = []
    if crop_size and crop_size > 0:
        transform_ops.append(transforms.CenterCrop(crop_size))
    transform_ops.append(transforms.ToTensor())
    return transforms.Compose(transform_ops)


def _pick_schedule(model, epochs: Optional[int], lr: Optional[float]):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    default_epochs = 1500 if params < 10000 else 2000
    default_lr = 0.002 if params < 10000 else 0.0015
    return params, (epochs if epochs is not None else default_epochs), (lr if lr is not None else default_lr)


def _serialize_metrics(path: Path, metrics: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def _resolve_real_noisy_pair(clean_img_path: str) -> str:
    clean_path = Path(clean_img_path)
    noisy_dir = clean_path.parent.parent / "noisy"

    candidate = noisy_dir / clean_path.name
    if candidate.exists():
        return str(candidate)

    raise FileNotFoundError(
        f"No matching noisy image found for clean image '{clean_path.name}' in '{noisy_dir}'. "
        "Expected the same filename in both clean/ and noisy/ directories."
    )


def denoise_single(
    method: str,
    model_name: str,
    img_type: str,
    noise_source: str,
    noise_type: str,
    noise_level: float,
    clean_img_path: str,
    noisy_img_path: Optional[str],
    output_dir: str,
    crop_size: int,
    epochs: Optional[int],
    lr: Optional[float],
    step_size: int,
    gamma: float,
    sigma_bm3d: float,
    device: str,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable. Falling back to CPU.")
        device = "cpu"

    transform = _build_transform(crop_size)
    clean_img = transform(load_image(clean_img_path, img_type)).unsqueeze(0).to(device)
    if noise_source == "synthetic":
        noisy_img = add_noise(clean_img, noise_type, noise_level)
    elif noise_source == "real":
        if not noisy_img_path:
            raise ValueError("--noisy-img-path is required when --noise-source real")
        noisy_img = transform(load_image(noisy_img_path, img_type)).unsqueeze(0).to(device)
    else:
        raise ValueError("noise_source must be 'synthetic' or 'real'")

    save_image_tensor(clean_img, str(out / "clean.png"))
    save_image_tensor(noisy_img, str(out / "noisy.png"))

    if method == "bm3d":
        start = time.time()
        denoised = torch.clamp(bm3d_denoise_torch(noisy_img, sigma=sigma_bm3d).to(device), 0, 1)
        elapsed = time.time() - start
        losses = []
        model_params = 0
    elif method == "zs":
        model = build_model(model_name, clean_img.shape[1], device).to(device)
        model_params, run_epochs, run_lr = _pick_schedule(model, epochs, lr)
        optimizer = optim.Adam(model.parameters(), lr=run_lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        losses = []
        start = time.time()
        for _ in tqdm(range(run_epochs), desc="Training", leave=False):
            losses.append(train_step(model, optimizer, noisy_img))
            scheduler.step()
        denoised = predict_denoised(model, noisy_img)
        elapsed = time.time() - start
    else:
        raise ValueError("method must be 'zs' or 'bm3d'")

    noisy_metrics = compute_quality_metrics(clean_img, noisy_img)
    denoised_metrics = compute_quality_metrics(clean_img, denoised)
    metrics = {
        "method": method,
        "model": model_name,
        "image_type": img_type,
        "noise_source": noise_source,
        "noise_type": noise_type,
        "noise_level": noise_level,
        "sigma_bm3d": sigma_bm3d,
        "model_parameters": model_params,
        "runtime_seconds": elapsed,
        "noisy": noisy_metrics,
        "denoised": denoised_metrics,
        "losses": losses,
    }

    save_image_tensor(denoised, str(out / "denoised.png"))
    _serialize_metrics(out / "metrics.json", metrics)
    return metrics


def evaluate_dataset(
    method: str,
    model_name: str,
    img_type: str,
    noise_source: str,
    noise_type: str,
    noise_level: float,
    data_folder: str,
    output_dir: str,
    crop_size: int,
    epochs: Optional[int],
    lr: Optional[float],
    step_size: int,
    gamma: float,
    sigma_bm3d: float,
    device: str,
    max_images: Optional[int],
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable. Falling back to CPU.")
        device = "cpu"
    transform = _build_transform(crop_size)

    if noise_source == "synthetic":
        path_list = list_images(data_folder)
    elif noise_source == "real":
        path_list = list_images(str(Path(data_folder) / "clean"))
    else:
        raise ValueError("noise_source must be 'synthetic' or 'real'")

    if max_images is not None:
        path_list = path_list[:max_images]

    if not path_list:
        raise ValueError(f"No images found in '{data_folder}'")

    metrics_per_image = []
    for idx, img_path in enumerate(path_list, start=1):
        print(f"Working on image_{idx}/{len(path_list)}: {img_path}")
        clean_img = transform(load_image(img_path, img_type)).unsqueeze(0).to(device)

        if noise_source == "synthetic":
            noisy_img = add_noise(clean_img, noise_type, noise_level)
        else:
            paired_noisy = _resolve_real_noisy_pair(img_path)
            noisy_img = transform(load_image(paired_noisy, img_type)).unsqueeze(0).to(device)

        if method == "bm3d":
            start = time.time()
            denoised = torch.clamp(bm3d_denoise_torch(noisy_img, sigma=sigma_bm3d).to(device), 0, 1)
            elapsed = time.time() - start
            model_params = 0
        elif method == "zs":
            model = build_model(model_name, clean_img.shape[1], device).to(device)
            model_params, run_epochs, run_lr = _pick_schedule(model, epochs, lr)
            optimizer = optim.Adam(model.parameters(), lr=run_lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            start = time.time()
            for _ in tqdm(range(run_epochs), desc=f"Training {idx}", leave=False):
                train_step(model, optimizer, noisy_img)
                scheduler.step()
            denoised = predict_denoised(model, noisy_img)
            elapsed = time.time() - start
        else:
            raise ValueError("method must be 'zs' or 'bm3d'")

        current = {
            "image": Path(img_path).name,
            "model_parameters": model_params,
            "runtime_seconds": elapsed,
            "noisy": compute_quality_metrics(clean_img, noisy_img),
            "denoised": compute_quality_metrics(clean_img, denoised),
        }
        metrics_per_image.append(current)

    avg = {
        "avg_psnr": sum(m["denoised"]["psnr"] for m in metrics_per_image) / len(metrics_per_image),
        "avg_ssim": sum(m["denoised"]["ssim"] for m in metrics_per_image) / len(metrics_per_image),
        "avg_ms_ssim": sum(m["denoised"]["ms_ssim"] for m in metrics_per_image) / len(metrics_per_image),
        "avg_runtime_seconds": sum(m["runtime_seconds"] for m in metrics_per_image) / len(metrics_per_image),
    }

    report = {
        "method": method,
        "model": model_name,
        "dataset_size": len(metrics_per_image),
        "summary": avg,
        "images": metrics_per_image,
    }
    _serialize_metrics(out / "dataset_metrics.json", report)
    return report
