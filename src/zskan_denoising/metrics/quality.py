import numpy as np
import torch


def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(gt, pred)


def compute_quality_metrics(clean_img: torch.Tensor, pred_img: torch.Tensor):
    from pytorch_msssim import ms_ssim, ssim

    mse_val = mse(clean_img, pred_img).item()
    psnr = float("inf") if mse_val == 0 else 10 * np.log10(1 / mse_val)
    ssim_val = ssim(clean_img, pred_img, data_range=1, size_average=False).item()
    ms_ssim_val = ms_ssim(clean_img, pred_img, data_range=1, size_average=False).item()

    return {
        "mse": mse_val,
        "psnr": psnr,
        "ssim": ssim_val,
        "ms_ssim": ms_ssim_val,
    }
