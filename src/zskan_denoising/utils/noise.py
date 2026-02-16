import numpy as np
import torch
from bm3d import bm3d


def add_noise(x: torch.Tensor, noise_type: str, noise_level: float) -> torch.Tensor:
    if noise_type == "gauss":
        noisy = x + torch.normal(0, noise_level / 255.0, x.shape, device=x.device)
    elif noise_type == "poiss":
        noisy = torch.poisson(noise_level * x) / noise_level
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}. Use 'gauss' or 'poiss'.")
    return torch.clamp(noisy, 0, 1)


def bm3d_denoise_torch(noisy_tensor: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    noisy_np = noisy_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    if noisy_np.shape[2] == 1:
        denoised_np = bm3d(noisy_np[:, :, 0], sigma)[:, :, np.newaxis]
    elif noisy_np.shape[2] == 3:
        denoised_np = np.stack([bm3d(noisy_np[:, :, i], sigma) for i in range(3)], axis=-1)
    else:
        raise ValueError(f"Unsupported number of channels: {noisy_np.shape[2]}")

    denoised_tensor = torch.from_numpy(denoised_np).permute(2, 0, 1).unsqueeze(0)
    return denoised_tensor.to(device=noisy_tensor.device, dtype=noisy_tensor.dtype)
