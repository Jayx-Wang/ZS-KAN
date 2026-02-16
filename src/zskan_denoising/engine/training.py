import torch
import torch.nn.functional as F

from zskan_denoising.metrics import mse


def pair_downsampler(img: torch.Tensor):
    c = img.shape[1]
    filter1 = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], device=img.device).repeat(c, 1, 1, 1)
    filter2 = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], device=img.device).repeat(c, 1, 1, 1)
    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)
    return output1, output2


def loss_func(noisy_img: torch.Tensor, model) -> torch.Tensor:
    noisy1, noisy2 = pair_downsampler(noisy_img)
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)
    loss_res = 0.5 * (mse(noisy1, pred2) + mse(noisy2, pred1))

    noisy_denoised = noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)
    loss_cons = 0.5 * (mse(pred1, denoised1) + mse(pred2, denoised2))
    return loss_res + loss_cons


def train_step(model, optimizer, noisy_img: torch.Tensor) -> float:
    loss = loss_func(noisy_img, model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def predict_denoised(model, noisy_img: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.clamp(noisy_img - model(noisy_img), 0, 1)
