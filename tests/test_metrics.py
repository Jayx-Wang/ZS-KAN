import pytest


def test_mse_and_metrics_keys():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_msssim")

    from zskan_denoising.metrics import compute_quality_metrics, mse

    x = torch.rand(1, 1, 32, 32)
    y = torch.clamp(x + 0.05 * torch.randn_like(x), 0, 1)
    mse_val = mse(x, y)
    metrics = compute_quality_metrics(x, y)

    assert mse_val.item() >= 0
    assert {"mse", "psnr", "ssim", "ms_ssim"}.issubset(metrics.keys())
