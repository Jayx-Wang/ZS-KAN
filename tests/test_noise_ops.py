import pytest


def test_add_noise_clamped_range():
    torch = pytest.importorskip("torch")
    from zskan_denoising.utils import add_noise

    x = torch.rand(1, 1, 16, 16)
    y_gauss = add_noise(x, "gauss", 25.0)
    y_poiss = add_noise(x, "poiss", 30.0)
    assert y_gauss.min().item() >= 0
    assert y_gauss.max().item() <= 1
    assert y_poiss.min().item() >= 0
    assert y_poiss.max().item() <= 1
