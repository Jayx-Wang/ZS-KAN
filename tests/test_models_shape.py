import pytest


def test_model_output_shape_gray_and_color():
    torch = pytest.importorskip("torch")
    from zskan_denoising.models import build_model

    for channels in [1, 3]:
        x = torch.rand(1, channels, 32, 32)
        for model_name in ["zs_n2n", "zs_kan", "zs_mkan"]:
            model = build_model(model_name, channels, device="cpu")
            y = model(x)
            assert tuple(y.shape) == tuple(x.shape)
