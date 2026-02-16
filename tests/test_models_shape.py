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


def test_zs_kan_auto_config_gray_vs_color():
    pytest.importorskip("torch")
    from zskan_denoising.models import build_model

    gray_model = build_model("zs_kan", 1, device="cpu")
    color_model = build_model("zs_kan", 3, device="cpu")

    assert gray_model.conv1.out_channels == 25
    assert gray_model.use_conv3 is False
    assert gray_model.conv3 is None

    assert color_model.conv1.out_channels == 35
    assert color_model.use_conv3 is True
    assert color_model.conv3 is not None
