from pathlib import Path

import pytest


def test_denoise_single_smoke(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_msssim")
    pytest.importorskip("bm3d")

    from PIL import Image
    import numpy as np

    from zskan_denoising.engine import denoise_single

    clean_path = tmp_path / "clean.png"
    noisy_path = tmp_path / "noisy.png"

    base = (np.random.rand(64, 64) * 255).astype(np.uint8)
    noisy = np.clip(base + np.random.normal(0, 8, size=base.shape), 0, 255).astype(np.uint8)
    Image.fromarray(base, mode="L").save(clean_path)
    Image.fromarray(noisy, mode="L").save(noisy_path)

    out_dir = tmp_path / "out"
    metrics = denoise_single(
        method="zs",
        model_name="zs_n2n",
        img_type="gray",
        noise_source="real",
        noise_type="gauss",
        noise_level=25.0,
        clean_img_path=str(clean_path),
        noisy_img_path=str(noisy_path),
        output_dir=str(out_dir),
        crop_size=0,
        epochs=1,
        lr=1e-3,
        step_size=1,
        gamma=0.5,
        sigma_bm3d=0.1,
        device="cpu",
    )

    assert "denoised" in metrics
    assert (out_dir / "denoised.png").exists()
    assert (out_dir / "metrics.json").exists()
