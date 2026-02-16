"""ZS-KAN-Denoising package."""

__all__ = ["build_model", "denoise_single", "evaluate_dataset"]


def build_model(*args, **kwargs):
    from .models import build_model as _build_model

    return _build_model(*args, **kwargs)


def denoise_single(*args, **kwargs):
    from .engine import denoise_single as _denoise_single

    return _denoise_single(*args, **kwargs)


def evaluate_dataset(*args, **kwargs):
    from .engine import evaluate_dataset as _evaluate_dataset

    return _evaluate_dataset(*args, **kwargs)
