from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image


def load_image(img_path: str, img_type: str) -> Image.Image:
    if img_type == "gray":
        return Image.open(img_path).convert("L")
    if img_type == "color":
        return Image.open(img_path).convert("RGB")
    raise ValueError(f"Unsupported image type: {img_type}. Use 'gray' or 'color'.")


def save_image_tensor(tensor: torch.Tensor, file_path: str) -> None:
    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tensor = tensor.detach().cpu()
    n_chan = tensor.shape[1]
    if n_chan == 1:
        image_array = (torch.squeeze(tensor).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(image_array, mode="L").save(out_path)
        return

    if n_chan == 3:
        image_array = (
            tensor.squeeze(0)
            .mul(255)
            .add(0.5)
            .clamp(0, 255)
            .permute(1, 2, 0)
            .to(torch.uint8)
            .numpy()
        )
        Image.fromarray(image_array).save(out_path)
        return

    raise ValueError(f"Unsupported channel count: {n_chan}")


def list_images(folder_path: str) -> List[str]:
    folder = Path(folder_path)
    exts: Iterable[str] = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff"}
    return sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in exts)
