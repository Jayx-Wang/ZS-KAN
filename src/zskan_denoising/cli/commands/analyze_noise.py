from pathlib import Path

def add_analyze_noise_args(parser):
    parser.add_argument("--img-type", type=str, default="gray", choices=["gray", "color"])
    parser.add_argument("--clean-img-path", type=str, required=True)
    parser.add_argument("--noisy-img-path", type=str, required=True)
    parser.add_argument(
        "--crop-size",
        type=int,
        default=0,
        help="Optional center-crop size. If omitted or <=0, no cropping is applied.",
    )
    parser.add_argument("--output-path", type=str, default="outputs/analyze_noise/noise_analysis.png")


def run_analyze_noise(args):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from scipy.fftpack import fft2, fftshift
    from scipy.signal import correlate2d
    from scipy.stats import norm
    from torchvision import transforms

    from zskan_denoising.utils import load_image

    transform_ops = []
    if args.crop_size and args.crop_size > 0:
        transform_ops.append(transforms.CenterCrop(args.crop_size))
    transform_ops.append(transforms.ToTensor())
    transform = transforms.Compose(transform_ops)

    clean_img = transform(load_image(args.clean_img_path, args.img_type)).unsqueeze(0)
    noisy_img = transform(load_image(args.noisy_img_path, args.img_type)).unsqueeze(0)
    noise = noisy_img - clean_img

    mean_noise = torch.mean(noise).item()
    var_noise = torch.var(noise).item()
    noise_np = noise.squeeze(0).numpy()
    c = noise.shape[1]
    autocorr = [correlate2d(noise_np[i], noise_np[i], mode="same") for i in range(c)]
    auto_corr_norm = [ac / (ac.max() + 1e-12) for ac in autocorr]
    autocorr_mean = np.mean(auto_corr_norm, axis=0)
    noise_fft = fftshift(fft2(noise_np[0]))
    magnitude_spectrum = np.abs(noise_fft)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(25, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(np.moveaxis(np.abs(noise_np), 0, -1), cmap="magma", vmin=0, vmax=0.1)
    plt.colorbar()
    plt.title("Residual Image")

    plt.subplot(1, 4, 2)
    count, bins, _ = plt.hist(noise_np.flatten(), bins=50, density=True, color="gray", alpha=0.7)
    x = np.linspace(bins[0], bins[-1], 100)
    plt.plot(x, norm.pdf(x, mean_noise, np.sqrt(var_noise)), "r--", lw=2)
    plt.title("Noise Distribution Histogram")

    plt.subplot(1, 4, 3)
    plt.imshow(autocorr_mean, cmap="hot", vmin=0, vmax=0.1)
    plt.colorbar()
    plt.title("Noise Autocorrelation")

    plt.subplot(1, 4, 4)
    plt.imshow(np.log1p(magnitude_spectrum), cmap="inferno")
    plt.colorbar()
    plt.title("Noise Frequency Spectrum")

    plt.savefig(out_path)
    plt.close()

    print(f"Noise Mean: {mean_noise:.6f}")
    print(f"Noise Variance: {var_noise:.6f}")
    print(f"Saved analysis figure to: {out_path}")
