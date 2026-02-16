import argparse

from .commands import (
    add_analyze_noise_args,
    add_denoise_single_args,
    add_evaluate_dataset_args,
    run_analyze_noise,
    run_denoise_single,
    run_evaluate_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    root_epilog = """Method-specific help:
  zskan denoise-single --help
  zskan evaluate-dataset --help
  zskan analyze-noise --help
"""
    parser = argparse.ArgumentParser(
        prog="zskan",
        description="ZS-KAN image denoising toolkit",
        epilog=root_epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single_epilog = """Examples:
  # ZS-KAN (synthetic, color)
  zskan denoise-single --method zs --model zs_kan --img-type color --noise-source synthetic --noise-type poiss --noise-level 80 --clean-img-path data/kodak24/clean/kodim01.png --device cuda

  # ZS-N2N (synthetic, gray)
  zskan denoise-single --method zs --model zs_n2n --img-type gray --noise-source synthetic --noise-type gauss --noise-level 25 --clean-img-path data/microscopy/clean/TwoPhoton_BPAE_B_4.png --device cuda

  # ZS-MKAN (real microscopy)
  zskan denoise-single --method zs --model zs_mkan --img-type gray --noise-source real --clean-img-path data/microscopy/clean/TwoPhoton_BPAE_B_4.png --noisy-img-path data/microscopy/noisy/TwoPhoton_BPAE_B_4.png --device cuda

  # BM3D (synthetic)
  zskan denoise-single --method bm3d --img-type color --noise-source synthetic --noise-type gauss --noise-level 25 --sigma-bm3d 0.1 --clean-img-path data/kodak24/clean/kodim01.png --device cpu

  # BM3D (real microscopy)
  zskan denoise-single --method bm3d --img-type gray --noise-source real --clean-img-path data/microscopy/clean/TwoPhoton_BPAE_B_4.png --noisy-img-path data/microscopy/noisy/TwoPhoton_BPAE_B_4.png --device cpu
"""
    single = subparsers.add_parser(
        "denoise-single",
        help="Denoise one image",
        description="Run single-image denoising with either ZS models or BM3D.",
        epilog=single_epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_denoise_single_args(single)
    single.set_defaults(func=run_denoise_single)

    dataset_epilog = """Examples:
  # ZS-KAN dataset evaluation (synthetic Kodak24)
  zskan evaluate-dataset --method zs --model zs_kan --img-type color --noise-source synthetic --noise-type poiss --noise-level 80 --data-folder data/kodak24/clean --device cuda

  # ZS-N2N dataset evaluation (synthetic Kodak24)
  zskan evaluate-dataset --method zs --model zs_n2n --img-type color --noise-source synthetic --noise-type gauss --noise-level 25 --data-folder data/kodak24/clean --device cuda

  # ZS-MKAN dataset evaluation (real microscopy)
  zskan evaluate-dataset --method zs --model zs_mkan --img-type gray --noise-source real --data-folder data/microscopy --device cuda

  # BM3D dataset evaluation (synthetic Kodak24)
  zskan evaluate-dataset --method bm3d --img-type color --noise-source synthetic --noise-type gauss --noise-level 25 --sigma-bm3d 0.1 --data-folder data/kodak24/clean --device cpu

  # BM3D dataset evaluation (real microscopy)
  zskan evaluate-dataset --method bm3d --img-type gray --noise-source real --data-folder data/microscopy --device cpu
"""
    dataset = subparsers.add_parser(
        "evaluate-dataset",
        help="Evaluate on dataset",
        description="Run dataset-level evaluation for ZS models or BM3D.",
        epilog=dataset_epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_evaluate_dataset_args(dataset)
    dataset.set_defaults(func=run_evaluate_dataset)

    noise = subparsers.add_parser(
        "analyze-noise",
        help="Analyze residual noise",
        description="Analyze residual noise statistics and visualization outputs.",
        epilog=(
            "Example:\n"
            "  zskan analyze-noise --img-type gray --clean-img-path "
            "data/microscopy/clean/TwoPhoton_BPAE_B_4.png --noisy-img-path "
            "data/microscopy/noisy/TwoPhoton_BPAE_B_4.png --output-path "
            "outputs/noise_analysis/TwoPhoton_BPAE_B_4.png\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_analyze_noise_args(noise)
    noise.set_defaults(func=run_analyze_noise)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
