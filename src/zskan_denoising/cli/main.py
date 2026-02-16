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
    parser = argparse.ArgumentParser(prog="zskan", description="ZS-KAN image denoising toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("denoise-single", help="Denoise one image")
    add_denoise_single_args(single)
    single.set_defaults(func=run_denoise_single)

    dataset = subparsers.add_parser("evaluate-dataset", help="Evaluate on dataset")
    add_evaluate_dataset_args(dataset)
    dataset.set_defaults(func=run_evaluate_dataset)

    noise = subparsers.add_parser("analyze-noise", help="Analyze residual noise")
    add_analyze_noise_args(noise)
    noise.set_defaults(func=run_analyze_noise)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
