def add_denoise_single_args(parser):
    parser.add_argument("--method", type=str, default="zs", choices=["zs", "bm3d"])
    parser.add_argument("--model", type=str, default="zs_kan", choices=["zs_n2n", "zs_kan", "zs_mkan"])
    parser.add_argument("--img-type", type=str, default="gray", choices=["gray", "color"])
    parser.add_argument("--noise-source", type=str, default="real", choices=["synthetic", "real"])
    parser.add_argument("--noise-type", type=str, default="gauss", choices=["gauss", "poiss"])
    parser.add_argument("--noise-level", type=float, default=25.5)
    parser.add_argument("--clean-img-path", type=str, required=True)
    parser.add_argument("--noisy-img-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/denoise_single")
    parser.add_argument(
        "--crop-size",
        type=int,
        default=0,
        help="Optional center-crop size. If omitted or <=0, no cropping is applied.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--step-size", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--sigma-bm3d", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")


def run_denoise_single(args):
    from zskan_denoising.engine import denoise_single

    result = denoise_single(
        method=args.method,
        model_name=args.model,
        img_type=args.img_type,
        noise_source=args.noise_source,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        clean_img_path=args.clean_img_path,
        noisy_img_path=args.noisy_img_path,
        output_dir=args.output_dir,
        crop_size=args.crop_size,
        epochs=args.epochs,
        lr=args.lr,
        step_size=args.step_size,
        gamma=args.gamma,
        sigma_bm3d=args.sigma_bm3d,
        device=args.device,
    )
    print(f"Saved outputs to: {args.output_dir}")
    print(f"Denoised PSNR: {result['denoised']['psnr']:.4f}")
