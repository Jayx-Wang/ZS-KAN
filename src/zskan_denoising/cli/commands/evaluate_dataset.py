def add_evaluate_dataset_args(parser):
    parser.add_argument("--method", type=str, default="zs", choices=["zs", "bm3d"])
    parser.add_argument("--model", type=str, default="zs_kan", choices=["zs_n2n", "zs_kan", "zs_mkan"])
    parser.add_argument("--img-type", type=str, default="color", choices=["gray", "color"])
    parser.add_argument("--noise-source", type=str, default="synthetic", choices=["synthetic", "real"])
    parser.add_argument("--noise-type", type=str, default="poiss", choices=["gauss", "poiss"])
    parser.add_argument("--noise-level", type=float, default=80.0)
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/evaluate_dataset")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--step-size", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--sigma-bm3d", type=float, default=0.157)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-images", type=int, default=None)


def run_evaluate_dataset(args):
    from zskan_denoising.engine import evaluate_dataset

    report = evaluate_dataset(
        method=args.method,
        model_name=args.model,
        img_type=args.img_type,
        noise_source=args.noise_source,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        data_folder=args.data_folder,
        output_dir=args.output_dir,
        crop_size=args.crop_size,
        epochs=args.epochs,
        lr=args.lr,
        step_size=args.step_size,
        gamma=args.gamma,
        sigma_bm3d=args.sigma_bm3d,
        device=args.device,
        max_images=args.max_images,
    )
    print(f"Saved report to: {args.output_dir}/dataset_metrics.json")
    print(f"Average PSNR: {report['summary']['avg_psnr']:.4f}")
