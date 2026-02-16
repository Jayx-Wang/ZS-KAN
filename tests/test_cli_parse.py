from zskan_denoising.cli.main import build_parser


def test_cli_has_expected_commands():
    parser = build_parser()
    args = parser.parse_args(["denoise-single", "--clean-img-path", "a.png"])
    assert args.command == "denoise-single"
    assert args.method in {"zs", "bm3d"}


def test_cli_evaluate_defaults():
    parser = build_parser()
    args = parser.parse_args(["evaluate-dataset", "--data-folder", "data/kodak24/clean"])
    assert args.command == "evaluate-dataset"
    assert args.model in {"zs_n2n", "zs_kan", "zs_mkan"}
