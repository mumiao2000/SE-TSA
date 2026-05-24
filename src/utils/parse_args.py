import argparse
import json
import os

import yaml


def parse_args() -> argparse.Namespace:
    # Get default args from config.yaml
    args = argparse.Namespace()
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config_path = os.path.abspath(config_path)
    print(f"Loading config from {config_path}")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if not hasattr(args, key):
                setattr(args, key, value)
    # Override with command line args
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--dataset", type=str, default="TEST", help="Dataset name")
    parser.add_argument(
        "--io-pairs",
        type=str,
        required=True,
        help="Input/output length pairs, e.g. '8,6;8,8;8,10'",
    )
    parser.add_argument(
        "--split-ratio",
        type=str,
        default="0.7,0.1,0.2",
        help="Train/val/test split ratios",
    )
    parser.add_argument(
        "--max-time", type=int, default=720, help="Total time budget in minutes"
    )
    cmd_args = parser.parse_args()
    for key, value in vars(cmd_args).items():
        if value is not None:
            setattr(args, key, value)
    # Post-process some args
    if isinstance(args.io_pairs, str):
        pairs = []
        for pair in args.io_pairs.split(";"):
            il, ol = pair.split(",")
            pairs.append((int(il), int(ol)))
        args.io_pairs = pairs
    args.input_len = args.io_pairs[0][0]
    args.output_len = args.io_pairs[0][1]
    if isinstance(args.split_ratio, str):
        args.split_ratio = [float(x) for x in args.split_ratio.split(",")]
    if isinstance(args.arch_families, str):
        args.arch_families = [x for x in args.arch_families.split(",")]
    if isinstance(args.preprocessing_types, str):
        args.preprocessing_types = [x for x in args.preprocessing_types.split(",")]
    args.output_dir = os.path.join(f"outputs/{args.dataset}_multi")
    # Print final args
    print(
        f"[Args] Final arguments:\n{json.dumps(vars(args), indent=4, ensure_ascii=False)}"
    )
    return args
