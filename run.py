import argparse
import json
import math
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parser():
    main = argparse.ArgumentParser(description="CIFAR-10 basic CifarCNN experiment")
    commands = main.add_subparsers(dest="command", required=True)
    for name in ("check", "train", "test"):
        command = commands.add_parser(name)
        if name == "test":
            command.add_argument("--run-dir", type=Path, required=True)
        else:
            command.add_argument("--config", type=Path, default=ROOT / "config" / "basic.json")
        command.add_argument("--gpu", type=int)
        command.add_argument("--batch-size", type=int)
        command.add_argument("--num-workers", type=int)
        command.add_argument("--seed", type=int)
        command.add_argument("--data-dir", type=Path)
        command.add_argument("--no-download", action="store_true")
        if name == "train":
            command.add_argument("--epochs", type=int)
            command.add_argument("--lr", type=float)
            command.add_argument("--weight-decay", type=float)
            command.add_argument("--run-name", default="cifarcnn")
    plotting = commands.add_parser("plot")
    plotting.add_argument("--run-dir", type=Path, required=True)
    plotting.add_argument("--figures-dir", type=Path)
    plotting.add_argument("--split", choices=("val", "test"), default="test")
    download = commands.add_parser("download")
    download.add_argument("--config", type=Path, default=ROOT / "config" / "basic.json")
    download.add_argument("--data-dir", type=Path)
    download.set_defaults(no_download=False)
    return main


def load_config(args):
    source = args.run_dir / "config.json" if args.command == "test" else args.config
    with source.open(encoding="utf-8") as stream:
        config = json.load(stream)
    for name in ("gpu", "seed"):
        if getattr(args, name, None) is not None:
            config[name] = getattr(args, name)
    for name in ("epochs", "batch_size", "lr", "weight_decay", "num_workers"):
        if getattr(args, name, None) is not None:
            config["training"][name] = getattr(args, name)
    if args.data_dir is not None:
        config["data"]["root"] = str(args.data_dir.resolve())
    if args.no_download:
        config["data"]["download"] = False
    for section, key in (("data", "root"), ("output", "results_root")):
        path = Path(config[section][key])
        config[section][key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    validate_config(config)
    return config


def validate_config(config):
    def integer(value, name, minimum=1):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(name + " must be an integer >= " + str(minimum))

    def number(value, name, minimum=0):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
            raise ValueError(name + " must be finite and >= " + str(minimum))

    integer(config["seed"], "seed", 0)
    if config["seed"] >= 2 ** 32:
        raise ValueError("seed must be smaller than 2**32")
    integer(config["gpu"], "gpu", 0)
    options = config["training"]
    integer(options["epochs"], "epochs")
    integer(options["batch_size"], "batch_size")
    integer(options["num_workers"], "num_workers", 0)
    number(options["lr"], "lr", 1e-12)
    number(options["weight_decay"], "weight_decay")
    if options["optimizer"] not in ("Adam", "AdamW", "SGD"):
        raise ValueError("Expected Adam, AdamW or SGD")
    model = config["model"]
    if len(model["channels"]) != 4 or len(model["depths"]) != 4:
        raise ValueError("Four stages are required")
    for key in ("channels", "depths"):
        for value in model[key]:
            integer(value, key)
    integer(model["head_channels"], "head_channels")
    if model["num_classes"] != 10 or not 0 <= model["dropout"] < 1:
        raise ValueError("Expected ten classes and dropout in [0,1)")
    data = config["data"]
    if data["split_seed"] != 42 or data["val_size"] != 5000:
        raise ValueError("The basic experiment fixes a stratified 45,000/5,000 split with seed 42")
    if data["mean"] != [0.5] * 3 or data["std"] != [0.5] * 3:
        raise ValueError("Expected mean/std (0.5,0.5,0.5)")
    integer(data["crop_padding"], "crop_padding", 0)
    if not 0 <= data["horizontal_flip_probability"] <= 1:
        raise ValueError("horizontal_flip_probability must be in [0,1]")
    if not isinstance(data["download"], bool):
        raise ValueError("download must be a boolean")
    urls = data.get("download_urls", [])
    if "download_urls" in data and (not isinstance(urls, list) or not urls or any(
            not isinstance(url, str) or not url.startswith("https://") for url in urls)):
        raise ValueError("download_urls must be a non-empty list of HTTPS URLs")
    safety = config["safety"]
    for key, minimum in (("activation_memory_multiplier", 6), ("workspace_gib", 1),
                         ("context_gib", 0.5), ("reserve_fraction", 0.2),
                         ("min_reserve_gib", 1)):
        number(safety[key], key, minimum)
    if safety["reserve_fraction"] >= 1:
        raise ValueError("reserve_fraction must be smaller than one")
    for key, default in (("error_examples_per_group", 4), ("cam_classes", 2),
                         ("calibration_bins", 10), ("top_confusions", 5)):
        integer(config["output"].get(key, default), key)
    if config["output"].get("cam_classes", 2) > 10:
        raise ValueError("cam_classes must not exceed ten")
    integer(config["output"]["figure_dpi"], "figure_dpi")
    if "protocol" in config:
        from recipes import validate_recipe
        validate_recipe(config)


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        if args.command == "plot":
            from visualization import plot_results
            plot_results(args.run_dir, figures_root=args.figures_dir, split=args.split)
            return 0
        config = load_config(args)
        if args.command == "download":
            from data import prepare_data
            prepare_data(config["data"])
            return 0
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from experiment import gpu_preflight, new_directory, test_only, train, write_json
        if args.command == "check":
            directory = new_directory(config["output"]["results_root"], "gpu_check")
            write_json(directory / "config.json", config)
            print("GPU check records:", directory, flush=True)
            return 0 if gpu_preflight(config, directory) else 2
        if args.command == "test":
            return test_only(args.run_dir, config)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_name):
            argument_parser.error("run-name must contain 1-64 letters, digits, underscores or hyphens")
        return train(config, args.run_name)
    except Exception as exc:
        print("Error: " + str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
