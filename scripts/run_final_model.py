import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from datetime import datetime, timezone
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FINAL_CONFIG = ROOT / "config/final.json"
ARCHIVED_EXTENDED_SUITE = ROOT / "results/extended_experiments/suite_20261120T170637Z_21e15ceb"
COMPLETE_MODEL_RUN = ARCHIVED_EXTENDED_SUITE / "N15"
RANDOM_ERASING_050_RUN = ARCHIVED_EXTENDED_SUITE / "N10"
WITHOUT_RANDAUGMENT_RUN = ROOT / "results/ablations/suite_20261119T072348Z_e0a7c9de/E1/e1_20261119T072350Z_4f3ea7ee"
ORIGINAL_TEST_RUN = ROOT / "results/final_test/n15_epoch98_20261122T094817Z_49974936"
OFFICIAL_TEST_FIGURES = ROOT / "figures/complete_model/official_test"
ORIGINAL_EPOCH = 98
ORIGINAL_SHA256 = "066dc8b8de172f5ee1bc08cebee6cdf16bc41fea4b16c189b2b63c38df32f1d2"
ORIGINAL_TEST_PROTOCOL = "locked-complete-model-test-v1"
FIGURES = ("learning_curves.png", "confusion_matrix.png", "class_performance.png",
           "confidence_diagnostics.png", "error_cases.png", "cam_comparison.png")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_gpu_environment(gpu):
    if gpu < 0:
        raise ValueError("GPU index must be nonnegative")
    os.environ.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu),
                       "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "42"})


def start_gpu():
    import torch
    from experiment import seed_everything

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; inference will not fall back to CPU")
    seed_everything(42)
    print("Physical GPU " + os.environ["CUDA_VISIBLE_DEVICES"] + ":",
          torch.cuda.get_device_name(0), flush=True)


def read_checkpoint(directory, experiment, *, lock_original=False):
    import torch
    from recipes import BASE_MODEL, experiment_name, fingerprint, validate_reference
    from run import validate_config

    directory = Path(directory).resolve()
    path = directory / "best.pt"
    if not path.is_file():
        raise FileNotFoundError("Existing checkpoint required; no retraining: " + str(path))
    source = read_json(directory / "config.json")
    validate_config(source)
    if source["seed"] != 42 or experiment_name(source) != experiment:
        raise ValueError("Unexpected experiment or seed: " + str(directory))
    if source["model"] != BASE_MODEL or source.get("structure", {}):
        raise ValueError("Expected the report's four-stage CifarCNN architecture")
    if experiment == "without_randaugment":
        validate_reference(source)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if lock_original and digest != ORIGINAL_SHA256:
        raise ValueError("The original complete model is locked to SHA-256 " + ORIGINAL_SHA256)
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    del payload
    if fingerprint(checkpoint["config"]) != fingerprint(source):
        raise ValueError("Checkpoint and source configuration disagree")
    if lock_original and checkpoint["epoch"] != ORIGINAL_EPOCH:
        raise ValueError("The original complete model uses the epoch-98 checkpoint")
    validation = read_json(directory / "val_metrics.json")
    if (validation["best_checkpoint_sha256"] != digest
            or validation["best_epoch"] != checkpoint["epoch"]
            or abs(validation["best_val_accuracy"] - checkpoint["val_accuracy"]) > 1e-12
            or abs(validation["accuracy"] - checkpoint["val_accuracy"]) > 1e-12):
        raise ValueError("Checkpoint does not match its saved validation provenance")
    lock = {"experiment": experiment, "checkpoint": str(path), "sha256": digest,
            "epoch": checkpoint["epoch"], "best_val_accuracy": checkpoint["val_accuracy"],
            "source_config_sha256": fingerprint(source)}
    return checkpoint, source, lock


def runtime_config(source, data_root, workers, purpose, gpu=0):
    from recipes import experiment_name
    from run import validate_config

    config = copy.deepcopy(source)
    config["experiment"] = experiment_name(source)
    config.pop("ablation", None)
    config["source_training_protocol"] = config.pop("protocol", None)
    config["evaluation_protocol"] = purpose
    config["gpu"] = gpu
    config["data"]["root"] = str(Path(data_root).resolve())
    config["data"]["download"] = False
    if workers is not None:
        config["training"]["num_workers"] = workers
    official_test = purpose == ORIGINAL_TEST_PROTOCOL
    config["output"].update({"plot": official_test, "evaluate_test": official_test,
                              "compute_cams": official_test})
    validate_config(config)
    return config


def load_frozen_model(checkpoint):
    from models.ablations import build_model

    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.requires_grad_(False).to("cuda:0").eval()


def save_environment(directory):
    from experiment import environment_info

    current = environment_info()

    def stable(value):
        device = value.get("selected_cuda_device", {})
        return {"packages": {key: value["packages"].get(key) for key in ("torch", "torchvision", "numpy")},
                "cuda": value["torch_cuda_build"], "cudnn": value["cudnn_version"],
                "device": {key: device.get(key) for key in ("name", "compute_capability")}}

    path = directory / "environment.json"
    if path.exists():
        if stable(read_json(path)) != stable(current):
            raise ValueError("Inference environment changed; use a new output directory")
    else:
        write_json(path, current)


def output_directory(root, prefix, resume, plan):
    from recipes import fingerprint

    if resume is not None:
        directory = Path(resume).resolve()
        if fingerprint(read_json(directory / "evaluation_plan.json")) != fingerprint(plan):
            raise ValueError("Resume plan differs; use a new output directory")
        return directory
    name = prefix + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "evaluation_plan.json", plan)
    return directory


def record_failure(directory, exc):
    write_json(directory / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                           "error": str(exc), "traceback": traceback.format_exc(),
                                           "automatic_retry": False, "retraining": False})


def test_original(args):
    gpu = 0 if args.gpu is None else args.gpu
    set_gpu_environment(gpu)
    source_run = args.run_dir or COMPLETE_MODEL_RUN
    checkpoint, source, lock = read_checkpoint(source_run, "complete_model", lock_original=True)
    history_path = source_run / "history.csv"
    with history_path.open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if [int(row["epoch"]) for row in history] != list(range(1, 101)):
        raise ValueError("All 100 training epochs are required for the existing learning-curve visualization")
    best = max(history, key=lambda row: float(row["val_accuracy"]))
    if int(best["epoch"]) != ORIGINAL_EPOCH or abs(float(best["val_accuracy"]) - checkpoint["val_accuracy"]) > 1e-12:
        raise ValueError("The training history does not identify the locked epoch-98 checkpoint")
    config = runtime_config(source, args.data_dir or ROOT / "dataset", args.num_workers,
                            ORIGINAL_TEST_PROTOCOL, gpu=gpu)
    results_root = ROOT / "results/complete_model/official_test"
    config["output"].update({"results_root": str(results_root), "figures_root": str(OFFICIAL_TEST_FIGURES)})
    from recipes import fingerprint

    plan = {"protocol": ORIGINAL_TEST_PROTOCOL, "checkpoint": lock, "config_sha256": fingerprint(config),
            "history_sha256": file_sha256(history_path), "split": "official CIFAR-10 test, 10000 samples",
            "inference": "eval mode; hard labels; no augmentation or TTA; frozen existing weights",
            "expected_figures": list(FIGURES)}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    directory = output_directory(results_root, "complete_model_epoch98", args.resume, plan)
    print("Results:", directory, flush=True)
    try:
        write_json(directory / "config.json", config)
        write_json(directory / "source_config.json", source)
        write_json(directory / "checkpoint_lock.json", lock)
        if not (directory / "history.csv").exists():
            shutil.copyfile(history_path, directory / "history.csv")
        if file_sha256(directory / "history.csv") != plan["history_sha256"]:
            raise ValueError("The copied history changed")
        start_gpu()
        from data import CLASSES, test_data
        from experiment import evaluate_split
        from visualization import plot_results

        save_environment(directory)
        write_json(directory / "status.json", {"status": "evaluating_locked_test"})
        model = load_frozen_model(checkpoint)
        _, loader = test_data(config)
        metrics = evaluate_split(model, loader, directory, "test", checkpoint, lock["sha256"], config)
        with (directory / "test_class_accuracy.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("class", "samples", "accuracy"))
            writer.writerows((name, metrics["per_class_samples"][name], metrics["per_class_accuracy"][name]) for name in CLASSES)
        with (directory / "test_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("true_class/predicted_class", *CLASSES))
            writer.writerows((name, *row) for name, row in zip(CLASSES, metrics["confusion_matrix"]))
        write_json(directory / "status.json", {"status": "plotting"})
        figures = plot_results(directory, split="test")
        if not all((figures / name).is_file() and (figures / name).stat().st_size for name in FIGURES):
            raise RuntimeError("The existing visualization set is incomplete")
        if file_sha256(lock["checkpoint"]) != lock["sha256"]:
            raise RuntimeError("Source checkpoint changed during evaluation")
        write_json(directory / "figures.json", {"directory": str(figures), "files": list(FIGURES)})
        write_json(directory / "status.json", {"status": "completed", "checkpoint": lock,
                                               "test_accuracy": metrics["accuracy"], "figures": str(figures),
                                               "training_performed": False})
        print("Original complete model epoch 98 test accuracy: {:.2%}".format(metrics["accuracy"]), flush=True)
        print("Confusion matrix: rows=true class; columns=predicted class", flush=True)
        for name, row in zip(CLASSES, metrics["confusion_matrix"]):
            print(name + ": " + " ".join(str(count) for count in row), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        record_failure(directory, exc)
        raise


def test(args):
    from recipes import experiment_name
    from run import main as run_command

    if args.original:
        if args.batch_size is not None:
            raise ValueError("The original test uses the batch size saved with the report checkpoint")
        return test_original(args)
    if args.dry_run or args.resume is not None:
        raise ValueError("--dry-run and --resume are available with --original")
    if args.run_dir is None:
        raise ValueError("Specify --run-dir for a new training run or --original for the report checkpoint")
    config = read_json(args.run_dir / "config.json")
    if experiment_name(config) != "complete_model" or "evaluation_protocol" in config:
        raise ValueError("--run-dir must contain a complete-model training run")
    if "ablation" in config:
        raise ValueError("Use --original to evaluate the archived report checkpoint in a fresh directory")
    command = ["test", "--run-dir", str(args.run_dir)]
    for name, value in (("gpu", args.gpu), ("data-dir", args.data_dir), ("num-workers", args.num_workers),
                        ("batch-size", args.batch_size)):
        if value is not None:
            command.extend(("--" + name, str(value)))
    if args.no_download:
        command.append("--no-download")
    return run_command(command)


def test_arguments(argv):
    parser = argparse.ArgumentParser(
        prog="run_final_model.py test",
        description="Evaluate a new complete-model run or reproduce the report's official test")
    parser.add_argument("--original", action="store_true", help="use the report's locked checkpoint")
    parser.add_argument("--run-dir", type=Path, help="training run directory; optional with --original")
    parser.add_argument("--gpu", type=int, help="physical GPU index; defaults to the saved run's GPU or 0 with --original")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--batch-size", type=int, help="override the batch size for a new run")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--resume", type=Path, help="resume an evaluation started with --original")
    parser.add_argument("--dry-run", action="store_true", help="verify the --original checkpoint without GPU work")
    return parser.parse_args(argv)


def plot(argv):
    from recipes import experiment_name
    from run import main as run_command

    parser = argparse.ArgumentParser(prog="run_final_model.py plot",
                                     description="Regenerate figures from saved complete-model results")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--original", action="store_true", help="use the report's saved official test results")
    source.add_argument("--run-dir", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args(argv)
    if args.original and args.split != "test":
        parser.error("The archived official test directory contains test results")
    directory = ORIGINAL_TEST_RUN if args.original else args.run_dir
    if experiment_name(read_json(directory / "config.json")) != "complete_model":
        parser.error("--run-dir must contain complete-model results")
    figures_dir = args.figures_dir or (OFFICIAL_TEST_FIGURES if args.original else None)
    command = ["plot", "--run-dir", str(directory), "--split", args.split]
    if figures_dir is not None:
        command.extend(("--figures-dir", str(figures_dir)))
    return run_command(command)


def main(argv=None):
    from run import main as run_command

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Train and evaluate the complete CIFAR-10 model from the report")
    parser.add_argument("command", nargs="?", default="train",
                        choices=("train", "check", "download", "test", "plot"))
    parser.epilog = "Choose a command followed by --help for its options. The default command is train."
    if arguments and arguments[0] in ("-h", "--help"):
        parser.parse_args(arguments[:1])
    if arguments and not arguments[0].startswith("-"):
        command = parser.parse_args(arguments[:1]).command
        options = arguments[1:]
    else:
        command, options = "train", arguments
    if command == "test":
        return test(test_arguments(options))
    if command == "plot":
        return plot(options)
    return run_command([command, *options], configuration=FINAL_CONFIG)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print("Error: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
