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
EXTENDED_SUITE = ROOT / "results/extended_experiments/suite_20261120T170637Z_21e15ceb"
N15_RUN = EXTENDED_SUITE / "N15"
N10_RUN = EXTENDED_SUITE / "N10"
E1_RUN = ROOT / "results/ablations/suite_20261119T072348Z_e0a7c9de/E1/e1_20261119T072350Z_4f3ea7ee"
N15_EPOCH = 98
N15_SHA256 = "066dc8b8de172f5ee1bc08cebee6cdf16bc41fea4b16c189b2b63c38df32f1d2"
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
    if gpu != 0:
        raise ValueError("These evaluations are fixed to GPU 0")
    os.environ.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0",
                       "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "42"})


def start_gpu():
    import torch
    from experiment import seed_everything

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; inference will not fall back to CPU")
    seed_everything(42)
    print("Physical GPU 0:", torch.cuda.get_device_name(0), flush=True)


def read_checkpoint(directory, experiment):
    import torch
    from recipes import BASE_MODEL, fingerprint, validate_e1
    from run import validate_config

    directory = Path(directory).resolve()
    path = directory / "best.pt"
    if not path.is_file():
        raise FileNotFoundError("Existing checkpoint required; no retraining: " + str(path))
    source = read_json(directory / "config.json")
    validate_config(source)
    if source["seed"] != 42 or source.get("ablation", {}).get("experiment") != experiment:
        raise ValueError("Unexpected experiment or seed: " + str(directory))
    if source["model"] != BASE_MODEL or source.get("structure", {}):
        raise ValueError("Expected the unchanged basic CNN architecture")
    if experiment == "E1":
        validate_e1(source)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if experiment == "N15" and digest != N15_SHA256:
        raise ValueError("N15 is locked to epoch 98 and SHA-256 " + N15_SHA256)
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    del payload
    if fingerprint(checkpoint["config"]) != fingerprint(source):
        raise ValueError("Checkpoint and source configuration disagree")
    if experiment == "N15" and checkpoint["epoch"] != N15_EPOCH:
        raise ValueError("Only the locked N15 epoch-98 checkpoint may be evaluated")
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


def runtime_config(source, data_root, workers, purpose):
    from run import validate_config

    config = copy.deepcopy(source)
    config["source_training_protocol"] = config.pop("protocol", None)
    config["evaluation_protocol"] = purpose
    config["gpu"] = 0
    config["data"]["root"] = str(Path(data_root).resolve())
    config["data"]["download"] = False
    if workers is not None:
        config["training"]["num_workers"] = workers
    config["output"].update({"plot": purpose == "locked-n15-test-v1",
                              "evaluate_test": purpose == "locked-n15-test-v1",
                              "compute_cams": purpose == "locked-n15-test-v1"})
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Official single-view test of the locked N15 epoch-98 checkpoint")
    parser.add_argument("--gpus", type=int, choices=(0,), default=0)
    parser.add_argument("--run-dir", type=Path, default=N15_RUN)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    set_gpu_environment(args.gpus)
    checkpoint, source, lock = read_checkpoint(args.run_dir, "N15")
    history_path = args.run_dir / "history.csv"
    with history_path.open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if [int(row["epoch"]) for row in history] != list(range(1, 101)):
        raise ValueError("All 100 training epochs are required for the existing learning-curve visualization")
    best = max(history, key=lambda row: float(row["val_accuracy"]))
    if int(best["epoch"]) != N15_EPOCH or abs(float(best["val_accuracy"]) - checkpoint["val_accuracy"]) > 1e-12:
        raise ValueError("The training history does not identify the locked epoch-98 checkpoint")
    config = runtime_config(source, args.data_dir, args.num_workers, "locked-n15-test-v1")
    config["output"].update({"results_root": str(ROOT / "results/final_test"),
                              "figures_root": str(ROOT / "figures/final_test")})
    from recipes import fingerprint

    plan = {"protocol": "locked-n15-test-v1", "checkpoint": lock, "config_sha256": fingerprint(config),
            "history_sha256": file_sha256(history_path), "split": "official CIFAR-10 test, 10000 samples",
            "inference": "eval mode; hard labels; no augmentation or TTA; frozen existing weights",
            "expected_figures": list(FIGURES)}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    directory = output_directory(ROOT / "results/final_test", "n15_epoch98", args.resume, plan)
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
        print("Locked N15 epoch 98 test accuracy: {:.2%}".format(metrics["accuracy"]), flush=True)
        print("Confusion matrix: rows=true class; columns=predicted class", flush=True)
        for name, row in zip(CLASSES, metrics["confusion_matrix"]):
            print(name + ": " + " ".join(str(count) for count in row), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        record_failure(directory, exc)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print("Error: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
