import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
import uuid

from run_final_model import (
    COMPLETE_MODEL_RUN, FIGURES, RANDOM_ERASING_050_RUN, ROOT, WITHOUT_RANDAUGMENT_RUN,
    file_sha256, load_frozen_model, output_directory, read_checkpoint, read_json,
    record_failure, runtime_config, save_environment, set_gpu_environment,
    start_gpu, write_json,
)
from recipes import ADDITIONAL_EXPERIMENTS, ADDITIONAL_PROTOCOL, fingerprint, make_additional_config


RESULTS_ROOT = ROOT / "results" / "additional_experiments"
FIGURES_ROOT = ROOT / "figures" / "additional_experiments"
DATA_ROOT = ROOT / "dataset"
ROBUSTNESS_MODELS = ("without_randaugment", "complete_model", "random_erasing_050")
ROBUSTNESS_PROTOCOL = "validation-robustness-v1"


def arguments(argv):
    parser = argparse.ArgumentParser(description="Train CIFAR-10 controls and compare robustness.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train selected controls, or all 32")
    train.add_argument("--experiments", nargs="+", metavar="NAME",
                       help="choose from " + ", ".join(ADDITIONAL_EXPERIMENTS))
    train.add_argument("--no-download", action="store_true")
    robust = commands.add_parser("robustness", help="compare frozen models on paired validation perturbations")
    robust.add_argument("--without-randaugment-run", type=Path, default=WITHOUT_RANDAUGMENT_RUN)
    robust.add_argument("--complete-model-run", type=Path, default=COMPLETE_MODEL_RUN)
    robust.add_argument("--random-erasing-run", type=Path, default=RANDOM_ERASING_050_RUN)
    robust.add_argument("--positions-file", type=Path)
    for command in (train, robust):
        command.add_argument("--gpu", type=int, help="physical GPU index (default: 0)")
        command.add_argument("--data-dir", type=Path, help="CIFAR-10 directory (default: dataset)")
        command.add_argument("--num-workers", type=int)
        command.add_argument("--resume", type=Path, help="directory of an existing run from this script")
        command.add_argument("--dry-run", action="store_true", help="print the plan without writing files")
    train.set_defaults(action=train_suite)
    robust.set_defaults(action=compare_robustness)
    return parser.parse_args(argv)


def run_worker(phase, directory):
    try:
        from experiment import gpu_preflight, train_recipe
        from run import validate_config

        config = read_json(directory / "config.json")
        validate_config(config)
        if phase == "train":
            return train_recipe(config, directory)
        write_json(directory / "status.json", {"status": "checking_gpu"})
        passed = gpu_preflight(config, directory)
        report = read_json(directory / "gpu_check.json")
        write_json(directory / "status.json", {
            "status": "ready" if passed else "preflight_failed",
            "reason": None if passed else "; ".join(report["blockers"]),
            "config_sha256": fingerprint(config),
        })
        return 0 if passed else 2
    except Exception as exc:
        write_json(directory / "status.json", {"status": "failed", "phase": phase,
                                               "reason": str(exc), "traceback": traceback.format_exc()})
        print("Error:", exc, file=sys.stderr, flush=True)
        return 1


def launch_worker(phase, directory, gpu):
    environment = os.environ.copy()
    environment.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu),
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "42",
                        "PYTHONIOENCODING": "utf-8"})
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "_worker", phase, str(directory)]
    try:
        with (directory / "run.log").open("a", encoding="utf-8") as log:
            log.write("\n" + datetime.now(timezone.utc).isoformat() + " " + phase + "\n")
            log.flush()
            with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace", bufsize=1) as process:
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    return process.wait()
                except KeyboardInterrupt:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    write_json(directory / "status.json", {
                        "status": "interrupted", "reason": "Interrupted during " + phase,
                        "resume": "from last completed epoch"})
                    raise
    except OSError as exc:
        write_json(directory / "status.json", {"status": "failed", "phase": phase,
                                               "reason": str(exc), "traceback": traceback.format_exc()})
        return 1


def completed_run(directory, config):
    required = ("status.json", "best.pt", "last.pt", "history.csv", "resources.json",
                "reproducibility.json", "environment.json", "model.json", "split_indices.npz")
    if not all((directory / name).is_file() for name in required):
        return False
    try:
        status = read_json(directory / "status.json")
        epochs = config["training"]["epochs"]
        if (status.get("status") != "completed" or status.get("epochs_completed") != epochs
                or status.get("config_sha256") != fingerprint(config)):
            return False
        resources = read_json(directory / "resources.json")
        if not resources.get("training_completed") or resources.get("completed_epochs") != epochs:
            return False
        with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        if [int(row["epoch"]) for row in history] != list(range(1, epochs + 1)):
            return False
        checkpoint_hash = file_sha256(directory / "best.pt")
        splits = ["train_eval", "val"]
        if config["output"]["evaluate_test"]:
            splits.append("test")
        for split in splits:
            metrics = read_json(directory / f"{split}_metrics.json")
            if metrics.get("best_checkpoint_sha256") != checkpoint_hash:
                return False
            artifacts = metrics.get("artifact_sha256", {})
            if split != "train_eval":
                expected = {f"{split}_outputs.npz", f"{split}_predictions.csv"}
                if config["output"]["compute_cams"]:
                    expected.add(f"{split}_examples.npz")
                if not expected.issubset(artifacts):
                    return False
            if any(file_sha256(directory / name) != digest for name, digest in artifacts.items()):
                return False
        if config["output"]["plot"]:
            figure_root = Path(config["output"]["figures_root"]) / directory.name
            if not any(all((plot / file).is_file() for file in FIGURES) for plot in figure_root.glob("plot_test_*")):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def summarize_suite(suite, manifest):
    metric_fields = ("accuracy", "loss", "macro_f1", "ece", "multiclass_brier", "error_count",
                     "wrong_mean_confidence", "cat_to_dog", "dog_to_cat",
                     "wrong_with_confidence_at_least_0.9", "wrong_with_confidence_at_least_0.99")
    resource_fields = ("total_training_seconds", "total_training_wall_seconds",
                       "peak_allocated_bytes", "peak_reserved_bytes")
    rows = []
    for name in manifest["experiments"]:
        directory = suite / name
        row = {"experiment": name, "run_dir": str(directory), "status": "pending"}
        if (directory / "status.json").is_file():
            row.update(read_json(directory / "status.json"))
        for split in ("train_eval", "val", "test"):
            path = directory / f"{split}_metrics.json"
            if path.is_file():
                metrics = read_json(path)
                row[split] = metrics
                row["best_epoch"] = metrics["best_epoch"]
                row.update({f"{split}_{key}": metrics.get(key) for key in metric_fields})
                for label in ("cat", "dog"):
                    row[f"{split}_{label}_accuracy"] = metrics["per_class_accuracy"][label]
        if (directory / "model.json").is_file():
            row["model"] = read_json(directory / "model.json")
            row["parameters"] = row["model"]["parameters"]
        if (directory / "resources.json").is_file():
            row["resources"] = read_json(directory / "resources.json")
            row.update({key: row["resources"].get(key) for key in resource_fields})
        rows.append(row)
    write_json(suite / "summary.json", {"seed": 42, "physical_gpu": manifest["gpu"],
                                         "protocol": ADDITIONAL_PROTOCOL, "runs": rows})
    fields = ["experiment", "status", "reason", "error", "best_epoch"]
    fields.extend(f"{split}_{key}" for split in ("train_eval", "val", "test")
                  for key in (*metric_fields, "cat_accuracy", "dog_accuracy"))
    fields.extend(("parameters", *resource_fields, "run_dir"))
    temporary = suite / "metrics.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(suite / "metrics.csv")


def train_suite(args):
    from run import validate_config

    if args.resume:
        suite = args.resume.resolve()
        manifest = read_json(suite / "suite.json")
        if manifest.get("protocol") != ADDITIONAL_PROTOCOL:
            raise ValueError("Resume requires a suite created by this additional experiments script")
        if (args.gpu is not None and args.gpu != manifest["gpu"]
                or args.data_dir is not None and args.data_dir.resolve() != Path(manifest["data_root"])
                or args.num_workers is not None and args.num_workers != manifest["num_workers"]
                or args.no_download and not manifest["no_download"]):
            raise ValueError("Resume requires the original GPU, data path, worker count and download setting")
    else:
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        suite = RESULTS_ROOT / f"suite_{suffix}"
        manifest = {"protocol": ADDITIONAL_PROTOCOL, "experiments": args.experiments or list(ADDITIONAL_EXPERIMENTS),
                    "data_root": str((args.data_dir or DATA_ROOT).resolve()), "gpu": 0 if args.gpu is None else args.gpu,
                    "num_workers": 4 if args.num_workers is None else args.num_workers,
                    "no_download": args.no_download}

    selected = args.experiments or manifest["experiments"]
    unknown = set(selected) - set(ADDITIONAL_EXPERIMENTS)
    if unknown:
        raise ValueError("Unknown additional experiment: " + ", ".join(sorted(unknown)))
    if len(set(selected)) != len(selected):
        raise ValueError("Each experiment may only be selected once")
    if not set(selected).issubset(manifest["experiments"]):
        raise ValueError("Cannot add new experiments to an existing suite")

    configs = {}
    for name in selected:
        directory = suite / name
        config = make_additional_config(
            name, directory, FIGURES_ROOT / suite.name, data_root=manifest["data_root"],
            workers=manifest["num_workers"], no_download=manifest["no_download"], gpu=manifest["gpu"])
        validate_config(config)
        stored = directory / "config.json"
        if stored.exists() and fingerprint(read_json(stored)) != fingerprint(config):
            raise ValueError(f"Configuration changed for {name}. Start a new suite.")
        configs[name] = config
    if args.dry_run:
        print(json.dumps({"suite": str(suite), "experiments": configs}, indent=2))
        return 0

    if not args.resume:
        suite.mkdir(parents=True, exist_ok=False)
        write_json(suite / "suite.json", manifest)
    for name, config in configs.items():
        directory = suite / name
        directory.mkdir(exist_ok=True)
        stored = directory / "config.json"
        if not stored.exists():
            write_json(stored, config)

    print("Suite results:", suite, flush=True)
    ready = []
    failed = False
    try:
        for name, config in configs.items():
            directory = suite / name
            if completed_run(directory, config):
                print(f"{name}: reusing completed experiment", flush=True)
                continue
            print(f"{name}: GPU preflight", flush=True)
            if launch_worker("check", directory, manifest["gpu"]) == 0:
                ready.append(name)
            else:
                failed = True
            summarize_suite(suite, manifest)
        for name in ready:
            directory = suite / name
            print(f"{name}: training", flush=True)
            code = launch_worker("train", directory, manifest["gpu"])
            if code or not completed_run(directory, configs[name]):
                failed = True
                if code == 0:
                    write_json(directory / "status.json", {
                        "status": "failed", "reason": "Training exited without consistent saved artifacts"})
            summarize_suite(suite, manifest)
    finally:
        summarize_suite(suite, manifest)
    return 1 if failed else 0


CONDITIONS = {
    "clean": {"kind": "clean"},
    "translate_left_2": {"kind": "translation", "dx": -2, "fill_normalized": 0},
    "translate_right_2": {"kind": "translation", "dx": 2, "fill_normalized": 0},
    "brightness_0.7": {"kind": "brightness", "factor": 0.7, "clip_raw": [0, 1]},
    "brightness_1.3": {"kind": "brightness", "factor": 1.3, "clip_raw": [0, 1]},
    "occlusion_4": {"kind": "occlusion", "size": 4, "fill_normalized": 0},
    "occlusion_8": {"kind": "occlusion", "size": 8, "fill_normalized": 0},
}


def perturb_image(raw, condition, mean, std, position=None):
    import torch

    specification = CONDITIONS[condition]
    if specification["kind"] == "brightness":
        raw = (raw * specification["factor"]).clamp(0, 1)
    image = (raw - mean) / std
    if specification["kind"] == "translation":
        shifted = torch.zeros_like(image)
        dx = specification["dx"]
        if dx < 0:
            shifted[..., :dx] = image[..., -dx:]
        else:
            shifted[..., dx:] = image[..., :-dx]
        return shifted
    if specification["kind"] == "occlusion":
        top, left = (int(value) for value in position)
        size = specification["size"]
        image = image.clone()
        image[:, top:top + size, left:left + size] = 0
    return image


class ValidationView:
    def __init__(self, dataset, indices, condition, positions, mean, std):
        import torch

        self.dataset, self.indices, self.condition = dataset, indices, condition
        self.positions = positions
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, target = self.dataset[int(self.indices[index])]
        specification = CONDITIONS[self.condition]
        position = (self.positions["top_left_" + str(specification["size"])][index]
                    if specification["kind"] == "occlusion" else None)
        return perturb_image(image, self.condition, self.mean, self.std, position), target


def positions_for(indices, source):
    import numpy as np

    if source is None:
        rng = np.random.default_rng(42)
        return {"sample_index": indices.copy(), "seed": np.int64(42),
                **{"top_left_" + str(size): rng.integers(0, 33 - size, size=(len(indices), 2))
                   for size in (4, 8)}}
    with np.load(source, allow_pickle=False) as saved:
        ids = saved["sample_index"]
        if (ids.shape != indices.shape or len(np.unique(ids)) != len(indices)
                or not np.array_equal(np.sort(ids), np.sort(indices)) or int(saved["seed"]) != 42):
            raise ValueError("Occlusion positions do not refer to this exact validation split and seed 42")
        order = np.argsort(ids)
        rows = order[np.searchsorted(ids[order], indices)]
        result = {"sample_index": indices.copy(), "seed": np.int64(42)}
        for size in (4, 8):
            values = saved["top_left_" + str(size)]
            if (values.shape != (len(indices), 2) or not np.issubdtype(values.dtype, np.integer)
                    or np.any(values < 0) or np.any(values > 32 - size)):
                raise ValueError("Invalid occlusion coordinates")
            result["top_left_" + str(size)] = values[rows].astype(np.int64, copy=True)
    return result


def positions_fingerprint(positions):
    import numpy as np

    digest = hashlib.sha256()
    for key in sorted(positions):
        values = np.ascontiguousarray(positions[key], dtype=np.int64)
        digest.update(key.encode("utf-8"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def write_comparison(directory, records, indices, labels):
    import numpy as np
    from data import CLASSES

    predictions = {}
    for name, record in records.items():
        if record["status"] != "completed":
            continue
        predictions[name] = {}
        for condition in CONDITIONS:
            with np.load(directory / name / condition / "val_outputs.npz", allow_pickle=False) as raw:
                if not np.array_equal(raw["sample_index"], indices) or not np.array_equal(raw["true_label"], labels):
                    raise ValueError("Per-sample identifiers or labels differ across conditions/models")
                predictions[name][condition] = raw["logits"].argmax(axis=1)
    common = None
    if set(predictions) == set(ROBUSTNESS_MODELS):
        common = np.logical_and.reduce([values["clean"] == labels for values in predictions.values()])
    rows, class_rows = [], []
    for name, values in predictions.items():
        clean = records[name]["conditions"]["clean"]
        clean_correct = values["clean"] == labels
        for condition, predicted in values.items():
            metrics = records[name]["conditions"][condition]
            correct = predicted == labels
            row = {"model": name, "epoch": records[name]["checkpoint"]["epoch"], "condition": condition,
                   "samples": len(indices), "clean_accuracy": clean["accuracy"],
                   "perturbed_accuracy": metrics["accuracy"],
                   "accuracy_drop_percentage_points": 100 * (clean["accuracy"] - metrics["accuracy"]),
                   "new_errors_from_clean": int((clean_correct & ~correct).sum()),
                   "recovered_clean_errors": int((~clean_correct & correct).sum()),
                   "common_clean_correct_samples": int(common.sum()) if common is not None else None,
                   "accuracy_on_common_clean_correct": float(correct[common].mean())
                   if common is not None and common.any() else None}
            rows.append(row)
            for label in CLASSES:
                class_rows.append({"model": name, "condition": condition, "class": label,
                                   "samples": metrics["per_class_samples"][label],
                                   "clean_accuracy": clean["per_class_accuracy"][label],
                                   "perturbed_accuracy": metrics["per_class_accuracy"][label],
                                   "accuracy_drop_percentage_points": 100 * (
                                       clean["per_class_accuracy"][label] - metrics["per_class_accuracy"][label])})
    for filename, entries in (("accuracy_drop.csv", rows), ("per_class_drop.csv", class_rows)):
        if entries:
            with (directory / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
                writer.writeheader()
                writer.writerows(entries)
    complete = set(predictions) == set(ROBUSTNESS_MODELS)
    write_json(directory / "robustness_summary.json", {
        "status": "completed" if complete else "incomplete", "split": "validation", "seed": 42,
        "primary_population": "all 5000 validation samples", "accuracy_scale": "fraction in [0,1]",
        "drop_scale": "percentage points; negative values mean an accuracy increase",
        "secondary_population": "intersection of samples correctly classified when clean by all three fixed models",
        "interpretation": "paired single-seed validation sensitivity; this validation set was also used for checkpoint selection, so it is not an independent robustness test set",
        "test_set_evaluated": False, "training_performed": False, "models": records, "comparison": rows})
    return rows, complete


def compare_robustness(args):
    gpu = 0 if args.gpu is None else args.gpu
    data_root = args.data_dir or DATA_ROOT
    set_gpu_environment(gpu)
    import numpy as np

    reference = args.without_randaugment_run.resolve()
    split_path = reference / "split_indices.npz"
    with np.load(split_path, allow_pickle=False) as saved:
        split = {key: saved[key].copy() for key in saved.files}
    indices = split["val"]
    if (indices.shape != (5000,) or split["train"].shape != (45000,)
            or len(np.unique(np.concatenate((split["train"], indices)))) != 50000):
        raise ValueError("Expected the fixed disjoint 45000/5000 split")
    default_positions = args.complete_model_run.parent / "shared/occlusion_positions.npz"
    source_positions = args.positions_file or (default_positions if default_positions.is_file() else None)
    positions = positions_for(indices, source_positions)
    checkpoints, sources, configs, locks, records = {}, {}, {}, {}, {}
    runs = (args.without_randaugment_run, args.complete_model_run, args.random_erasing_run)
    for name, run in zip(ROBUSTNESS_MODELS, runs):
        try:
            lock_original = name == "complete_model" and run.resolve() == COMPLETE_MODEL_RUN.resolve()
            checkpoint, source, lock = read_checkpoint(run, name, lock_original=lock_original)
            with np.load(run / "split_indices.npz", allow_pickle=False) as saved:
                if set(saved.files) != set(split) or any(not np.array_equal(saved[key], value) for key, value in split.items()):
                    raise ValueError("All models must use exactly the same saved training/validation split")
            checkpoints[name], sources[name], locks[name] = checkpoint, source, lock
            configs[name] = runtime_config(source, data_root, args.num_workers, ROBUSTNESS_PROTOCOL, gpu=gpu)
            records[name] = {"status": "pending", "checkpoint": lock, "conditions": {}}
        except Exception as exc:
            records[name] = {"status": "skipped" if isinstance(exc, FileNotFoundError) else "failed",
                             "error": str(exc), "run_dir": str(run.resolve()), "retraining": False}
    plan = {"protocol": ROBUSTNESS_PROTOCOL, "seed": 42, "gpu": gpu,
            "reference_split_sha256": file_sha256(split_path), "conditions": CONDITIONS,
            "occlusion_positions_fingerprint": positions_fingerprint(positions),
            "positions_source": str(source_positions.resolve()) if source_positions is not None else "generated once with seed 42",
            "checkpoints": locks, "config_sha256": {name: fingerprint(config) for name, config in configs.items()},
            "test_set_evaluation": False, "training": False, "test_time_augmentation": False}
    if args.dry_run:
        print(json.dumps({"plan": plan, "models": records}, indent=2))
        return 0 if len(locks) == 3 else 1
    directory = output_directory(RESULTS_ROOT / "robustness", "validation_robustness", args.resume, plan)
    print("Results:", directory, flush=True)
    try:
        if not configs:
            write_json(directory / "status.json", {"status": "failed", "models": records, "reason": "No valid existing checkpoints"})
            return 1
        start_gpu()
        save_environment(directory)
        import torch
        from torchvision import datasets, transforms
        from data import make_loader, prepare_data, stratified_split
        from experiment import evaluate_split, seed_everything, write_npz

        config = configs.get("without_randaugment", next(iter(configs.values())))
        prepare_data(config["data"])
        dataset = datasets.CIFAR10(config["data"]["root"], train=True, download=False, transform=transforms.ToTensor())
        _, validation, _ = stratified_split(dataset.targets, config["data"]["root"], index_path=split_path)
        if not np.array_equal(validation, indices):
            raise ValueError("Dataset and shared validation split disagree")
        labels = np.asarray(dataset.targets, dtype=np.int64)[indices]
        positions_path = directory / "occlusion_positions.npz"
        if positions_path.exists():
            if positions_fingerprint(positions_for(indices, positions_path)) != positions_fingerprint(positions):
                raise ValueError("Stored evaluation occlusion positions changed")
        else:
            write_npz(positions_path, positions)
        if not (directory / "split_indices.npz").exists():
            (directory / "split_indices.npz").write_bytes(split_path.read_bytes())
        if file_sha256(directory / "split_indices.npz") != plan["reference_split_sha256"]:
            raise ValueError("Stored evaluation split changed")
        write_json(directory / "status.json", {"status": "evaluating", "test_set_evaluation": False})
        for name, checkpoint in checkpoints.items():
            model = None
            try:
                seed_everything(42)
                model = load_frozen_model(checkpoint)
                model_directory = directory / name
                model_directory.mkdir(exist_ok=True)
                write_json(model_directory / "source_config.json", sources[name])
                write_json(model_directory / "config.json", configs[name])
                write_json(model_directory / "checkpoint_lock.json", locks[name])
                for condition in CONDITIONS:
                    print(name + " / " + condition, flush=True)
                    condition_directory = model_directory / condition
                    condition_directory.mkdir(exist_ok=True)
                    view = ValidationView(dataset, indices, condition, positions,
                                          config["data"]["mean"], config["data"]["std"])
                    loader = make_loader(view, configs[name])
                    cached = (condition_directory / "val_metrics.json").exists()
                    torch.cuda.synchronize(0)
                    started = time.perf_counter()
                    metrics = evaluate_split(model, loader, condition_directory, "val", checkpoint,
                                             locks[name]["sha256"], configs[name])
                    torch.cuda.synchronize(0)
                    elapsed = time.perf_counter() - started
                    metrics.update({"condition": CONDITIONS[condition], "condition_name": condition,
                                    "evaluation_mode": "eval; fixed paired validation perturbation; hard labels; no training augmentations, CAM or TTA",
                                    "occlusion_positions_fingerprint": plan["occlusion_positions_fingerprint"],
                                    "reference_split_sha256": plan["reference_split_sha256"]})
                    metrics.setdefault("evaluation_seconds", None if cached else elapsed)
                    write_json(condition_directory / "val_metrics.json", metrics)
                    records[name]["conditions"][condition] = metrics
                records[name]["clean_accuracy_difference_from_saved_percentage_points"] = 100 * (
                    records[name]["conditions"]["clean"]["accuracy"] - checkpoint["val_accuracy"])
                if file_sha256(locks[name]["checkpoint"]) != locks[name]["sha256"]:
                    raise RuntimeError("Source checkpoint changed during evaluation")
                records[name]["status"] = "completed"
            except Exception as exc:
                records[name].update({"status": "failed", "error": str(exc), "retraining": False})
                print(name + " failed: " + str(exc), file=sys.stderr, flush=True)
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
            write_json(directory / "progress.json", records)
        rows, complete = write_comparison(directory, records, indices, labels)
        for row in rows:
            print("{model} {condition}: clean {clean_accuracy:.2%}, perturbed {perturbed_accuracy:.2%}, drop {accuracy_drop_percentage_points:+.2f} pp".format(**row), flush=True)
        write_json(directory / "status.json", {"status": "completed" if complete else "incomplete",
                                               "test_set_evaluation": False, "training_performed": False,
                                               "models": {name: value["status"] for name, value in records.items()}})
        return 0 if complete else 1
    except (Exception, KeyboardInterrupt) as exc:
        record_failure(directory, exc)
        raise


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 3 and argv[0] == "_worker" and argv[1] in ("check", "train"):
        return run_worker(argv[1], Path(argv[2]).resolve())
    args = arguments(argv)
    return args.action(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print("Error: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
