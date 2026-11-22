import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recipes import COMPARISONS, PROTOCOL, RECIPES, fingerprint, make_config, validate_e1, validate_recipe


E1 = ROOT / "results/ablations/suite_20261119T072348Z_e0a7c9de/E1/e1_20261119T072350Z_4f3ea7ee"
ENVIRONMENT = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0",
               "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "42",
               "PYTHONIOENCODING": "utf-8"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description="Serial N4-N28 training; all preflights before training; no analyses")
    parser.add_argument("--gpus", type=int, choices=(0,), default=0)
    parser.add_argument("--experiments", nargs="+", choices=RECIPES)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--resume", type=Path, help="existing suite directory")
    parser.add_argument("--dry-run", action="store_true", help="print configurations; no files or GPU work")
    parser.add_argument("--worker", choices=("check", "train"), help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def worker(args):
    os.environ.update(ENVIRONMENT)
    from run import validate_config
    from experiment import failure_status, gpu_preflight, train_recipe

    directory = args.run_dir.resolve()
    try:
        config = read_json(args.config)
        validate_config(config)
        if fingerprint(config) != fingerprint(read_json(directory / "config.json")):
            raise ValueError("Worker configuration differs from the stored run")
        if args.worker == "train":
            return train_recipe(config, directory)
        write_json(directory / "status.json", {"status": "checking_gpu"})
        passed = gpu_preflight(config, directory)
        report = read_json(directory / "gpu_check.json")
        write_json(directory / "status.json", {
            "status": "ready" if passed else "preflight_failed",
            "reason": None if passed else "; ".join(report["blockers"]),
            "config_sha256": fingerprint(config), "automatic_retry": False})
        return 0 if passed else 2
    except Exception as exc:
        failure_status(directory, exc)
        return 1


def child(phase, directory):
    environment = os.environ.copy()
    environment.update(ENVIRONMENT)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", phase,
               "--config", str(directory / "config.json"), "--run-dir", str(directory)]
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
                    try:
                        if process.poll() is None:
                            process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    except OSError:
                        pass
                    write_json(directory / "status.json", {
                        "status": "interrupted", "reason": "Launcher interrupted during " + phase,
                        "resume": "last completed epoch; partial epoch is replayed"})
                    raise
    except Exception as exc:
        write_json(directory / "status.json", {"status": "failed", "phase": phase,
                                               "reason": str(exc), "traceback": traceback.format_exc()})
        return 1


def completed_files(directory, config):
    required = ("status.json", "best.pt", "last.pt", "history.csv", "resources.json",
                "reproducibility.json", "environment.json", "model.json", "split_indices.npz",
                "train_eval_metrics.json", "val_metrics.json", "val_outputs.npz", "val_predictions.csv")
    if not all((directory / name).is_file() for name in required):
        return False
    status = read_json(directory / "status.json")
    if status.get("status") != "completed" or status.get("epochs_completed") != 100:
        return False
    if status.get("config_sha256") != fingerprint(config):
        return False
    resources = read_json(directory / "resources.json")
    if not resources.get("training_completed") or resources.get("completed_epochs") != 100:
        return False
    with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if [int(row["epoch"]) for row in history] != list(range(1, 101)):
        return False
    checkpoint_hash = hashlib.sha256((directory / "best.pt").read_bytes()).hexdigest()
    for split in ("train_eval", "val"):
        metrics = read_json(directory / (split + "_metrics.json"))
        if metrics.get("best_checkpoint_sha256") != checkpoint_hash:
            return False
        for name, digest in metrics.get("artifact_sha256", {}).items():
            path = directory / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                return False
    return True


def completed(directory, config):
    try:
        return completed_files(directory, config)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_assets(suite, manifest):
    import numpy as np

    shared = suite / "shared"
    shared.mkdir(exist_ok=True)
    reference = Path(manifest["reference_run"])
    with np.load(reference / "split_indices.npz", allow_pickle=False) as saved:
        indices = saved["val"].copy()
    if len(indices) != 5000 or len(np.unique(indices)) != 5000:
        raise ValueError("The E1 validation split is invalid")
    rng = np.random.default_rng(42)
    positions = {"sample_index": indices, "seed": np.int64(42)}
    for size in (4, 8):
        positions["top_left_" + str(size)] = rng.integers(0, 33 - size, size=(len(indices), 2))
    representative = np.sort(rng.choice(indices, size=50, replace=False)).tolist()
    occlusion_path = shared / "occlusion_positions.npz"
    if occlusion_path.exists():
        with np.load(occlusion_path, allow_pickle=False) as saved:
            if set(saved.files) != set(positions) or any(
                    not np.array_equal(saved[key], value) for key, value in positions.items()):
                raise ValueError("Saved shared occlusion positions differ from seed 42")
    else:
        with occlusion_path.open("xb") as stream:
            np.savez_compressed(stream, **positions)
    representatives_path = shared / "representative_samples.json"
    representatives = {"seed": 42, "sample_index": representative,
                       "selection": "fixed uniform validation sample; independent of model predictions",
                       "scope": "illustrative only; conclusions require all 5000 validation samples"}
    if representatives_path.exists():
        if read_json(representatives_path) != representatives:
            raise ValueError("Saved representative samples changed")
    else:
        write_json(representatives_path, representatives)
    return {str(path.relative_to(suite)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (occlusion_path, representatives_path)}


def analysis_inputs(suite, manifest):
    import numpy as np

    candidates = {Path(manifest["reference_run"]).resolve()}
    for path in (ROOT / "results").rglob("config.json"):
        if (path.parent / "history.csv").is_file() or (path.parent / "best.pt").is_file():
            candidates.add(path.parent.resolve())
    candidates.update((suite / name).resolve() for name in manifest["experiments"])
    inventory = []
    for directory in sorted(candidates):
        record = {"run_dir": str(directory), "analysis_performed": False}
        try:
            config = read_json(directory / "config.json")
            record["experiment"] = config.get("ablation", {}).get("experiment", "basic")
            has_checkpoint = (directory / "best.pt").is_file()
            record.update({"checkpoint_status": "available" if has_checkpoint else "missing",
                           "weight_dependent_analysis_status": "pending" if has_checkpoint else "skipped",
                           "reason": None if has_checkpoint else "best.pt missing; never retrain for analysis"})
            record["inputs"] = {name: str(directory / name) if (directory / name).is_file() else None
                                for name in ("best.pt", "config.json", "history.csv", "split_indices.npz",
                                             "train_eval_metrics.json", "val_metrics.json", "val_outputs.npz",
                                             "val_predictions.csv", "resources.json")}
            record["history_schema_valid"] = False
            if (directory / "history.csv").is_file():
                with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    record["history_schema_valid"] = set(("epoch", "val_accuracy", "val_loss", "lr", "seconds")).issubset(reader.fieldnames or [])
                    epochs = [int(row["epoch"]) for row in reader]
                    record["history_schema_valid"] = record["history_schema_valid"] and epochs == list(range(1, 101))
            record["validation_output_schema_valid"] = False
            raw_path = directory / "val_outputs.npz"
            if raw_path.is_file():
                with np.load(raw_path, allow_pickle=False) as raw:
                    if {"sample_index", "true_label", "logits"}.issubset(raw.files):
                        record["validation_output_schema_valid"] = (
                            raw["sample_index"].shape == (5000,) and raw["true_label"].shape == (5000,)
                            and raw["logits"].shape == (5000, 10)
                            and len(np.unique(raw["sample_index"])) == 5000
                            and bool(np.isfinite(raw["logits"]).all()))
            record["early_stop_input_status"] = "ready" if record["history_schema_valid"] else "missing_or_incomplete_history"
            record["sample_comparison_input_status"] = "ready" if record["validation_output_schema_valid"] else "missing_or_invalid_validation_outputs"
        except Exception as exc:
            record.update({"weight_dependent_analysis_status": "skipped", "reason": str(exc)})
        inventory.append(record)
    write_json(suite / "analysis_inputs.json", {
        "status": "inputs_only", "analyses_performed": False, "retrain_missing_checkpoints": False,
        "shared_assets_sha256": manifest["shared_assets_sha256"], "runs": inventory,
        "early_stopping": {"patience": [10, 15, 20], "improvement": "strictly greater than historical best",
                           "input": "history.csv", "restore": "best logged validation metrics before stop"},
        "robustness": {"split": "val", "independent_perturbations": True, "translations_x": [-2, 2],
                       "translation_fill_normalized": 0, "circular_translation": False,
                       "brightness_factors_before_normalization": [0.7, 1.3], "clip_pixels": [0, 255],
                       "occlusion_sizes": [4, 8], "occlusion_fill_normalized": 0,
                       "positions": "shared/occlusion_positions.npz", "position_order": "row, column"},
        "tta": {"split": "val", "views": ["original", "horizontal_flip"], "forward_passes": 2,
                "aggregation": "mean class probabilities; never average logits",
                "timing": "separate single-view and two-view timings; synchronize CUDA around timing"},
        "sample_comparison": {"default_reference": "E1", "component_references": COMPARISONS,
                              "join": "official training sample_index; verify identical labels and splits",
                              "future_counts": ["corrected_errors", "new_errors", "common_errors"],
                              "representatives": "shared/representative_samples.json"}})


def summary(suite, manifest):
    rows = []
    for name in manifest["experiments"]:
        directory = suite / name
        status_path = directory / "status.json"
        row = {"experiment": name, "run_dir": str(directory), "status": "pending"}
        if status_path.is_file():
            row.update(read_json(status_path))
        if (directory / "val_metrics.json").is_file():
            metrics = read_json(directory / "val_metrics.json")
            row["val"] = metrics
            row.update({"best_epoch": metrics["best_epoch"], "val_accuracy": metrics["accuracy"],
                        "val_loss": metrics["loss"], "macro_f1": metrics["macro_f1"], "ece": metrics["ece"],
                        "high_confidence_errors_90": metrics["wrong_with_confidence_at_least_0.9"],
                        "high_confidence_errors_99": metrics["wrong_with_confidence_at_least_0.99"],
                        "cat_to_dog": metrics["confusion_matrix"][3][5],
                        "dog_to_cat": metrics["confusion_matrix"][5][3]})
        for key, filename in (("train_eval", "train_eval_metrics.json"), ("model", "model.json"),
                              ("resources", "resources.json")):
            if (directory / filename).is_file():
                row[key] = read_json(directory / filename)
        if "model" in row:
            row["parameters"] = row["model"]["parameters"]
        row.update({key: row.get("resources", {}).get(key) for key in (
            "total_training_seconds", "total_training_wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes")})
        rows.append(row)
    write_json(suite / "summary.json", {"seed": 42, "physical_gpu": 0, "protocol": PROTOCOL,
                                         "test_evaluation": False, "cam": False, "plots": False,
                                         "analyses_performed": False, "runs": rows})
    fields = ("experiment", "status", "reason", "error", "best_epoch", "val_accuracy", "val_loss",
              "macro_f1", "ece", "high_confidence_errors_90", "high_confidence_errors_99",
              "cat_to_dog", "dog_to_cat", "parameters", "peak_allocated_bytes", "peak_reserved_bytes",
              "total_training_seconds", "total_training_wall_seconds", "run_dir")
    with (suite / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    args = arguments(argv)
    if args.worker:
        if args.config is None or args.run_dir is None:
            raise ValueError("Worker requires config and run-dir")
        return worker(args)
    if args.resume:
        suite = args.resume.resolve()
        manifest = read_json(suite / "suite.json")
        if manifest["protocol"] != PROTOCOL:
            raise ValueError("Not an N4-N28 suite")
        reference = Path(manifest["reference_run"])
        if args.baseline_run is not None and args.baseline_run.resolve() != reference:
            raise ValueError("Cannot change the reference while resuming")
        base = manifest["reference_config"]
        data_root = args.data_dir if args.data_dir is not None else manifest["data_root"]
        workers = args.num_workers if args.num_workers is not None else manifest["num_workers"]
        no_download = args.no_download or manifest["no_download"]
        selected = args.experiments if args.experiments is not None else manifest["experiments"]
        if not set(selected).issubset(manifest["experiments"]):
            raise ValueError("Cannot add new experiments to an existing suite")
    else:
        reference = (args.baseline_run or E1).resolve()
        base = read_json(reference / "config.json")
        data_root = (args.data_dir or ROOT / "dataset").resolve()
        workers = args.num_workers if args.num_workers is not None else base["training"]["num_workers"]
        no_download = args.no_download
        selected = args.experiments if args.experiments is not None else list(RECIPES)
        suite = ROOT / "results/extended_experiments" / (
            "suite_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
        manifest = {"protocol": PROTOCOL, "reference_run": str(reference), "reference_config": base,
                    "experiments": selected, "data_root": str(data_root), "num_workers": workers,
                    "no_download": no_download}
    if len(set(selected)) != len(selected):
        raise ValueError("Each experiment may only be selected once")
    validate_e1(base)
    split_hash = hashlib.sha256((reference / "split_indices.npz").read_bytes()).hexdigest()
    if args.resume and manifest["reference_split_sha256"] != split_hash:
        raise ValueError("The reference split changed")
    manifest["reference_split_sha256"] = split_hash
    configs = {name: make_config(base, name, reference, suite, data_root, workers, no_download) for name in selected}
    for name, config in configs.items():
        validate_recipe(config)
        stored = suite / name / "config.json"
        if args.resume and stored.exists() and fingerprint(read_json(stored)) != fingerprint(config):
            raise ValueError("Configuration changed for " + name + "; use a new suite")
    if args.dry_run:
        print(json.dumps({"suite": str(suite), "experiments": configs, "analyses_performed": False}, indent=2))
        return 0
    if not args.resume:
        suite.mkdir(parents=True, exist_ok=False)
    write_json(suite / "suite.json", manifest)
    for name, config in configs.items():
        directory = suite / name
        directory.mkdir(exist_ok=True)
        if not (directory / "config.json").exists():
            write_json(directory / "config.json", config)
    assets = prepare_assets(suite, manifest)
    if args.resume and manifest.get("shared_assets_sha256") not in (None, assets):
        raise ValueError("Shared analysis inputs changed")
    manifest["shared_assets_sha256"] = assets
    write_json(suite / "suite.json", manifest)
    analysis_inputs(suite, manifest)
    print("Suite: " + str(suite), flush=True)
    ready = []
    failed = False
    try:
        for name, config in configs.items():
            directory = suite / name
            if completed(directory, config):
                print(name + ": reusing completed experiment", flush=True)
                continue
            print(name + ": preflight", flush=True)
            if child("check", directory) == 0:
                ready.append(name)
            else:
                failed = True
            summary(suite, manifest)
        for name in ready:
            print(name + ": training", flush=True)
            code = child("train", suite / name)
            if code != 0 or not completed(suite / name, configs[name]):
                failed = True
                if code == 0:
                    write_json(suite / name / "status.json", {
                        "status": "failed", "reason": "Worker exited without a complete, consistent set of artifacts"})
            summary(suite, manifest)
    finally:
        summary(suite, manifest)
        analysis_inputs(suite, manifest)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; use --resume with the printed suite directory.", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print("Error: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
