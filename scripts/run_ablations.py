import argparse
import copy
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
BASELINE = ROOT / "results/basic/cifarcnn_20261118T160642Z_93925a89"
VARIANTS = {
    "E1": {"multiscale_fusion": False, "spatial_attention": False},
    "E2": {"multiscale_fusion": True, "spatial_attention": False},
    "E3": {"multiscale_fusion": False, "spatial_attention": True},
    "E4": {"multiscale_fusion": True, "spatial_attention": True},
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description="Run E1-E4 with the existing fixed-LR baseline")
    parser.add_argument("--experiments", nargs="+", choices=VARIANTS,
                        default=list(VARIANTS), help="default: E1 E2 E3 E4")
    parser.add_argument("--gpus", type=int, choices=(0,), default=0,
                        help="physical GPU; fixed to 0 for this suite")
    parser.add_argument("--baseline-run", type=Path, default=BASELINE)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print settings without importing torch or writing files")
    return parser.parse_args(argv)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def baseline_settings(directory):
    config = read_json(directory / "config.json")
    expected_model = {"channels": [64, 128, 256, 384], "depths": [2, 2, 3, 3],
                      "head_channels": 128, "dropout": 0.3, "num_classes": 10}
    expected_training = {"epochs": 100, "batch_size": 128, "optimizer": "Adam",
                         "lr": 0.001, "weight_decay": 0.0001}
    expected_data = {"split_seed": 42, "val_size": 5000, "mean": [0.5] * 3,
                     "std": [0.5] * 3, "crop_padding": 4,
                     "horizontal_flip_probability": 0.5}
    if config["seed"] != 42 or config["model"] != expected_model:
        raise ValueError("Baseline must use seed 42 and the original CifarCNN")
    for section, expected in (("training", expected_training), ("data", expected_data)):
        if any(config[section].get(key) != value for key, value in expected.items()):
            raise ValueError("Baseline has unexpected " + section + " settings")
    if "ablation" in config or "scheduler" in config["training"]:
        raise ValueError("Baseline must be the existing fixed-LR experiment")
    required = ("config.json", "split_indices.npz", "val_metrics.json", "test_metrics.json")
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in required}
    val, test = read_json(directory / "val_metrics.json"), read_json(directory / "test_metrics.json")
    if val.get("best_checkpoint_sha256") != test.get("best_checkpoint_sha256"):
        raise ValueError("Baseline validation/test results must use the same best checkpoint")
    record = {"experiment": "basic", "status": "reused", "run_dir": str(directory),
              "best_epoch": val["best_epoch"], "val_accuracy": val["accuracy"],
              "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"],
              "cat_accuracy": test["per_class_accuracy"]["cat"],
              "dog_accuracy": test["per_class_accuracy"]["dog"],
              "cat_to_dog": test["confusion_matrix"][3][5],
              "dog_to_cat": test["confusion_matrix"][5][3],
              "source_sha256": hashes}
    return config, record


def experiment_config(base, name, args, suite):
    config = copy.deepcopy(base)
    config["gpu"] = args.gpus
    config["data"]["root"] = str(args.data_dir.resolve())
    if args.no_download:
        config["data"]["download"] = False
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("num-workers must be nonnegative")
        config["training"]["num_workers"] = args.num_workers
    config["training"]["scheduler"] = {"name": "cosine", "T_max": 99, "eta_min": 0.00001}
    config["ablation"] = {"experiment": name, "baseline_run": str(args.baseline_run.resolve())}
    config["output"]["results_root"] = str(suite / name)
    config["output"]["figures_root"] = str(ROOT / "figures/ablations" / suite.name / name)
    return config


def run_one(name, config, suite, environment):
    record = {"experiment": name, **VARIANTS[name], "status": "failed"}
    config_path = suite / (name + "_config.json")
    try:
        write_json(config_path, config)
        command = [sys.executable, "-u", str(ROOT / "run.py"), "train",
                   "--config", str(config_path), "--gpu", "0", "--run-name", name.lower()]
        print("\n=== {}: {} ===".format(name, VARIANTS[name]), flush=True)
        with (suite / (name + ".log")).open("x", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                record["returncode"] = process.wait()
        directories = sorted(Path(config["output"]["results_root"]).glob(name.lower() + "_*"))
        if directories:
            directory = directories[-1]
            record["run_dir"] = str(directory)
            status_path = directory / "status.json"
            if status_path.is_file():
                record["status"] = read_json(status_path)["status"]
            test_path = directory / "test_metrics.json"
            if test_path.is_file():
                metrics = read_json(test_path)
                record.update({
                    "best_epoch": metrics["best_epoch"],
                    "val_accuracy": metrics["best_val_accuracy"],
                    "test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"],
                    "cat_accuracy": metrics["per_class_accuracy"]["cat"],
                    "dog_accuracy": metrics["per_class_accuracy"]["dog"],
                    "cat_to_dog": metrics["confusion_matrix"][3][5],
                    "dog_to_cat": metrics["confusion_matrix"][5][3],
                })
        if record["returncode"] != 0 or record["status"] != "completed":
            print(name + " failed; continuing with the next experiment.", flush=True)
    except Exception as exc:
        record.update({"status": "failed", "returncode": 1, "error": str(exc),
                       "traceback": traceback.format_exc()})
        print(name + " failed: " + str(exc) + "; continuing.", flush=True)
    return record


def save_summary(suite, baseline, records):
    rows = [baseline, *records]
    effects = {}
    indexed = {row["experiment"]: row for row in rows}
    comparisons = {"cosine_vs_fixed_lr": ("E1", "basic"),
                   "fusion_vs_E1": ("E2", "E1"), "attention_vs_E1": ("E3", "E1"),
                   "combined_vs_E1": ("E4", "E1"), "fusion_with_attention": ("E4", "E3"),
                   "attention_with_fusion": ("E4", "E2")}
    for metric in ("val_accuracy", "test_accuracy"):
        values = {key: row[metric] for key, row in indexed.items() if metric in row}
        effects[metric + "_percentage_points"] = {
            label: 100 * (values[left] - values[right])
            for label, (left, right) in comparisons.items() if left in values and right in values}
        if all(key in values for key in VARIANTS):
            effects[metric + "_percentage_points"]["interaction"] = 100 * (
                values["E4"] - values["E2"] - values["E3"] + values["E1"])
    write_json(suite / "summary.json", {"seed": 42, "physical_gpu": 0,
                                         "accuracy_scale": "fraction in [0,1]",
                                         "runs": rows, "effects": effects})
    fields = ("experiment", "status", "returncode", "multiscale_fusion", "spatial_attention",
              "best_epoch", "val_accuracy", "test_accuracy", "test_macro_f1", "cat_accuracy",
              "dog_accuracy", "cat_to_dog", "dog_to_cat", "run_dir")
    with (suite / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = arguments(argv)
    if len(set(args.experiments)) != len(args.experiments):
        raise ValueError("Each experiment may only be selected once")
    args.baseline_run = args.baseline_run.resolve()
    base, baseline = baseline_settings(args.baseline_run)
    name = "suite_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    suite = ROOT / "results/ablations" / name
    configs = {key: experiment_config(base, key, args, suite) for key in args.experiments}
    if args.dry_run:
        print(json.dumps({"baseline": baseline, "experiments": configs}, indent=2, ensure_ascii=False))
        return 0
    suite.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0",
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "42",
                        "PYTHONIOENCODING": "utf-8"})
    records = []
    save_summary(suite, baseline, records)
    print("Suite results: " + str(suite), flush=True)
    for key, config in configs.items():
        records.append(run_one(key, config, suite, environment))
        save_summary(suite, baseline, records)
    print("\nComparison: " + str(suite / "comparison.csv"), flush=True)
    return 0 if all(row.get("returncode") == 0 and row["status"] == "completed"
                    for row in records) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("Error: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
