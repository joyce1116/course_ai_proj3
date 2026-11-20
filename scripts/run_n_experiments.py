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

from run_ablations import ROOT, experiment_config, read_json, write_json


E1 = ROOT / "results/ablations/suite_20261119T072348Z_e0a7c9de/E1/e1_20261119T072350Z_4f3ea7ee"
VARIANTS = {"N1": "random_erasing", "N2": "label_smoothing", "N3": "se_channel_attention"}
TEST_METRICS = ("loss", "accuracy", "macro_f1", "ece", "error_count", "wrong_mean_confidence",
                "wrong_with_confidence_at_least_0.9", "wrong_with_confidence_at_least_0.99",
                "multiclass_brier")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description="Run N1-N3 independently using the E1 training settings")
    parser.add_argument("--experiments", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--gpus", type=int, choices=(0,), default=0)
    parser.add_argument("--baseline-run", type=Path, default=E1, help="existing E1 result directory")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print configurations without training or writing files")
    return parser.parse_args(argv)


def metrics_record(directory):
    val = read_json(directory / "val_metrics.json")
    test = read_json(directory / "test_metrics.json")
    if val.get("best_checkpoint_sha256") != test.get("best_checkpoint_sha256"):
        raise ValueError("Validation and test must use the same best checkpoint")
    record = {"run_dir": str(directory), "best_epoch": test["best_epoch"],
              "val_loss": val["loss"], "val_accuracy": val["accuracy"]}
    record.update({"test_" + key: test[key] for key in TEST_METRICS})
    record.update({"cat_accuracy": test["per_class_accuracy"]["cat"],
                   "dog_accuracy": test["per_class_accuracy"]["dog"],
                   "cat_to_dog": test["confusion_matrix"][3][5],
                   "dog_to_cat": test["confusion_matrix"][5][3]})
    return record


def e1_settings(directory):
    config = read_json(directory / "config.json")
    reference = read_json(ROOT / "config/basic.json")
    if config["seed"] != 42 or config.get("ablation", {}).get("experiment") != "E1":
        raise ValueError("The reference must be the existing E1 experiment with seed 42")
    if config["model"] != reference["model"]:
        raise ValueError("E1 must use the original CifarCNN and Dropout settings")
    expected_training = {"epochs": 100, "batch_size": 128, "optimizer": "Adam", "lr": 0.001,
                         "weight_decay": 0.0001, "num_workers": config["training"]["num_workers"],
                         "scheduler": {"name": "cosine", "T_max": 99, "eta_min": 0.00001}}
    expected_data = reference["data"].copy()
    for key in ("root", "download", "download_urls"):
        if key in config["data"]:
            expected_data[key] = config["data"][key]
        else:
            expected_data.pop(key, None)
    if config["training"] != expected_training or config["data"] != expected_data:
        raise ValueError("E1 training, split, normalization and augmentation settings must be unchanged")
    record = {"experiment": "E1", "change": "reference", "status": "reused", **metrics_record(directory)}
    record["source_sha256"] = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("config.json", "split_indices.npz", "val_metrics.json", "test_metrics.json")}
    return config, record


def n_config(base, name, args, suite):
    config = experiment_config(base, name, args, suite)
    config["output"]["plot"] = False
    config["output"]["figures_root"] = str(ROOT / "figures/n_experiments" / suite.name / name)
    if name == "N1":
        config["data"]["random_erasing"] = {
            "p": 0.25, "scale": [0.02, 0.15], "ratio": [0.3, 3.3], "value": 0, "inplace": False}
    elif name == "N2":
        config["training"]["label_smoothing"] = 0.1
    return config


def run_one(name, config, suite, environment):
    record = {"experiment": name, "change": VARIANTS[name], "status": "failed", "returncode": 1}
    try:
        config_path = suite / (name + "_config.json")
        write_json(config_path, config)
        command = [sys.executable, "-u", str(ROOT / "run.py"), "train", "--config", str(config_path),
                   "--gpu", "0", "--run-name", name.lower()]
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
            if (directory / "status.json").is_file():
                record["status"] = read_json(directory / "status.json")["status"]
            if (directory / "test_metrics.json").is_file():
                record.update(metrics_record(directory))
    except Exception as exc:
        record.update({"status": "failed", "returncode": 1, "error": str(exc),
                       "traceback": traceback.format_exc()})
    if record["returncode"] != 0 or record["status"] != "completed":
        print(name + " failed; continuing with the next experiment.", flush=True)
    return record


def save_summary(suite, baseline, records):
    deltas = {}
    for row in records:
        if "test_accuracy" in row:
            deltas[row["experiment"]] = {
                "val_accuracy_percentage_points": 100 * (row["val_accuracy"] - baseline["val_accuracy"]),
                "test_accuracy_percentage_points": 100 * (row["test_accuracy"] - baseline["test_accuracy"]),
                **{key: row[key] - baseline[key] for key in (
                    "test_ece", "test_error_count", "test_wrong_with_confidence_at_least_0.9",
                    "test_wrong_with_confidence_at_least_0.99", "cat_to_dog", "dog_to_cat")}}
    rows = [baseline, *records]
    write_json(suite / "summary.json", {"seed": 42, "physical_gpu": 0, "reference": "E1",
                                         "accuracy_scale": "fraction in [0,1]",
                                         "runs": rows, "deltas_vs_E1": deltas})
    fields = ["experiment", "change", "status", "returncode"]
    fields.extend(key for key in baseline if key not in fields and key != "source_sha256")
    with (suite / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = arguments(argv)
    if len(set(args.experiments)) != len(args.experiments):
        raise ValueError("Each experiment may only be selected once")
    args.baseline_run = args.baseline_run.resolve()
    base, baseline = e1_settings(args.baseline_run)
    name = "suite_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    suite = ROOT / "results/n_experiments" / name
    configs = {key: n_config(base, key, args, suite) for key in args.experiments}
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
    return 0 if all(row["returncode"] == 0 and row["status"] == "completed" for row in records) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("Error: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
