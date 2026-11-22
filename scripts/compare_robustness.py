import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

from test_n15_locked import (E1_RUN, N10_RUN, N15_RUN, ROOT, file_sha256,
                             load_frozen_model, output_directory, read_checkpoint, read_json,
                             record_failure, runtime_config, save_environment,
                             set_gpu_environment, start_gpu, write_json)


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
    if set(predictions) == {"E1", "N15", "N10"}:
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
    complete = set(predictions) == {"E1", "N15", "N10"}
    write_json(directory / "robustness_summary.json", {
        "status": "completed" if complete else "incomplete", "split": "validation", "seed": 42,
        "primary_population": "all 5000 validation samples", "accuracy_scale": "fraction in [0,1]",
        "drop_scale": "percentage points; negative values mean an accuracy increase",
        "secondary_population": "intersection of samples correctly classified when clean by all three fixed models",
        "interpretation": "paired single-seed validation sensitivity; this validation set was also used for checkpoint selection, so it is not an independent robustness test set",
        "test_set_evaluated": False, "training_performed": False, "models": records, "comparison": rows})
    return rows, complete


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired E1/N15/N10 validation robustness evaluation using existing frozen checkpoints")
    parser.add_argument("--gpus", type=int, choices=(0,), default=0)
    parser.add_argument("--e1-run", type=Path, default=E1_RUN)
    parser.add_argument("--n15-run", type=Path, default=N15_RUN)
    parser.add_argument("--n10-run", type=Path, default=N10_RUN)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--positions-file", type=Path)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    set_gpu_environment(args.gpus)
    import numpy as np
    from recipes import fingerprint

    reference = args.e1_run.resolve()
    split_path = reference / "split_indices.npz"
    with np.load(split_path, allow_pickle=False) as saved:
        split = {key: saved[key].copy() for key in saved.files}
    indices = split["val"]
    if (indices.shape != (5000,) or split["train"].shape != (45000,)
            or len(np.unique(np.concatenate((split["train"], indices)))) != 50000):
        raise ValueError("Expected the fixed disjoint 45000/5000 split")
    default_positions = args.n15_run.parent / "shared/occlusion_positions.npz"
    source_positions = args.positions_file or (default_positions if default_positions.is_file() else None)
    positions = positions_for(indices, source_positions)
    checkpoints, sources, configs, locks, records = {}, {}, {}, {}, {}
    for name, run in (("E1", args.e1_run), ("N15", args.n15_run), ("N10", args.n10_run)):
        try:
            checkpoint, source, lock = read_checkpoint(run, name)
            with np.load(run / "split_indices.npz", allow_pickle=False) as saved:
                if set(saved.files) != set(split) or any(not np.array_equal(saved[key], value) for key, value in split.items()):
                    raise ValueError("All models must use exactly the same saved training/validation split")
            checkpoints[name], sources[name], locks[name] = checkpoint, source, lock
            configs[name] = runtime_config(source, args.data_dir, args.num_workers, "paired-val-robustness-v1")
            records[name] = {"status": "pending", "checkpoint": lock, "conditions": {}}
        except Exception as exc:
            records[name] = {"status": "skipped" if isinstance(exc, FileNotFoundError) else "failed",
                             "error": str(exc), "run_dir": str(run.resolve()), "retraining": False}
    plan = {"protocol": "paired-val-robustness-v1", "seed": 42, "gpu": 0,
            "reference_split_sha256": file_sha256(split_path), "conditions": CONDITIONS,
            "occlusion_positions_fingerprint": positions_fingerprint(positions),
            "positions_source": str(source_positions.resolve()) if source_positions is not None else "generated once with seed 42",
            "checkpoints": locks, "config_sha256": {name: fingerprint(config) for name, config in configs.items()},
            "test_set_evaluation": False, "training": False, "test_time_augmentation": False}
    if args.dry_run:
        print(json.dumps({"plan": plan, "models": records}, indent=2))
        return 0 if len(locks) == 3 else 1
    directory = output_directory(ROOT / "results/robustness", "e1_n15_n10", args.resume, plan)
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

        config = configs.get("E1", next(iter(configs.values())))
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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print("Error: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
