import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import CLASSES, evaluation_data, test_data, training_data
from models.cnn import CifarCNN
from visualization import plot_results


GIB = 1024 ** 3
HISTORY_FIELDS = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
                  "train_samples", "val_samples", "seconds", "lr"]
PREDICTION_FIELDS = ["sample_index", "true_label", "true_class", "predicted_label",
                     "predicted_class", "confidence", "correct", "true_class_probability",
                     "cross_entropy", "top2_margin"]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def write_npz(path, values):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def record_reproducibility(directory, config, model, split_info):
    source_root = Path(__file__).resolve().parent
    files = ("run.py", "data.py", "experiment.py", "visualization.py", "models/cnn.py",
             "config/basic.json", "scripts/run_basic.sh", "requirements.txt")
    initial_state = hashlib.sha256()
    for name, value in model.state_dict().items():
        initial_state.update(name.encode("utf-8"))
        initial_state.update(value.detach().cpu().contiguous().numpy().tobytes())
    write_json(directory / "reproducibility.json", {
        "seed": config["seed"], "split_seed": config["data"]["split_seed"],
        "loader_generator_seed": config["seed"],
        "worker_seed": "torch.initial_seed() modulo 2**32 for NumPy and Python random",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "initial_state_sha256": initial_state.hexdigest(),
        "split_file": "split_indices.npz", "split_sha256": split_info["file_sha256"],
        "source_sha256": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                          for name in files},
    })


def new_directory(root, prefix):
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(root) / (prefix + "_" + name + "_" + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def environment_info():
    packages = {}
    for package in ("torch", "torchvision", "numpy", "matplotlib", "Pillow"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    info = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "packages": packages,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "command": sys.argv,
    }
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=10, check=True)
        info["nvidia_smi"] = [dict(zip(
            ("physical_index", "uuid", "name", "total_mib", "free_mib", "driver"),
            [part.strip() for part in line.split(",")]))
            for line in result.stdout.strip().splitlines()]
    except (OSError, subprocess.SubprocessError) as exc:
        info["nvidia_smi_error"] = str(exc)
    if info["cuda_available"]:
        properties = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        info["selected_cuda_device"] = {
            "logical_index": 0, "name": properties.name,
            "total_bytes": total, "free_bytes": free,
            "compute_capability": [properties.major, properties.minor],
        }
    return info


def model_info(config):
    model = CifarCNN(**config["model"])
    channels = config["model"]["channels"]
    head_channels = config["model"]["head_channels"]
    return {
        "name": "CifarCNN",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                    if parameter.requires_grad),
        "shapes": {
            "input": ["B", 3, 32, 32],
            "stage1": ["B", channels[0], 16, 16],
            "stage2": ["B", channels[1], 8, 8],
            "stage3": ["B", channels[2], 4, 4],
            "stage4": ["B", channels[3], 4, 4],
            "channel_mapping": ["B", head_channels, 4, 4],
            "global_pool": ["B", head_channels, 1, 1],
            "logits": ["B", config["model"]["num_classes"]],
        },
    }


def memory_estimate(config, parameters):
    batch = config["training"]["batch_size"]
    size = 32
    elements = batch * 3 * size * size
    for index, (channels, depth) in enumerate(zip(
            config["model"]["channels"], config["model"]["depths"])):
        elements += batch * channels * size * size * depth * 3
        if index < 3:
            size //= 2
            elements += batch * channels * size * size
    elements += batch * config["model"]["head_channels"] * (size * size * 3 + 2)
    elements += batch * config["model"]["num_classes"]
    safety = config["safety"]
    values = {
        "precision": "float32", "batch_size": batch,
        "weights_gradients_adam_moments_bytes": parameters * 4 * 4,
        "forward_activation_bytes": elements * 4,
        "activation_memory_multiplier": safety["activation_memory_multiplier"],
        "activation_and_backward_budget_bytes": int(
            elements * 4 * safety["activation_memory_multiplier"]),
        "workspace_budget_bytes": int(safety["workspace_gib"] * GIB),
        "context_budget_bytes": int(safety["context_gib"] * GIB),
    }
    values["estimated_peak_bytes"] = sum(values[key] for key in (
        "weights_gradients_adam_moments_bytes", "activation_and_backward_budget_bytes",
        "workspace_budget_bytes", "context_budget_bytes"))
    return values


def is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def gpu_preflight(config, directory):
    environment = environment_info()
    spec = model_info(config)
    estimate = memory_estimate(config, spec["parameters"])
    write_json(directory / "environment.json", environment)
    write_json(directory / "model.json", spec)
    report = {"requested_physical_gpu": config["gpu"], "estimate": estimate,
              "status": "blocked", "blockers": [], "preflight_executed": False}
    if not environment["cuda_available"]:
        report["blockers"].append("CUDA is unavailable in this Python environment")
    try:
        import torchvision
        report["torchvision_import"] = torchvision.__version__
    except Exception as exc:
        report["blockers"].append("torchvision import failed: " + str(exc))
    device_info = environment.get("selected_cuda_device")
    if device_info:
        reserve = max(int(device_info["total_bytes"] * config["safety"]["reserve_fraction"]),
                      int(config["safety"]["min_reserve_gib"] * GIB))
        report["reserve_bytes"] = reserve
        report["free_bytes_before_preflight"] = device_info["free_bytes"]
        if estimate["estimated_peak_bytes"] + reserve > device_info["free_bytes"]:
            report["blockers"].append("Estimated training memory plus reserve exceeds free memory")
    print("CifarCNN parameters: {:,}".format(spec["parameters"]), flush=True)
    print("CUDA available:", environment["cuda_available"], flush=True)
    print("Estimated FP32 peak: {:.2f} GiB".format(estimate["estimated_peak_bytes"] / GIB),
          flush=True)
    for gpu in environment.get("nvidia_smi", []):
        print("Physical GPU {physical_index}: {name}; total {total_mib} MiB; "
              "free {free_mib} MiB".format(**gpu), flush=True)
    if not report["blockers"]:
        model = inputs = labels = optimizer = loss = logits = None
        try:
            seed_everything(config["seed"])
            torch.cuda.reset_peak_memory_stats(0)
            model = CifarCNN(**config["model"]).to("cuda:0").train()
            optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"],
                                         weight_decay=config["training"]["weight_decay"],
                                         foreach=False)
            batch = config["training"]["batch_size"]
            inputs = torch.randn(batch, 3, 32, 32, device="cuda:0")
            labels = torch.randint(0, 10, (batch,), device="cuda:0")
            report["preflight_executed"] = True
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite preflight loss")
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(0)
            report["measured_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            report["measured_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
            free, total = torch.cuda.mem_get_info(0)
            report["free_bytes_after_step"] = free
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            external_used = max(0, total - free - reserved)
            peak_with_growth = int(max(estimate["estimated_peak_bytes"],
                                       report["measured_peak_reserved_bytes"],
                                       report["measured_peak_allocated_bytes"] * 1.25))
            report["measured_live_allocated_bytes"] = allocated
            report["budget_after_probe_bytes"] = peak_with_growth
            if total - external_used - peak_with_growth < report["reserve_bytes"]:
                report["blockers"].append("Measured peak with growth allowance leaves too little reserve")
            else:
                report["status"] = "passed"
        except Exception as exc:
            report["status"] = "oom" if is_oom(exc) else "failed"
            report["blockers"].append(str(exc))
        finally:
            model = inputs = labels = optimizer = loss = logits = None
            gc.collect()
            torch.cuda.empty_cache()
    write_json(directory / "gpu_check.json", report)
    print("GPU preflight:", report["status"], flush=True)
    for blocker in report["blockers"]:
        print(blocker, flush=True)
    return report["status"] == "passed"


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs = inputs.to("cuda:0", non_blocking=True)
            labels = labels.to("cuda:0", non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if training:
                loss.backward()
                optimizer.step()
            batch = labels.size(0)
            loss_sum += loss.item() * batch
            correct += (logits.argmax(1) == labels).sum().item()
            count += batch
    if count != len(loader.dataset):
        raise RuntimeError("The epoch did not retain all samples")
    return loss_sum / count, correct / count, count


def save_checkpoint(path, model, epoch, val_accuracy, config):
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                "val_accuracy": val_accuracy, "config": config}, temporary)
    os.replace(temporary, path)


def load_best(directory):
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    model = CifarCNN(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to("cuda:0").eval(), checkpoint


def classification_metrics(targets, logits, bins, top_confusions):
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probabilities = np.exp(log_probabilities)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    losses = -log_probabilities[np.arange(len(targets)), targets]
    ordered = np.sort(probabilities, axis=1)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    diagonal = confusion.diagonal()
    precision = np.divide(diagonal, predicted_count, out=np.zeros(10), where=predicted_count > 0)
    recall = np.divide(diagonal, support, out=np.zeros(10), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(10), where=precision + recall > 0)
    calibration = []
    bin_index = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = bin_index == index
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else None
        mean_confidence = float(confidence[selected].mean()) if count else None
        if count:
            ece += count / len(targets) * abs(accuracy - mean_confidence)
        calibration.append({"lower": index / bins, "upper": (index + 1) / bins,
                            "count": count, "accuracy": accuracy, "confidence": mean_confidence})
    pairs = sorted(((-int(confusion[true, predicted]), true, predicted)
                    for true in range(10) for predicted in range(10)
                    if true != predicted and confusion[true, predicted] > 0))
    errors = int((~correct).sum())
    metrics = {
        "samples": len(targets), "loss": float(losses.mean()), "accuracy": float(correct.mean()),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()), "error_count": errors,
        "mean_confidence": float(confidence.mean()),
        "correct_mean_confidence": float(confidence[correct].mean()) if correct.any() else None,
        "wrong_mean_confidence": float(confidence[~correct].mean()) if errors else None,
        "wrong_with_confidence_at_least_0.9": int(((~correct) & (confidence >= 0.9)).sum()),
        "wrong_with_confidence_at_least_0.99": int(((~correct) & (confidence >= 0.99)).sum()),
        "ece": float(ece), "calibration": calibration,
        "multiclass_brier": float((np.square(probabilities).sum(axis=1)
                                    - 2 * probabilities[np.arange(len(targets)), targets] + 1).mean()),
        "multiclass_brier_reduction": "mean over samples of the sum over ten classes",
        "per_class_accuracy": dict(zip(CLASSES, recall.tolist())),
        "per_class_samples": dict(zip(CLASSES, support.tolist())),
        "per_class": {label: {"support": int(support[index]),
                               "predicted_count": int(predicted_count[index]),
                               "correct": int(diagonal[index]), "precision": float(precision[index]),
                               "recall": float(recall[index]), "f1": float(f1[index])}
                      for index, label in enumerate(CLASSES)},
        "confusion_matrix": confusion.tolist(),
        "confusion_rows": "true labels", "confusion_columns": "predicted labels",
        "top_confusions": [{"true_label": true, "predicted_label": predicted,
                             "true_class": CLASSES[true], "predicted_class": CLASSES[predicted],
                             "count": -negative_count,
                             "fraction_of_all_errors": -negative_count / errors,
                             "fraction_of_true_class": -negative_count / int(support[true])}
                            for negative_count, true, predicted in pairs[:top_confusions]],
    }
    details = {"predicted_label": predictions, "confidence": confidence, "correct": correct,
               "true_class_probability": probabilities[np.arange(len(targets)), targets],
               "cross_entropy": losses, "top2_margin": ordered[:, -1] - ordered[:, -2]}
    return metrics, details


@torch.inference_mode()
def collect_outputs(model, loader, with_cams):
    dataset = loader.dataset
    base_dataset = dataset.dataset if hasattr(dataset, "indices") else dataset
    indices = np.asarray(dataset.indices if hasattr(dataset, "indices")
                         else np.arange(len(dataset)), dtype=np.int64)
    outputs, targets, true_cams, predicted_cams = [], [], [], []
    activation = {}

    def keep_maps(_module, _inputs, output):
        activation["maps"] = output

    handle = model.head[2].register_forward_hook(keep_maps) if with_cams else None
    model.eval()
    try:
        for inputs, labels in loader:
            inputs = inputs.to("cuda:0", non_blocking=True)
            device_labels = labels.to("cuda:0", non_blocking=True)
            logits = model(inputs)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite evaluation logits")
            outputs.append(logits.cpu().numpy())
            targets.append(labels.numpy().copy())
            if with_cams:
                maps = activation.pop("maps")
                weights = model.head[-1].weight
                true_cams.append(torch.einsum("bc,bchw->bhw", weights[device_labels], maps).cpu().numpy())
                predicted_cams.append(torch.einsum(
                    "bc,bchw->bhw", weights[logits.argmax(1)], maps).cpu().numpy())
    finally:
        if handle is not None:
            handle.remove()
    targets, logits = np.concatenate(targets), np.concatenate(outputs)
    if (len(targets) != len(dataset) or len(np.unique(indices)) != len(indices)
            or not np.array_equal(targets, np.asarray(base_dataset.targets)[indices])):
        raise RuntimeError("Evaluation sample indices and labels are inconsistent")
    cams = (np.stack([np.concatenate(true_cams), np.concatenate(predicted_cams)], axis=1)
            if with_cams else None)
    return indices, targets, logits, cams, base_dataset


def select_examples(base_dataset, indices, targets, details, cams, metrics, config):
    predictions, confidence = details["predicted_label"], details["confidence"]
    wrong = np.flatnonzero(~details["correct"])
    per_group = config["output"].get("error_examples_per_group", 4)
    rng = np.random.default_rng(config["seed"])
    selected, group_names, used = [], [], set()

    def add(positions, name):
        for position in positions:
            position = int(position)
            if position not in used:
                selected.append(position)
                group_names.append(name)
                used.add(position)

    add(rng.permutation(wrong)[:per_group], "Random errors")
    frequent = []
    for pair in metrics["top_confusions"]:
        candidates = np.flatnonzero((targets == pair["true_label"])
                                     & (predictions == pair["predicted_label"]))
        candidates = np.asarray([position for position in candidates if int(position) not in used])
        if len(candidates):
            median = np.median(confidence[candidates])
            frequent.append(candidates[np.argmin(np.abs(confidence[candidates] - median))])
        if len(frequent) >= per_group:
            break
    add(frequent, "Frequent confusions")
    ranked_wrong = sorted(wrong, key=lambda position: (-confidence[position], indices[position]))
    add([position for position in ranked_wrong if int(position) not in used][:per_group],
        "High-confidence errors")
    errors_by_class = np.bincount(targets[wrong], minlength=10)
    cam_pairs = []
    for label in np.argsort(-errors_by_class, kind="stable"):
        good = np.flatnonzero((targets == label) & details["correct"])
        bad = np.flatnonzero((targets == label) & ~details["correct"])
        if len(good) and len(bad):
            pair = [int(group[np.argmin(np.abs(confidence[group] - np.median(confidence[group])))])
                    for group in (good, bad)]
            cam_pairs.append(pair)
        if len(cam_pairs) >= config["output"].get("cam_classes", 2):
            break
    positions = np.unique(np.asarray(selected + [item for pair in cam_pairs for item in pair],
                                     dtype=np.int64))
    lookup = {int(position): index for index, position in enumerate(positions)}
    cache = {
        "images": base_dataset.data[indices[positions]], "sample_index": indices[positions],
        "true_label": targets[positions], "predicted_label": predictions[positions],
        "confidence": confidence[positions], "cam_true": cams[positions, 0],
        "cam_predicted": cams[positions, 1],
        "error_cache_index": np.asarray([lookup[position] for position in selected], dtype=np.int64),
        "error_group": np.asarray(group_names, dtype="U32"),
        "cam_pairs": np.asarray([[lookup[item] for item in pair] for pair in cam_pairs],
                                dtype=np.int64).reshape(-1, 2),
    }
    return cache


def evaluate_split(model, loader, directory, split, checkpoint, checkpoint_hash, config):
    metric_path = directory / (split + "_metrics.json")
    exported = split != "train_eval"
    artifacts = [directory / (split + suffix) for suffix in
                 ("_predictions.csv", "_outputs.npz", "_examples.npz")] if exported else []
    if metric_path.exists():
        existing = json.loads(metric_path.read_text(encoding="utf-8"))
        if (existing.get("schema_version") != 2
                or existing.get("best_checkpoint_sha256") != checkpoint_hash):
            raise FileExistsError("Existing evaluation belongs to another schema/checkpoint: " + str(metric_path))
        if not all(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()
                   == existing.get("artifact_sha256", {}).get(path.name) for path in artifacts):
            raise RuntimeError("Cached evaluation artifacts are missing or modified: " + split)
        print("Reusing completed evaluation:", split, flush=True)
        return existing
    print("Evaluating best checkpoint:", split, flush=True)
    indices, targets, logits, cams, base_dataset = collect_outputs(model, loader, exported)
    expected = {"train_eval": 45000, "val": 5000, "test": 10000}[split]
    if len(indices) != expected:
        raise RuntimeError("Unexpected evaluation sample count: " + split)
    metrics, details = classification_metrics(
        targets, logits, config["output"].get("calibration_bins", 10),
        config["output"].get("top_confusions", 5))
    metrics.update({
        "schema_version": 2, "split": split, "class_names": list(CLASSES),
        split + "_samples": len(targets), split + "_loss": metrics["loss"],
        split + "_accuracy": metrics["accuracy"], "accuracy_scale": "fraction in [0,1]",
        "confidence_definition": "maximum softmax probability",
        "sample_index_definition": "zero-based index in the official "
                                   + ("test" if split == "test" else "training") + " set",
        "evaluation_mode": "eval, no gradients, no augmentation, all samples retained",
        "best_epoch": checkpoint["epoch"], "best_val_accuracy": checkpoint["val_accuracy"],
        "best_checkpoint_sha256": checkpoint_hash,
        "calibration_binning": "equal-width confidence bins; left-closed, final bin includes 1",
        "artifact_sha256": {},
    })
    if exported:
        csv_path, raw_path, examples_path = artifacts
        temporary = csv_path.with_name(csv_path.name + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            for row, sample_index in enumerate(indices):
                true, predicted = int(targets[row]), int(details["predicted_label"][row])
                values = [int(sample_index), true, CLASSES[true], predicted, CLASSES[predicted],
                          float(details["confidence"][row]), bool(details["correct"][row]),
                          float(details["true_class_probability"][row]),
                          float(details["cross_entropy"][row]), float(details["top2_margin"][row])]
                writer.writerow(dict(zip(PREDICTION_FIELDS, values)))
        os.replace(temporary, csv_path)
        write_npz(raw_path, {"sample_index": indices, "true_label": targets, "logits": logits})
        cache = select_examples(base_dataset, indices, targets, details, cams, metrics, config)
        write_npz(examples_path, cache)
        metrics["example_selection"] = {
            "seed": config["seed"], "error_groups_disjoint": True,
            "random_errors": "uniform sample without replacement",
            "frequent_confusions": "median-confidence error from each leading directed confusion pair",
            "high_confidence_errors": "highest-confidence errors remaining after other groups",
            "cam_pairs": "correct/error pair from each highest-error true class; median confidence within each group",
            "cam_feature_shape": [config["model"]["head_channels"], 4, 4],
            "cam_values": "signed class-weighted activation sum before display ReLU and normalization",
        }
        metrics["artifact_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in artifacts}
    write_json(metric_path, metrics)
    print("{}: loss {:.4f}; accuracy {:.2%}; macro F1 {:.4f}; ECE {:.4f}".format(
        split, metrics["loss"], metrics["accuracy"], metrics["macro_f1"], metrics["ece"]), flush=True)
    if split == "test":
        for label, accuracy in metrics["per_class_accuracy"].items():
            print("{}: {:.2%}".format(label, accuracy), flush=True)
    return metrics


def evaluate_best(directory, config):
    for split in ("val", "test"):
        path = directory / (split + "_metrics.json")
        if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("schema_version") != 2:
            raise FileExistsError("Preserving existing legacy evaluation: " + str(path))
    split_path = directory / "split_indices.npz"
    if not split_path.exists():
        source = Path(config["data"]["root"]) / "split_seed42.npz"
        if not source.is_file():
            raise FileNotFoundError("Restore the original split indices before evaluation: " + str(source))
        shutil.copyfile(source, split_path)
    model, checkpoint = load_best(directory)
    checkpoint_hash = hashlib.sha256((directory / "best.pt").read_bytes()).hexdigest()
    train_loader, val_loader = evaluation_data(config, split_path)
    train_metrics = evaluate_split(model, train_loader, directory, "train_eval",
                                   checkpoint, checkpoint_hash, config)
    del train_loader
    val_metrics = evaluate_split(model, val_loader, directory, "val", checkpoint, checkpoint_hash, config)
    del val_loader
    _, test_loader = test_data(config)
    test_metrics = evaluate_split(model, test_loader, directory, "test", checkpoint, checkpoint_hash, config)
    summary = {
        "best_epoch": checkpoint["epoch"], "best_checkpoint_sha256": checkpoint_hash,
        "evaluation_mode": "same best checkpoint; eval mode; no augmentation; all samples",
        "train_val_accuracy_gap": train_metrics["accuracy"] - val_metrics["accuracy"],
        "val_accuracy_matches_checkpoint": abs(val_metrics["accuracy"] - checkpoint["val_accuracy"]) < 1e-12,
    }
    for split, metrics in (("train_eval", train_metrics), ("val", val_metrics), ("test", test_metrics)):
        summary[split] = {key: metrics[key] for key in
                          ("samples", "loss", "accuracy", "macro_f1", "ece", "multiclass_brier")}
    write_json(directory / "best_model_metrics.json", summary)
    return test_metrics


def failure_status(directory, exc):
    status = "oom" if is_oom(exc) else "failed"
    write_json(directory / "status.json", {
        "status": status, "error": str(exc), "traceback": traceback.format_exc(),
        "automatic_retry": False,
    })
    print("Stopped ({}): {}".format(status, exc), file=sys.stderr, flush=True)


def train(config, run_name):
    directory = new_directory(config["output"]["results_root"], run_name)
    write_json(directory / "config.json", config)
    write_json(directory / "status.json", {"status": "checking_gpu"})
    print("Results:", directory, flush=True)
    training_completed = False
    try:
        if not gpu_preflight(config, directory):
            write_json(directory / "status.json", {"status": "blocked_by_gpu_preflight"})
            return 2
        seed_everything(config["seed"])
        train_loader, val_loader, split_info = training_data(config)
        shutil.copyfile(split_info["path"], directory / "split_indices.npz")
        split_info["source_path"] = split_info["path"]
        split_info["path"] = "split_indices.npz"
        write_json(directory / "split.json", split_info)
        model = CifarCNN(**config["model"]).to("cuda:0")
        record_reproducibility(directory, config, model, split_info)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"],
                                     weight_decay=config["training"]["weight_decay"],
                                     foreach=False)
        criterion = nn.CrossEntropyLoss()
        best_accuracy = -1.0
        write_json(directory / "status.json", {"status": "training"})
        with (directory / "history.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            stream.flush()
            for epoch in range(1, config["training"]["epochs"] + 1):
                started = time.perf_counter()
                train_loss, train_acc, train_count = run_epoch(model, train_loader, criterion, optimizer)
                val_loss, val_acc, val_count = run_epoch(model, val_loader, criterion)
                writer.writerow(dict(zip(HISTORY_FIELDS, (
                    epoch, train_loss, train_acc, val_loss, val_acc, train_count, val_count,
                    time.perf_counter() - started, optimizer.param_groups[0]["lr"]))))
                stream.flush()
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    save_checkpoint(directory / "best.pt", model, epoch, val_acc, config)
                print("Epoch {:3d}/{:3d}  train {:.4f}/{:.2%}  val {:.4f}/{:.2%}".format(
                    epoch, config["training"]["epochs"], train_loss, train_acc, val_loss, val_acc),
                    flush=True)
        del model, optimizer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()
        training_completed = True
        write_json(directory / "status.json", {"status": "evaluating_best"})
        evaluate_best(directory, config)
        write_json(directory / "status.json", {"status": "evaluation_completed"})
    except Exception as exc:
        failure_status(directory, exc)
        if training_completed:
            print("Training is complete. Resume evaluation with run.py test --run-dir "
                  + str(directory) + "; no retraining is needed.", file=sys.stderr, flush=True)
        return 1
    try:
        write_json(directory / "status.json", {"status": "plotting"})
        plot_results(directory)
    except Exception as exc:
        write_json(directory / "plot_error.json", {"error": str(exc)})
        write_json(directory / "status.json", {"status": "plotting_failed",
                                               "training_completed": True, "evaluation_completed": True})
        print("Training and test completed; plotting failed: " + str(exc), file=sys.stderr)
        return 1
    write_json(directory / "status.json", {"status": "completed"})
    return 0


def test_only(directory, config):
    directory = Path(directory).resolve()
    if not (directory / "best.pt").is_file():
        raise FileNotFoundError(directory / "best.pt")
    check_directory = new_directory(directory, "test_gpu_check")
    write_json(check_directory / "config.json", config)
    try:
        if not gpu_preflight(config, check_directory):
            return 2
        seed_everything(config["seed"])
        evaluate_best(directory, config)
    except Exception as exc:
        failure_status(check_directory, exc)
        return 1
    plot_results(directory)
    write_json(directory / "status.json", {"status": "completed"})
    return 0
