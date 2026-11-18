import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
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

from data import CLASSES, test_data, training_data
from models.cnn import CifarCNN
from visualization import plot_results


GIB = 1024 ** 3
HISTORY_FIELDS = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
                  "train_samples", "val_samples", "seconds", "lr"]
PREDICTION_FIELDS = ["sample_index", "true_label", "true_class", "predicted_label",
                     "predicted_class", "confidence"]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


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


def select_misclassified(images, targets, predictions, confidence, limit):
    groups = [list(np.flatnonzero((targets == label) & (targets != predictions)))
              for label in range(10)]
    groups = [sorted(group, key=lambda index: (-confidence[index], index)) for group in groups]
    indices = []
    for rank in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if rank < len(group):
                indices.append(group[rank])
                if len(indices) == limit:
                    break
        if len(indices) == limit:
            break
    indices = np.asarray(indices, dtype=np.int64)
    return {"images": images[indices], "sample_index": indices,
            "true_label": targets[indices], "predicted_label": predictions[indices],
            "confidence": confidence[indices]}


@torch.inference_mode()
def evaluate_test(directory, config):
    for filename in ("test_predictions.csv", "test_metrics.json", "misclassified_samples.npz"):
        if (directory / filename).exists():
            raise FileExistsError("Refusing to overwrite final test output: " + filename)
    model, checkpoint = load_best(directory)
    dataset, loader = test_data(config)
    true_labels, predicted_labels, confidences = [], [], []
    loss_sum = 0.0
    for inputs, labels in loader:
        logits = model(inputs.to("cuda:0", non_blocking=True))
        labels_device = labels.to("cuda:0", non_blocking=True)
        loss_sum += nn.functional.cross_entropy(logits, labels_device, reduction="sum").item()
        confidence, predictions = logits.softmax(1).max(1)
        true_labels.append(labels.numpy())
        predicted_labels.append(predictions.cpu().numpy())
        confidences.append(confidence.cpu().numpy())
    targets = np.concatenate(true_labels)
    predictions = np.concatenate(predicted_labels)
    confidence = np.concatenate(confidences)
    if len(targets) != 10000 or not np.array_equal(targets, np.asarray(dataset.targets)):
        raise RuntimeError("Expected all 10,000 test images in official index order")
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    per_class = confusion.diagonal() / confusion.sum(axis=1)
    metrics = {
        "test_samples": len(targets), "test_loss": loss_sum / len(targets),
        "test_accuracy": float(np.mean(targets == predictions)),
        "accuracy_scale": "fraction in [0,1]", "confidence_definition": "maximum softmax probability",
        "sample_index_definition": "zero-based index in the official test set",
        "best_epoch": checkpoint["epoch"], "best_val_accuracy": checkpoint["val_accuracy"],
        "best_checkpoint_sha256": hashlib.sha256((directory / "best.pt").read_bytes()).hexdigest(),
        "class_names": list(CLASSES),
        "per_class_accuracy": dict(zip(CLASSES, per_class.tolist())),
        "per_class_samples": dict(zip(CLASSES, confusion.sum(axis=1).tolist())),
        "confusion_matrix": confusion.tolist(),
        "confusion_rows": "true labels", "confusion_columns": "predicted labels",
        "misclassified_selection": "descending confidence within each true class, class round-robin",
    }
    with (directory / "test_predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for index, (true, predicted, score) in enumerate(zip(targets, predictions, confidence)):
            writer.writerow(dict(zip(PREDICTION_FIELDS, (
                index, int(true), CLASSES[true], int(predicted), CLASSES[predicted], float(score)))))
    samples = select_misclassified(dataset.data, targets, predictions, confidence,
                                  config["output"]["misclassified_count"])
    with (directory / "misclassified_samples.npz").open("xb") as stream:
        np.savez_compressed(stream, **samples)
    write_json(directory / "test_metrics.json", metrics)
    print("Best epoch: {}; test accuracy: {:.2%}".format(checkpoint["epoch"], metrics["test_accuracy"]),
          flush=True)
    for label, accuracy in metrics["per_class_accuracy"].items():
        print("{}: {:.2%}".format(label, accuracy), flush=True)
    return metrics


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
    try:
        if not gpu_preflight(config, directory):
            write_json(directory / "status.json", {"status": "blocked_by_gpu_preflight"})
            return 2
        seed_everything(config["seed"])
        train_loader, val_loader, split_info = training_data(config)
        write_json(directory / "split.json", split_info)
        model = CifarCNN(**config["model"]).to("cuda:0")
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
        write_json(directory / "status.json", {"status": "final_test"})
        evaluate_test(directory, config)
        write_json(directory / "status.json", {"status": "completed"})
    except Exception as exc:
        failure_status(directory, exc)
        return 1
    try:
        plot_results(directory)
    except Exception as exc:
        write_json(directory / "plot_error.json", {"error": str(exc)})
        print("Training and test completed; plotting failed: " + str(exc), file=sys.stderr)
        return 1
    return 0


def test_only(directory, config):
    directory = Path(directory).resolve()
    if not (directory / "best.pt").is_file():
        raise FileNotFoundError(directory / "best.pt")
    for filename in ("test_predictions.csv", "test_metrics.json", "misclassified_samples.npz"):
        if (directory / filename).exists():
            raise FileExistsError("Final test outputs already exist; use plot to replot")
    check_directory = new_directory(directory, "test_gpu_check")
    write_json(check_directory / "config.json", config)
    try:
        if not gpu_preflight(config, check_directory):
            return 2
        seed_everything(config["seed"])
        evaluate_test(directory, config)
    except Exception as exc:
        failure_status(check_directory, exc)
        return 1
    plot_results(directory)
    write_json(directory / "status.json", {"status": "completed"})
    return 0
