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

from data import CLASSES, evaluation_data, mix_batch, test_data, training_data, training_transform
from models.ablations import STRUCTURAL_COMPARISONS, build_model
from recipes import (BASE_MODEL, COMPLETE_MODEL, experiment_name, fingerprint, is_resumable,
                     training_definition, uses_recipe_model)
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
             "models/ablations.py", "recipes.py", "config/final.json", "requirements.txt")
    if experiment_name(config) == COMPLETE_MODEL:
        files += ("scripts/run_final_model.py",)
    else:
        files += ("scripts/run_additional_experiments.py",)
    version = {}
    for key, command in (("commit", ["git", "rev-parse", "HEAD"]),
                         ("working_tree", ["git", "status", "--porcelain"])):
        try:
            version[key] = subprocess.run(command, cwd=source_root, check=True, timeout=10,
                                          capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            version[key] = str(exc)
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
        "code_version": version, "config_sha256": fingerprint(config),
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
    model = build_model(config)
    name = experiment_name(config)
    channels = config["model"]["channels"]
    head_channels = config["model"]["head_channels"]
    spec = {
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
    spec["experiment"] = name
    if name in STRUCTURAL_COMPARISONS:
        multiscale, attention = STRUCTURAL_COMPARISONS[name]
        size = 8 if multiscale else 4
        spec["name"] = "CifarCNN_" + name
        spec["multiscale_fusion"] = multiscale
        spec["spatial_attention"] = attention
        spec["shapes"]["pre_pool"] = ["B", head_channels, size, size]
        if multiscale:
            spec["shapes"]["middle_projection"] = ["B", head_channels, 8, 8]
            spec["shapes"]["deep_upsampled"] = ["B", head_channels, 8, 8]
            spec["shapes"]["concatenation"] = ["B", head_channels * 2, 8, 8]
            spec["shapes"]["fusion"] = ["B", head_channels, 8, 8]
        if attention:
            spec["shapes"]["spatial_weights"] = ["B", 1, size, size]
        if name == "head_se":
            spec["se_channel_attention"] = True
            spec["shapes"]["se_descriptor"] = ["B", head_channels]
            spec["shapes"]["se_hidden"] = ["B", 8]
            spec["shapes"]["se_weights"] = ["B", head_channels, 1, 1]
    if uses_recipe_model(config):
        spec["structure"] = config["structure"]
        spec["spatial_attention"] = config["structure"].get("spatial_attention", False)
        if config["structure"].get("se"):
            size = 8 if config["structure"]["se"] == "stage2" else 4
            spec["shapes"]["se_input"] = ["B", head_channels, size, size]
            spec["shapes"]["se_hidden"] = ["B", 8]
            spec["shapes"]["se_weights"] = ["B", head_channels, 1, 1]
        if spec["spatial_attention"]:
            spec["shapes"]["spatial_weights"] = ["B", 1, 4, 4]
        spec["compute"] = compute_cost(model)
        if config["structure"].get("residual"):
            from models.cnn import CifarCNN
            baseline = CifarCNN(**BASE_MODEL)
            baseline_cost = compute_cost(baseline)
            spec["parameters_added_vs_without_randaugment"] = spec["parameters"] - sum(
                p.numel() for p in baseline.parameters())
            spec["compute"]["macs_added_vs_without_randaugment"] = (
                spec["compute"]["macs_per_image"] - baseline_cost["macs_per_image"])
            spec["compute"]["residual_additions_per_image"] = sum(
                c * s * s for c, s in zip(config["model"]["channels"], (32, 16, 8, 4)))
    return spec


def compute_cost(model):
    macs = 0
    handles = []

    def count(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            macs += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(count))
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 3, 32, 32))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return {"macs_per_image": macs, "multiply_add_flops_per_image": 2 * macs,
            "scope": "Conv2d and Linear only; excludes BN, activations, pooling and elementwise operations"}


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
    name = experiment_name(config)
    if name in STRUCTURAL_COMPARISONS:
        multiscale, attention = STRUCTURAL_COMPARISONS[name]
        head_channels = config["model"]["head_channels"]
        if multiscale:
            elements += batch * head_channels * 8 * 8 * 10
        if attention:
            elements += batch * (head_channels + 4) * (8 if multiscale else 4) ** 2
        if name == "head_se":
            elements += batch * (head_channels * (size * size + 3) + 8)
    if uses_recipe_model(config):
        elements += batch * 3 * 32 * 32 * 4
        if config["structure"].get("residual"):
            elements += batch * sum(c * s * s * 5 for c, s in zip(
                config["model"]["channels"], (32, 16, 8, 4)))
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
              "status": "blocked", "blockers": [], "preflight_executed": False,
              "config_sha256": fingerprint(config)}
    if not environment["cuda_available"]:
        report["blockers"].append("CUDA is unavailable in this Python environment")
    try:
        import torchvision
        report["torchvision_import"] = torchvision.__version__
        if is_resumable(config) or experiment_name(config) == COMPLETE_MODEL:
            from PIL import Image
            sample = training_transform(config["data"])(Image.new("RGB", (32, 32), (128, 128, 128)))
            if tuple(sample.shape) != (3, 32, 32) or not torch.isfinite(sample).all():
                raise ValueError("Training transforms produced invalid input")
            report["training_transform_checked"] = True
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
        model = inputs = labels = optimizer = loss = logits = scheduler = None
        try:
            seed_everything(config["seed"])
            torch.cuda.reset_peak_memory_stats(0)
            model = build_model(config).to("cuda:0").train()
            optimizer = make_optimizer(model, config)
            scheduler = make_scheduler(optimizer, config)
            batch = config["training"]["batch_size"]
            inputs = torch.randn(batch, 3, 32, 32, device="cuda:0")
            labels = torch.randint(0, 10, (batch,), device="cuda:0")
            inputs, labels, paired_labels, weight = mix_batch(
                inputs, labels, config["training"].get("batch_mixing"), force=True)
            report["preflight_executed"] = True
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits, labels, label_smoothing=config["training"].get("label_smoothing", 0.0))
            if paired_labels is not None:
                loss = weight * loss + (1 - weight) * nn.functional.cross_entropy(logits, paired_labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite preflight loss")
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(float(loss.detach()))
                else:
                    scheduler.step()
            report["optimizer"] = config["training"]["optimizer"]
            report["scheduler_checked"] = True
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
            model = inputs = labels = optimizer = loss = logits = paired_labels = scheduler = None
            gc.collect()
            torch.cuda.empty_cache()
    write_json(directory / "gpu_check.json", report)
    print("GPU preflight:", report["status"], flush=True)
    for blocker in report["blockers"]:
        print(blocker, flush=True)
    return report["status"] == "passed"


def run_epoch(model, loader, criterion, optimizer=None, batch_mixing=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs = inputs.to("cuda:0", non_blocking=True)
            labels = labels.to("cuda:0", non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            paired_labels, weight = None, 1.0
            if training:
                inputs, labels, paired_labels, weight = mix_batch(inputs, labels, batch_mixing)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if paired_labels is not None:
                loss = weight * loss + (1 - weight) * criterion(logits, paired_labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if training:
                loss.backward()
                optimizer.step()
            batch = labels.size(0)
            loss_sum += loss.item() * batch
            predictions = logits.argmax(1)
            correct += weight * (predictions == labels).sum().item()
            if paired_labels is not None:
                correct += (1 - weight) * (predictions == paired_labels).sum().item()
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
    model = build_model(checkpoint["config"])
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

    handle = None
    if with_cams:
        cam_layer = getattr(model, "cam_layer", model.head[2])
        handle = cam_layer.register_forward_hook(keep_maps)
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
    with_cams = exported and config["output"].get("compute_cams", True)
    suffixes = ("_predictions.csv", "_outputs.npz") + (("_examples.npz",) if with_cams else ())
    artifacts = [directory / (split + suffix) for suffix in
                 suffixes] if exported else []
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
    indices, targets, logits, cams, base_dataset = collect_outputs(model, loader, with_cams)
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
        "loss_definition": "ordinary hard-label cross entropy; no smoothing or mixed labels",
        "cam_computed": with_cams,
        "cat_to_dog": int(metrics["confusion_matrix"][3][5]),
        "dog_to_cat": int(metrics["confusion_matrix"][5][3]),
    })
    if exported:
        csv_path, raw_path = artifacts[:2]
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
        if with_cams:
            cache = select_examples(base_dataset, indices, targets, details, cams, metrics, config)
            write_npz(artifacts[2], cache)
            metrics["example_selection"] = {
                "seed": config["seed"], "error_groups_disjoint": True,
                "random_errors": "uniform sample without replacement",
                "frequent_confusions": "median-confidence error from each leading directed confusion pair",
                "high_confidence_errors": "highest-confidence errors remaining after other groups",
                "cam_pairs": "correct/error pair from each highest-error true class; median confidence within each group",
                "cam_feature_shape": [config["model"]["head_channels"], *cams.shape[-2:]],
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
    evaluate_test = config["output"].get("evaluate_test", True)
    for split in (("val", "test") if evaluate_test else ("val",)):
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
    evaluated = [("train_eval", train_metrics), ("val", val_metrics)]
    if evaluate_test:
        _, test_loader = test_data(config)
        test_metrics = evaluate_split(model, test_loader, directory, "test", checkpoint, checkpoint_hash, config)
        evaluated.append(("test", test_metrics))
    summary = {
        "best_epoch": checkpoint["epoch"], "best_checkpoint_sha256": checkpoint_hash,
        "evaluation_mode": "same best checkpoint; eval mode; no augmentation; all samples",
        "train_val_accuracy_gap": train_metrics["accuracy"] - val_metrics["accuracy"],
        "val_accuracy_matches_checkpoint": abs(val_metrics["accuracy"] - checkpoint["val_accuracy"]) < 1e-12,
        "test_evaluation_enabled": evaluate_test,
    }
    for split, metrics in evaluated:
        summary[split] = {key: metrics[key] for key in
                          ("samples", "loss", "accuracy", "macro_f1", "ece", "multiclass_brier")}
    write_json(directory / "best_model_metrics.json", summary)
    return test_metrics if evaluate_test else val_metrics


def failure_status(directory, exc):
    status = "oom" if is_oom(exc) else "failed"
    write_json(directory / "status.json", {
        "status": status, "error": str(exc), "traceback": traceback.format_exc(),
        "automatic_retry": False,
    })
    print("Stopped ({}): {}".format(status, exc), file=sys.stderr, flush=True)


def make_optimizer(model, config):
    options = config["training"]
    common = {"lr": options["lr"], "weight_decay": options["weight_decay"], "foreach": False}
    if options["optimizer"] in ("Adam", "AdamW"):
        optimizer = getattr(torch.optim, options["optimizer"])
        return optimizer(model.parameters(), **common, betas=tuple(options.get("betas", (0.9, 0.999))),
                         eps=options.get("eps", 1e-8))
    if options["optimizer"] == "SGD":
        return torch.optim.SGD(model.parameters(), **common, momentum=options.get("momentum", 0.0),
                               nesterov=options.get("nesterov", False), dampening=0)
    raise ValueError("Unsupported optimizer")


def make_scheduler(optimizer, config):
    options = config["training"].get("scheduler")
    if options is None:
        return None
    if options["name"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **{key: value for key, value in options.items() if key != "name"})
    if options["name"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=options["T_max"], eta_min=options["eta_min"])
    raise ValueError("Unsupported learning-rate scheduler: " + str(options["name"]))


def verify_ablation_split(config, split_info):
    if "ablation" not in config:
        return
    reference = Path(config["ablation"]["baseline_run"]) / "split_indices.npz"
    with np.load(reference, allow_pickle=False) as baseline, np.load(
            split_info["path"], allow_pickle=False) as current:
        if set(baseline.files) != set(current.files) or any(
                not np.array_equal(baseline[key], current[key]) for key in baseline.files):
            raise ValueError("Ablation split differs from the existing baseline")


def atomic_torch_save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def rng_snapshot(train_loader, val_loader):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), *state[2:]),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "train_loader": train_loader.generator.get_state(),
            "val_loader": val_loader.generator.get_state()}


def restore_rng(state, train_loader, val_loader):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    train_loader.generator.set_state(state["train_loader"])
    val_loader.generator.set_state(state["val_loader"])


def write_history(directory, history):
    path = directory / "history.csv"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def update_resources(directory, resources, history, completed=False):
    resources.update({"completed_epochs": len(history), "training_completed": completed,
                      "total_training_seconds": sum(row["seconds"] for row in history),
                      "total_training_time_definition": "sum of completed training+validation epoch wall times",
                      "peak_allocated_bytes": max(resources.get("peak_allocated_bytes", 0),
                                                   torch.cuda.max_memory_allocated(0)),
                      "peak_reserved_bytes": max(resources.get("peak_reserved_bytes", 0),
                                                  torch.cuda.max_memory_reserved(0))})
    write_json(directory / "resources.json", resources)


def train_recipe(config, directory):
    from run import validate_config

    directory = Path(directory)
    validate_config(config)
    saved_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if fingerprint(saved_config) != fingerprint(config):
        raise ValueError("Refusing to overwrite a different experiment configuration")
    check = json.loads((directory / "gpu_check.json").read_text(encoding="utf-8"))
    if check["status"] != "passed" or check.get("config_sha256") != fingerprint(config):
        raise ValueError("A matching successful GPU preflight is required")
    history, resources = [], {}
    try:
        seed_everything(42)
        train_loader, val_loader, split_info = training_data(config)
        verify_ablation_split(config, split_info)
        split_path = directory / "split_indices.npz"
        if split_path.exists():
            if hashlib.sha256(split_path.read_bytes()).hexdigest() != split_info["file_sha256"]:
                raise ValueError("The stored split changed")
        else:
            shutil.copyfile(split_info["path"], split_path)
        split_info["source_path"], split_info["path"] = split_info["path"], "split_indices.npz"
        write_json(directory / "split.json", split_info)
        model = build_model(config).to("cuda:0")
        optimizer = make_optimizer(model, config)
        scheduler = make_scheduler(optimizer, config)
        criterion = nn.CrossEntropyLoss(label_smoothing=config["training"].get("label_smoothing", 0.0))
        validation_criterion = nn.CrossEntropyLoss()
        write_json(directory / "training_definition.json", training_definition(config))
        write_json(directory / "optimizer.json", {"name": config["training"]["optimizer"],
                                                   "defaults": optimizer.defaults})
        last_path = directory / "last.pt"
        best_accuracy, best_epoch, start = -1.0, 0, 0
        if last_path.exists():
            last = torch.load(last_path, map_location="cpu", weights_only=True)
            if last["config_sha256"] != fingerprint(config):
                raise ValueError("The resume checkpoint belongs to a different configuration")
            history, resources = last["history"], last["resources"]
            model.load_state_dict(last["model_state_dict"], strict=True)
            optimizer.load_state_dict(last["optimizer_state_dict"])
            if scheduler is not None:
                scheduler.load_state_dict(last["scheduler_state_dict"])
            start, best_epoch, best_accuracy = last["epoch"], last["best_epoch"], last["best_accuracy"]
            if len(history) != start or start > config["training"]["epochs"]:
                raise ValueError("Invalid saved epoch history")
            if start and best_epoch == start:
                repair_best = True
                if (directory / "best.pt").is_file():
                    try:
                        best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
                        repair_best = not (best["epoch"] == best_epoch and best["val_accuracy"] == best_accuracy
                                           and fingerprint(best["config"]) == fingerprint(config))
                        del best
                    except Exception:
                        repair_best = True
                if repair_best:
                    save_checkpoint(directory / "best.pt", model, best_epoch, best_accuracy, config)
            elif start:
                best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
                if (best["epoch"] != best_epoch or best["val_accuracy"] != best_accuracy
                        or fingerprint(best["config"]) != fingerprint(config)):
                    raise ValueError("The saved best checkpoint is inconsistent with last.pt")
                del best
            restore_rng(last["rng"], train_loader, val_loader)
            del last
        else:
            record_reproducibility(directory, config, model, split_info)
        torch.cuda.reset_peak_memory_stats(0)
        wall_started = time.perf_counter()
        prior_wall = resources.get("total_training_wall_seconds", 0.0)

        def persist(epoch):
            resources["total_training_wall_seconds"] = prior_wall + time.perf_counter() - wall_started
            resources["training_wall_time_definition"] = "training and validation plus checkpoint/log I/O; excludes preflight, setup and final evaluation"
            update_resources(directory, resources, history, completed=epoch == config["training"]["epochs"])
            atomic_torch_save(last_path, {
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "config_sha256": fingerprint(config),
                "epoch": epoch, "best_epoch": best_epoch, "best_accuracy": best_accuracy,
                "history": history, "resources": resources, "rng": rng_snapshot(train_loader, val_loader)})

        if not last_path.exists():
            persist(0)
        write_history(directory, history)
        write_json(directory / "status.json", {"status": "training", "resumed_after_epoch": start})
        for epoch in range(start + 1, config["training"]["epochs"] + 1):
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_accuracy, train_count = run_epoch(
                model, train_loader, criterion, optimizer, config["training"].get("batch_mixing"))
            val_loss, val_accuracy, val_count = run_epoch(model, val_loader, validation_criterion)
            torch.cuda.synchronize(0)
            history.append(dict(zip(HISTORY_FIELDS, (epoch, train_loss, train_accuracy, val_loss,
                           val_accuracy, train_count, val_count, time.perf_counter() - started, lr))))
            if val_accuracy > best_accuracy:
                best_accuracy, best_epoch = val_accuracy, epoch
            if scheduler is not None and epoch < config["training"]["epochs"]:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            persist(epoch)
            if best_epoch == epoch:
                save_checkpoint(directory / "best.pt", model, epoch, best_accuracy, config)
            write_history(directory, history)
            print("Epoch {:3d}/{} train {:.4f}/{:.2%} val {:.4f}/{:.2%} lr {:.8g}".format(
                epoch, config["training"]["epochs"], train_loss, train_accuracy, val_loss,
                val_accuracy, lr), flush=True)
        resources["total_training_wall_seconds"] = prior_wall + time.perf_counter() - wall_started
        update_resources(directory, resources, history, completed=True)
        del model, optimizer, scheduler, criterion, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()
        write_json(directory / "status.json", {"status": "evaluating_best"})
        evaluate_best(directory, config)
        output = config["output"]
        if output["plot"]:
            plot_results(directory)
        write_json(directory / "status.json", {
            "status": "completed", "config_sha256": fingerprint(config),
            "epochs_completed": config["training"]["epochs"],
            "best_epoch": best_epoch, "best_val_accuracy": best_accuracy,
            "test_evaluation": "completed" if output["evaluate_test"] else "disabled",
            "cam": "completed" if output["compute_cams"] else "disabled",
            "plotting": "completed" if output["plot"] else "disabled"})
        return 0
    except KeyboardInterrupt:
        write_json(directory / "status.json", {"status": "interrupted", "reason": "User interruption",
                                               "resume": "last completed epoch from last.pt"})
        raise
    except Exception as exc:
        failure_status(directory, exc)
        try:
            update_resources(directory, resources, history, completed=len(history) == config["training"]["epochs"])
        except Exception:
            pass
        return 1


def train(config, run_name):
    if is_resumable(config):
        directory = new_directory(config["output"]["results_root"], run_name)
        write_json(directory / "config.json", config)
        if not gpu_preflight(config, directory):
            write_json(directory / "status.json", {"status": "preflight_failed"})
            return 2
        return train_recipe(config, directory)
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
        verify_ablation_split(config, split_info)
        shutil.copyfile(split_info["path"], directory / "split_indices.npz")
        split_info["source_path"] = split_info["path"]
        split_info["path"] = "split_indices.npz"
        write_json(directory / "split.json", split_info)
        model = build_model(config).to("cuda:0")
        record_reproducibility(directory, config, model, split_info)
        optimizer = make_optimizer(model, config)
        scheduler = make_scheduler(optimizer, config)
        criterion = nn.CrossEntropyLoss()
        train_criterion = nn.CrossEntropyLoss(
            label_smoothing=config["training"].get("label_smoothing", 0.0))
        best_accuracy = -1.0
        write_json(directory / "status.json", {"status": "training"})
        with (directory / "history.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            stream.flush()
            for epoch in range(1, config["training"]["epochs"] + 1):
                started = time.perf_counter()
                train_loss, train_acc, train_count = run_epoch(model, train_loader, train_criterion, optimizer)
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
                if scheduler is not None and epoch < config["training"]["epochs"]:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
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
    if not config["output"].get("plot", True):
        write_json(directory / "status.json", {"status": "completed", "plotting_skipped": True})
        return 0
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
    if config["output"].get("plot", True):
        plot_results(directory)
    write_json(directory / "status.json", {"status": "completed"})
    return 0
