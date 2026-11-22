import copy
import hashlib
import json
from pathlib import Path


PROTOCOL = "n4-n28-v1"
ERASING = {"p": 0.25, "scale": [0.02, 0.15], "ratio": [0.3, 3.3], "value": 0, "inplace": False}
COSINE = {"name": "cosine", "T_max": 99, "eta_min": 0.00001}
BASE_MODEL = {"channels": [64, 128, 256, 384], "depths": [2, 2, 3, 3],
              "head_channels": 128, "dropout": 0.3, "num_classes": 10}
BASE_TRAINING = {"epochs": 100, "batch_size": 128, "optimizer": "Adam", "lr": 0.001,
                 "weight_decay": 0.0001, "scheduler": COSINE}
BASE_DATA = {"split_seed": 42, "val_size": 5000, "mean": [0.5] * 3, "std": [0.5] * 3,
             "crop_padding": 4, "horizontal_flip_probability": 0.5}
RECIPES = {
    "N4": {"data": {"random_erasing": ERASING}, "structure": {"se": "head"}},
    "N5": {"training": {"label_smoothing": 0.05}},
    "N6": {"data": {"random_erasing": ERASING}, "training": {"label_smoothing": 0.05}},
    "N7": {"structure": {"se": "head"}, "training": {"label_smoothing": 0.05}},
    "N8": {"data": {"random_erasing": ERASING}, "structure": {"se": "head"},
           "training": {"label_smoothing": 0.05}},
    "N9": {"data": {"random_erasing": {**ERASING, "p": 0.10}}},
    "N10": {"data": {"random_erasing": {**ERASING, "p": 0.50}}},
    "N11": {"training": {"label_smoothing": 0.02}},
    "N12": {"training": {"label_smoothing": 0.20}},
    "N13": {"training": {"batch_mixing": {"name": "mixup", "alpha": 0.2, "probability": 1.0}}},
    "N14": {"training": {"batch_mixing": {"name": "cutmix", "alpha": 1.0, "probability": 0.5}}},
    "N15": {"data": {"randaugment": {"num_ops": 2, "magnitude": 9,
                                      "interpolation": "bilinear", "fill": 128}}},
    "N16": {"data": {"color_jitter": {"brightness": 0.2, "contrast": 0.2,
                                       "saturation": 0.2, "hue": 0.05}}},
    "N17": {"structure": {"se": "head", "spatial_attention": True}},
    "N18": {"structure": {"se": "stage2"}},
    "N19": {"structure": {"global_pool": "max"}},
    "N20": {"structure": {"global_pool": "avgmax"}},
    "N21": {"structure": {"downsampling": "avg"}},
    "N22": {"structure": {"residual": True}},
    "N23": {"model": {"dropout": 0.0}},
    "N24": {"model": {"dropout": 0.5}},
    "N25": {"training": {"optimizer": "AdamW"}},
    "N26": {"training": {"optimizer": "AdamW", "weight_decay": 0.01}},
    "N27": {"training": {"optimizer": "SGD", "momentum": 0.9, "nesterov": False,
                          "lr": 0.05, "scheduler": {**COSINE, "eta_min": 0.0005}}},
    "N28": {"training": {"scheduler": {"name": "plateau", "mode": "min", "factor": 0.5,
                          "patience": 5, "threshold": 0.0001, "threshold_mode": "abs",
                          "cooldown": 0, "min_lr": 0.00001}}},
}
COMPARISONS = {"N4": ["E1", "N1", "N3"], "N6": ["E1", "N1", "N5"],
               "N7": ["E1", "N3", "N5"], "N8": ["E1", "N1", "N3", "N5", "N4", "N6", "N7"],
               "N17": ["E1", "N3", "E3"]}


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def validate_e1(config):
    if config.get("seed") != 42 or config.get("ablation", {}).get("experiment") != "E1":
        raise ValueError("The reference must be the original E1 with seed 42")
    if config["model"] != BASE_MODEL:
        raise ValueError("The reference model differs from E1")
    expected = {**BASE_TRAINING, "num_workers": config["training"]["num_workers"]}
    if config["training"] != expected:
        raise ValueError("The reference training settings differ from E1")
    data = {key: value for key, value in config["data"].items()
            if key not in ("root", "download", "download_urls")}
    if data != BASE_DATA:
        raise ValueError("The reference split, normalization or augmentation differs from E1")


def make_config(base, name, reference, suite, data_root=None, workers=None, no_download=False):
    validate_e1(base)
    if name not in RECIPES:
        raise ValueError("Unknown experiment: " + name)
    config = copy.deepcopy(base)
    config["protocol"] = PROTOCOL
    config["seed"], config["gpu"] = 42, 0
    config["ablation"] = {"experiment": name, "baseline_run": str(Path(reference).resolve())}
    config["structure"] = {}
    for section, changes in RECIPES[name].items():
        config[section].update(copy.deepcopy(changes))
    if data_root is not None:
        config["data"]["root"] = str(Path(data_root).resolve())
    if workers is not None:
        config["training"]["num_workers"] = workers
    if no_download:
        config["data"]["download"] = False
    config["output"].update({"results_root": str(Path(suite).resolve()), "plot": False,
                              "evaluate_test": False, "compute_cams": False})
    config["output"]["figures_root"] = str(Path(suite).resolve() / "unused_figures")
    validate_recipe(config)
    return config


def validate_recipe(config):
    name = config.get("ablation", {}).get("experiment")
    if config.get("protocol") != PROTOCOL or name not in RECIPES:
        raise ValueError("Unknown recipe protocol or experiment")
    expected_model = {**BASE_MODEL, **RECIPES[name].get("model", {})}
    expected_training = {**BASE_TRAINING, "num_workers": config["training"]["num_workers"],
                         **RECIPES[name].get("training", {})}
    expected_data = {**BASE_DATA, **RECIPES[name].get("data", {})}
    actual_data = {key: value for key, value in config["data"].items()
                   if key not in ("root", "download", "download_urls")}
    if config["seed"] != 42 or config["gpu"] != 0 or config["model"] != expected_model:
        raise ValueError(name + " must preserve its prescribed seed, GPU and model")
    if config["training"] != expected_training or actual_data != expected_data:
        raise ValueError(name + " contains missing or mixed training/data changes")
    if config.get("structure") != RECIPES[name].get("structure", {}):
        raise ValueError(name + " contains missing or mixed structure changes")
    if any(config["output"].get(key) is not False
           for key in ("plot", "evaluate_test", "compute_cams")):
        raise ValueError("Recipes disable plots, test evaluation and CAM")
    workers = config["training"]["num_workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")


def training_definition(config):
    options = config["training"]
    mixing = options.get("batch_mixing")
    smoothing = options.get("label_smoothing", 0.0)
    return {
        "training_loss": ("lambda * CE(logits, y_a) + (1-lambda) * CE(logits, y_b)"
                          if mixing else "cross entropy"),
        "label_smoothing": smoothing,
        "batch_mixing": mixing,
        "training_accuracy": ("mean(lambda * 1[pred=y_a] + (1-lambda) * 1[pred=y_b]); "
                              "lambda is corrected for actual clipped CutMix area"
                              if mixing else "mean(1[pred=true hard label])"),
        "evaluation_loss": "ordinary cross entropy with true hard labels; no smoothing or mixing",
        "evaluation_accuracy": "mean(1[pred=true hard label])",
        "early_stopping": False, "training_budget_epochs": 100,
        "selection": "strictly highest validation accuracy; first epoch wins ties",
    }
