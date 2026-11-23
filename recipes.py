import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FINAL_CONFIG = ROOT / "config" / "final.json"
COMPLETE_MODEL = "complete_model"
ADDITIONAL_PROTOCOL = "additional-experiments-v1"
LEGACY_PROTOCOL = "n4-n28-v1"
ERASING = {"p": 0.25, "scale": [0.02, 0.15], "ratio": [0.3, 3.3], "value": 0, "inplace": False}
COSINE = {"name": "cosine", "T_max": 99, "eta_min": 0.00001}
BASE_MODEL = {"channels": [64, 128, 256, 384], "depths": [2, 2, 3, 3],
              "head_channels": 128, "dropout": 0.3, "num_classes": 10}
BASE_TRAINING = {"epochs": 100, "batch_size": 128, "optimizer": "Adam", "lr": 0.001,
                 "weight_decay": 0.0001, "scheduler": COSINE}
BASE_DATA = {"split_seed": 42, "val_size": 5000, "mean": [0.5] * 3, "std": [0.5] * 3,
             "crop_padding": 4, "horizontal_flip_probability": 0.5}
ADDITIONAL_EXPERIMENTS = {
    "fixed_learning_rate": {},
    "without_randaugment": {},
    "multiscale_fusion": {},
    "spatial_attention": {},
    "multiscale_fusion_spatial_attention": {},
    "random_erasing_025": {"data": {"random_erasing": ERASING}},
    "label_smoothing_010": {"training": {"label_smoothing": 0.10}},
    "head_se": {},
    "random_erasing_head_se": {"data": {"random_erasing": ERASING}, "structure": {"se": "head"}},
    "label_smoothing_005": {"training": {"label_smoothing": 0.05}},
    "random_erasing_label_smoothing": {
        "data": {"random_erasing": ERASING}, "training": {"label_smoothing": 0.05}},
    "head_se_label_smoothing": {"structure": {"se": "head"}, "training": {"label_smoothing": 0.05}},
    "random_erasing_head_se_label_smoothing": {
        "data": {"random_erasing": ERASING}, "structure": {"se": "head"},
        "training": {"label_smoothing": 0.05}},
    "random_erasing_010": {"data": {"random_erasing": {**ERASING, "p": 0.10}}},
    "random_erasing_050": {"data": {"random_erasing": {**ERASING, "p": 0.50}}},
    "label_smoothing_002": {"training": {"label_smoothing": 0.02}},
    "label_smoothing_020": {"training": {"label_smoothing": 0.20}},
    "mixup": {"training": {"batch_mixing": {"name": "mixup", "alpha": 0.2, "probability": 1.0}}},
    "cutmix": {"training": {"batch_mixing": {"name": "cutmix", "alpha": 1.0, "probability": 0.5}}},
    "color_jitter": {"data": {"color_jitter": {"brightness": 0.2, "contrast": 0.2,
                                               "saturation": 0.2, "hue": 0.05}}},
    "head_se_spatial_attention": {"structure": {"se": "head", "spatial_attention": True}},
    "stage2_se": {"structure": {"se": "stage2"}},
    "global_max_pooling": {"structure": {"global_pool": "max"}},
    "global_average_max_pooling": {"structure": {"global_pool": "avgmax"}},
    "average_downsampling": {"structure": {"downsampling": "avg"}},
    "stage_residual": {"structure": {"residual": True}},
    "dropout_0": {"model": {"dropout": 0.0}},
    "dropout_05": {"model": {"dropout": 0.5}},
    "adamw_1e4": {"training": {"optimizer": "AdamW"}},
    "adamw_1e2": {"training": {"optimizer": "AdamW", "weight_decay": 0.01}},
    "momentum_sgd": {"training": {"optimizer": "SGD", "momentum": 0.9, "nesterov": False,
                                  "lr": 0.05, "scheduler": {**COSINE, "eta_min": 0.0005}}},
    "validation_loss_scheduler": {
        "training": {"scheduler": {"name": "plateau", "mode": "min", "factor": 0.5,
                                   "patience": 5, "threshold": 0.0001, "threshold_mode": "abs",
                                   "cooldown": 0, "min_lr": 0.00001}}},
}

# Saved results and checkpoints retain their original identities and fingerprints.
ARCHIVE_NAMES = {
    "basic": "fixed_learning_rate", "E1": "without_randaugment", "E2": "multiscale_fusion",
    "E3": "spatial_attention", "E4": "multiscale_fusion_spatial_attention",
    "N1": "random_erasing_025", "N2": "label_smoothing_010", "N3": "head_se",
    "N4": "random_erasing_head_se", "N5": "label_smoothing_005",
    "N6": "random_erasing_label_smoothing", "N7": "head_se_label_smoothing",
    "N8": "random_erasing_head_se_label_smoothing", "N9": "random_erasing_010",
    "N10": "random_erasing_050", "N11": "label_smoothing_002", "N12": "label_smoothing_020",
    "N13": "mixup", "N14": "cutmix", "N15": COMPLETE_MODEL, "N16": "color_jitter",
    "N17": "head_se_spatial_attention", "N18": "stage2_se", "N19": "global_max_pooling",
    "N20": "global_average_max_pooling", "N21": "average_downsampling", "N22": "stage_residual",
    "N23": "dropout_0", "N24": "dropout_05", "N25": "adamw_1e4", "N26": "adamw_1e2",
    "N27": "momentum_sgd", "N28": "validation_loss_scheduler",
}
TESTED_COMPARISONS = frozenset({
    "fixed_learning_rate", "without_randaugment", "multiscale_fusion", "spatial_attention",
    "multiscale_fusion_spatial_attention", "random_erasing_025", "label_smoothing_010", "head_se",
})
NO_COMPARISON_PLOTS = frozenset({"random_erasing_025", "label_smoothing_010", "head_se"})
RAND_AUGMENT = {"num_ops": 2, "magnitude": 9, "interpolation": "bilinear", "fill": 128}


def experiment_name(config):
    if "experiment" in config:
        return config["experiment"]
    archived = config.get("ablation", {}).get("experiment", "basic")
    return ARCHIVE_NAMES.get(archived, archived)


def is_resumable(config):
    return config.get("protocol") in (ADDITIONAL_PROTOCOL, LEGACY_PROTOCOL)


def uses_recipe_model(config):
    name = experiment_name(config)
    return ((name in ADDITIONAL_EXPERIMENTS and name not in TESTED_COMPARISONS)
            or (name == COMPLETE_MODEL and "ablation" in config))


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def validate_reference(config):
    if config.get("seed") != 42 or experiment_name(config) != "without_randaugment":
        raise ValueError("The reference must be the model without RandAugment with seed 42")
    if config["model"] != BASE_MODEL:
        raise ValueError("The reference architecture differs from the complete model")
    expected = {**BASE_TRAINING, "num_workers": config["training"]["num_workers"]}
    if config["training"] != expected:
        raise ValueError("The reference training settings differ from the complete model")
    data = {key: value for key, value in config["data"].items()
            if key not in ("root", "download", "download_urls")}
    if data != BASE_DATA:
        raise ValueError("The reference must use the shared split, normalization and basic augmentation")


def make_additional_config(name, results_root, figures_root, *, data_root=None, workers=None,
                           no_download=False, gpu=0):
    if name not in ADDITIONAL_EXPERIMENTS:
        raise ValueError("Unknown experiment: " + name)
    config = json.loads(FINAL_CONFIG.read_text(encoding="utf-8"))
    config.update(experiment=name, gpu=gpu, protocol=ADDITIONAL_PROTOCOL)
    config["data"].pop("randaugment")
    config["structure"] = {}
    if name == "fixed_learning_rate":
        config["training"].pop("scheduler")
    for section, changes in ADDITIONAL_EXPERIMENTS[name].items():
        config[section].update(copy.deepcopy(changes))
    config["data"]["root"] = str(Path(data_root or ROOT / config["data"]["root"]).resolve())
    if workers is not None:
        config["training"]["num_workers"] = workers
    if no_download:
        config["data"]["download"] = False
    evaluate_test = name in TESTED_COMPARISONS
    config["output"].update({"results_root": str(Path(results_root).resolve()),
                             "figures_root": str(Path(figures_root).resolve()),
                             "plot": evaluate_test and name not in NO_COMPARISON_PLOTS,
                             "evaluate_test": evaluate_test, "compute_cams": evaluate_test})
    return config


def validate_recipe(config):
    name = experiment_name(config)
    if config.get("protocol") == LEGACY_PROTOCOL and name == COMPLETE_MODEL:
        recipe = {"data": {"randaugment": RAND_AUGMENT}}
    elif is_resumable(config) and name in ADDITIONAL_EXPERIMENTS:
        recipe = ADDITIONAL_EXPERIMENTS[name]
    else:
        raise ValueError("Unknown recipe protocol or experiment")
    expected_model = {**BASE_MODEL, **recipe.get("model", {})}
    expected_training = {**BASE_TRAINING, "num_workers": config["training"]["num_workers"],
                         **recipe.get("training", {})}
    if name == "fixed_learning_rate":
        expected_training.pop("scheduler")
    expected_data = {**BASE_DATA, **recipe.get("data", {})}
    actual_data = {key: value for key, value in config["data"].items()
                   if key not in ("root", "download", "download_urls")}
    if config["seed"] != 42 or config["model"] != expected_model:
        raise ValueError(name + " must preserve its prescribed seed and model")
    if config["training"] != expected_training or actual_data != expected_data:
        raise ValueError(name + " contains missing or mixed training/data changes")
    if config.get("structure") != recipe.get("structure", {}):
        raise ValueError(name + " contains missing or mixed structure changes")
    expected_output = name in TESTED_COMPARISONS and config.get("protocol") == ADDITIONAL_PROTOCOL
    if (config["output"].get("evaluate_test") is not expected_output
            or config["output"].get("compute_cams") is not expected_output
            or config["output"].get("plot") is not (expected_output and name not in NO_COMPARISON_PLOTS)):
        raise ValueError("Unexpected evaluation or plotting settings for " + name)


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
