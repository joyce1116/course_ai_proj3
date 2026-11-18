import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data import CLASSES


def plot_results(directory, figures_root=None, split="test"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    directory = Path(directory).resolve()
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    history = []
    if (directory / "history.csv").exists():
        with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
    path = directory / (split + "_metrics.json")
    metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if not history and metrics is None:
        raise ValueError("No training history or requested split metrics exist; nothing to plot")
    root = Path(figures_root) if figures_root is not None else Path(config["output"]["figures_root"])
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    plot_name = "plot_" + split + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    figures = root / directory.name / (plot_name + "_" + uuid.uuid4().hex[:8])
    figures.mkdir(parents=True, exist_ok=False)
    dpi = config["output"]["figure_dpi"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(figure, filename):
        figure.savefig(figures / filename, dpi=dpi, bbox_inches="tight")
        plt.close(figure)

    if history:
        best = max(history, key=lambda row: float(row["val_accuracy"]))
        best_epoch = int(best["epoch"])
        epochs = [int(row["epoch"]) for row in history]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for axis, metric, title in zip(axes, ("loss", "accuracy"), ("Cross-entropy loss", "Accuracy")):
            axis.plot(epochs, [float(row["train_" + metric]) for row in history],
                      label="Train (augmented, train mode)")
            axis.plot(epochs, [float(row["val_" + metric]) for row in history], label="Validation (eval)")
            axis.axvline(best_epoch, color="#777777", linestyle="--", linewidth=1,
                         label="Best validation accuracy: epoch {}".format(best_epoch))
            axis.scatter([best_epoch], [float(best["val_" + metric])], color="#CC6677", zorder=5)
            axis.set(xlabel="Epoch", ylabel=title, title=title)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=0.2)
            if metric == "accuracy":
                axis.yaxis.set_major_formatter(PercentFormatter(1.0))
                axis.set_ylim(0, 1)
        axes[0].legend(fontsize=8)
        figure.suptitle("CifarCNN training and validation")
        save(figure, "learning_curves.png")

    if metrics is None:
        print("Split metrics are not saved yet; only learning curves were produced:", split, flush=True)
        print("Figures:", figures, flush=True)
        return figures
    accuracy = metrics.get("accuracy", metrics.get(split + "_accuracy"))
    confusion = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    chart = axis.imshow(confusion, cmap="Blues")
    figure.colorbar(chart, ax=axis, label="Image count", shrink=0.8)
    axis.set(xticks=np.arange(10), yticks=np.arange(10), xticklabels=CLASSES,
             yticklabels=CLASSES, xlabel="Predicted class", ylabel="True class",
             title="{} confusion matrix (accuracy: {:.2%})".format(split.capitalize(), accuracy))
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for true in range(10):
        for predicted in range(10):
            value = confusion[true, predicted]
            axis.text(predicted, true, str(value), ha="center", va="center", fontsize=8,
                      color="white" if value > confusion.max() / 2 else "black")
    save(figure, "confusion_matrix.png")

    diagonal = confusion.diagonal()
    support, predicted_count = confusion.sum(axis=1), confusion.sum(axis=0)
    precision = np.divide(diagonal, predicted_count, out=np.zeros(10), where=predicted_count > 0)
    recall = np.divide(diagonal, support, out=np.zeros(10), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(10), where=precision + recall > 0)
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    positions = np.arange(10)
    axis.bar(positions - 0.25, precision, 0.24, label="Precision", color="#4477AA")
    bars = axis.bar(positions, recall, 0.24, label="Recall (= class accuracy)", color="#228833")
    axis.bar(positions + 0.25, f1, 0.24, label="F1", color="#CCBB44")
    for bar, value in zip(bars, recall):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.018, "{:.1%}".format(value),
                  ha="center", fontsize=8, rotation=45)
    axis.set(xticks=positions, xticklabels=CLASSES, ylabel="Score", ylim=(0, 1.17),
             title="{} class performance (macro F1: {:.3f})".format(split.capitalize(), f1.mean()))
    axis.axhline(accuracy, color="#AA3377", linestyle="--", linewidth=1)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3, fontsize=8)
    save(figure, "class_performance.png")

    predictions_path = directory / (split + "_predictions.csv")
    if predictions_path.exists():
        with predictions_path.open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        confidence = np.asarray([float(row["confidence"]) for row in predictions])
        correct = np.asarray([row["true_label"] == row["predicted_label"] for row in predictions])
        bins = config["output"].get("calibration_bins", 10)
        calibration = metrics.get("calibration")
        if calibration is None:
            calibration = []
            bin_index = np.minimum((confidence * bins).astype(int), bins - 1)
            for index in range(bins):
                selected = bin_index == index
                if selected.any():
                    calibration.append({"count": int(selected.sum()),
                                        "confidence": float(confidence[selected].mean()),
                                        "accuracy": float(correct[selected].mean())})
        populated = [entry for entry in calibration if entry["count"]]
        ece = sum(entry["count"] / len(confidence) * abs(entry["accuracy"] - entry["confidence"])
                  for entry in populated)
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].plot([0, 1], [0, 1], color="#888888", linestyle="--", label="Perfect calibration")
        axes[0].plot([entry["confidence"] for entry in populated],
                     [entry["accuracy"] for entry in populated], "o-", color="#4477AA", label="Observed")
        axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted confidence",
                    ylabel="Observed accuracy", title="Reliability (ECE: {:.2%})".format(ece))
        axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
        axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[0].legend(fontsize=8)
        for selected, label, color in ((correct, "Correct", "#4477AA"), (~correct, "Wrong", "#CC6677")):
            count = int(selected.sum())
            if count:
                axes[1].hist(confidence[selected], bins=np.linspace(0, 1, bins + 1),
                             weights=np.full(count, 1 / count), histtype="step", linewidth=2,
                             color=color, label="{} (n={})".format(label, count))
        axes[1].set(xlim=(0, 1), xlabel="Predicted confidence", ylabel="Fraction within each group",
                    title="Correct / wrong confidence distribution")
        axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
        axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[1].legend(fontsize=8)
        figure.suptitle(split.capitalize() + " confidence diagnostics")
        save(figure, "confidence_diagnostics.png")

    examples_path = directory / (split + "_examples.npz")
    legacy = False
    if not examples_path.exists() and split == "test":
        examples_path = directory / "misclassified_samples.npz"
        legacy = True
    if examples_path.exists():
        with np.load(examples_path, allow_pickle=False) as saved:
            samples = {key: saved[key].copy() for key in saved.files}
        per_group = config["output"].get("error_examples_per_group", 4)
        if legacy:
            error_indices = np.arange(len(samples["sample_index"]))
            groups = np.full(len(error_indices), "High-confidence errors")
        else:
            error_indices, groups = samples["error_cache_index"], samples["error_group"]
        group_names = list(dict.fromkeys(groups.tolist()))
        plot_rows = []
        for name in group_names:
            chosen = error_indices[groups == name]
            plot_rows.extend((name, chosen[start:start + per_group])
                             for start in range(0, len(chosen), per_group))
        figure, axes = plt.subplots(max(1, len(plot_rows)), per_group,
                                    figsize=(2.7 * per_group, 2.9 * max(1, len(plot_rows))),
                                    squeeze=False, constrained_layout=True)
        for axis in axes.flat:
            axis.set_axis_off()
        for row, (name, chosen) in enumerate(plot_rows):
            axes[row, 0].text(-0.1, 0.5, name, rotation=90, va="center", ha="right",
                              transform=axes[row, 0].transAxes, fontsize=10)
            for column, index in enumerate(chosen):
                axis = axes[row, column]
                axis.imshow(samples["images"][index], interpolation="nearest")
                axis.set_title("#{}  True: {}\nPred: {} ({:.3%})".format(
                    samples["sample_index"][index], CLASSES[samples["true_label"][index]],
                    CLASSES[samples["predicted_label"][index]], samples["confidence"][index]), fontsize=9)
        if not plot_rows:
            axes[0, 0].text(0.5, 0.5, "No errors", ha="center", va="center")
        figure.suptitle(split.capitalize() + " error cases (selection shown by row)")
        save(figure, "error_cases.png")

        if not legacy:
            pairs = samples["cam_pairs"]
            figure, axes = plt.subplots(max(1, len(pairs)), 5,
                                        figsize=(12.5, 2.9 * max(1, len(pairs))),
                                        squeeze=False, constrained_layout=True)
            for axis in axes.flat:
                axis.set_axis_off()

            def show_cam(axis, image, values):
                axis.imshow(image, interpolation="nearest")
                positive = np.maximum(values, 0)
                if positive.max() > 0:
                    axis.imshow(positive / positive.max(), cmap="magma", alpha=0.5,
                                vmin=0, vmax=1, interpolation="bilinear",
                                extent=(-0.5, 31.5, 31.5, -0.5))
                else:
                    axis.text(0.5, 0.02, "No positive CAM", ha="center", fontsize=8,
                              color="white", transform=axis.transAxes)

            for row, (good, bad) in enumerate(pairs):
                true_class = CLASSES[samples["true_label"][bad]]
                predicted_class = CLASSES[samples["predicted_label"][bad]]
                axes[row, 0].imshow(samples["images"][good], interpolation="nearest")
                axes[row, 0].set_title("Correct #{}\nTrue/Pred: {}".format(
                    samples["sample_index"][good], true_class), fontsize=9)
                show_cam(axes[row, 1], samples["images"][good], samples["cam_predicted"][good])
                axes[row, 1].set_title("Correct predicted CAM\np={:.3%}".format(
                    samples["confidence"][good]), fontsize=9)
                axes[row, 2].imshow(samples["images"][bad], interpolation="nearest")
                axes[row, 2].set_title("Error #{}\nTrue: {} / Pred: {}".format(
                    samples["sample_index"][bad], true_class, predicted_class), fontsize=9)
                show_cam(axes[row, 3], samples["images"][bad], samples["cam_predicted"][bad])
                axes[row, 3].set_title("Error predicted CAM: {}\np={:.3%}".format(
                    predicted_class, samples["confidence"][bad]), fontsize=9)
                show_cam(axes[row, 4], samples["images"][bad], samples["cam_true"][bad])
                axes[row, 4].set_title("Error true-class CAM\n" + true_class, fontsize=9)
            if not len(pairs):
                axes[0, 0].text(0.5, 0.5, "No correct/error pairs", ha="center", va="center")
            figure.suptitle(split.capitalize() + " CAM (4x4 maps; each map normalized separately)")
            save(figure, "cam_comparison.png")
    print("Figures:", figures, flush=True)
    return figures
