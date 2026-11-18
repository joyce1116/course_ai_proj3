import csv
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data import CLASSES


def plot_results(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    directory = Path(directory).resolve()
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    history = []
    if (directory / "history.csv").exists():
        with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
    metrics_path = directory / "test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    if not history and metrics is None:
        raise ValueError("No training history or test metrics exist; nothing to plot")
    plot_name = "plot_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    figures = Path(config["output"]["figures_root"]) / directory.name / plot_name
    figures.mkdir(parents=True, exist_ok=False)
    dpi = config["output"]["figure_dpi"]

    def save(figure, filename):
        figure.savefig(figures / filename, dpi=dpi, bbox_inches="tight")
        plt.close(figure)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    if history:
        epochs = [int(row["epoch"]) for row in history]
        for metric, title, filename in (("loss", "Cross-entropy loss", "loss.png"),
                                         ("accuracy", "Accuracy", "accuracy.png")):
            figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            axis.plot(epochs, [float(row["train_" + metric]) for row in history], label="Train")
            axis.plot(epochs, [float(row["val_" + metric]) for row in history], label="Validation")
            axis.set(xlabel="Epoch", ylabel=title, title="CifarCNN: " + title)
            if metric == "accuracy":
                axis.yaxis.set_major_formatter(PercentFormatter(1.0))
                axis.set_ylim(0, 1)
            axis.grid(alpha=0.25)
            axis.legend()
            save(figure, filename)
    if metrics:
        confusion = np.asarray(metrics["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
        chart = axis.imshow(confusion, cmap="Blues")
        figure.colorbar(chart, ax=axis, label="Number of images", shrink=0.8)
        axis.set(xticks=np.arange(10), yticks=np.arange(10), xticklabels=CLASSES,
                 yticklabels=CLASSES, xlabel="Predicted class", ylabel="True class",
                 title="CIFAR-10 test confusion matrix")
        plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for true in range(10):
            for predicted in range(10):
                value = confusion[true, predicted]
                axis.text(predicted, true, str(value), ha="center", va="center", fontsize=8,
                          color="white" if value > confusion.max() / 2 else "black")
        save(figure, "confusion_matrix.png")
        accuracies = [metrics["per_class_accuracy"][label] for label in CLASSES]
        figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        bars = axis.bar(CLASSES, accuracies, color="#4477AA")
        axis.axhline(metrics["test_accuracy"], color="#CC6677", linestyle="--")
        for bar, value in zip(bars, accuracies):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                      "{:.1%}".format(value), ha="center", fontsize=9)
        axis.set(ylabel="Accuracy", ylim=(0, 1.12),
                 title="CIFAR-10 test accuracy by class\nOverall accuracy: {:.2%}".format(metrics["test_accuracy"]))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
        save(figure, "per_class_accuracy.png")
        sample_path = directory / "misclassified_samples.npz"
        if sample_path.exists():
            with np.load(sample_path, allow_pickle=False) as samples:
                count = len(samples["sample_index"])
                columns = min(4, max(1, count))
                rows = max(1, math.ceil(count / columns))
                figure, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.2 * rows),
                                            squeeze=False, constrained_layout=True)
                for index, axis in enumerate(axes.flat):
                    axis.set_axis_off()
                    if index < count:
                        axis.imshow(samples["images"][index], interpolation="nearest")
                        axis.set_title("#{}  True: {}\nPred: {} ({:.1%})".format(
                            samples["sample_index"][index], CLASSES[samples["true_label"][index]],
                            CLASSES[samples["predicted_label"][index]], samples["confidence"][index]),
                            fontsize=10)
                if count == 0:
                    axes[0, 0].text(0.5, 0.5, "No misclassified test images", ha="center", va="center")
                figure.suptitle("Misclassified test samples")
                save(figure, "misclassified.png")
    print("Figures:", figures, flush=True)
    return figures
