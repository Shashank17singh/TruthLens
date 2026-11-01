"""Paper figures for every modality.

Colours were run through the colour-blindness validator: blue for the main
series and orange for the second. Text stays in neutral ink.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
BLUE_SCALE = LinearSegmentedColormap.from_list("truthlens_blue", ["#eef4fc", "#2a78d6", "#123a6b"])
LINE_WIDTH = 2.0

plt.rcParams.update({
    "figure.dpi": 200,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "font.family": "DejaVu Sans",
})


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def roc_figure(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    """`curves` maps a legend label to (labels, probabilities)."""
    figure, axis = plt.subplots(figsize=(3.4, 3.2))
    axis.plot([0, 1], [0, 1], color=GRID, linewidth=1.0, linestyle="--")
    for (name, (labels, probabilities)), colour in zip(curves.items(), (BLUE, ORANGE)):
        false_positive, true_positive, _ = roc_curve(labels, probabilities)
        axis.plot(false_positive, true_positive, color=colour, linewidth=LINE_WIDTH, label=name)
    axis.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02), title="ROC curve")
    axis.legend(loc="lower right")
    _save(figure, path)


def confusion_figure(matrix: list[list[int]], path: Path) -> None:
    values = np.array(matrix)
    figure, axis = plt.subplots(figsize=(2.9, 2.6))
    axis.imshow(values, cmap=BLUE_SCALE, vmin=0)
    axis.grid(False)
    axis.set_xticks([0, 1], ["Real", "Fake"])
    axis.set_yticks([0, 1], ["Real", "Fake"])
    axis.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix")
    midpoint = values.max() / 2
    for row in range(2):
        for column in range(2):
            ink = "white" if values[row, column] > midpoint else INK
            axis.text(column, row, f"{values[row, column]:,}", ha="center", va="center", color=ink, fontsize=9)
    _save(figure, path)


def reliability_figure(before: list[dict], after: list[dict], ece_before: float, ece_after: float, path: Path) -> None:
    """Sample counts get their own panel, so the figure never needs a second y-axis."""
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(3.4, 4.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    top.plot([0, 1], [0, 1], color=GRID, linewidth=1.0, linestyle="--")
    for rows, colour, marker, name in (
        (before, ORANGE, "o", f"Uncalibrated, ECE {ece_before:.3f}"),
        (after, BLUE, "s", f"Calibrated, ECE {ece_after:.3f}"),
    ):
        filled = [row for row in rows if row["count"]]
        top.plot(
            [row["confidence"] for row in filled], [row["accuracy"] for row in filled],
            color=colour, linewidth=LINE_WIDTH, marker=marker, markersize=4, label=name,
        )
    top.set(ylabel="Observed fake rate", ylim=(0, 1.02))
    top.set_title("Reliability diagram", pad=38)
    # Above the plot, because calibration curves can pass through any corner of it.
    top.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0))

    centres = [(row["low"] + row["high"]) / 2 for row in after]
    bottom.bar(centres, [row["count"] for row in after], width=0.055, color=BLUE)
    bottom.set(xlabel="Predicted probability", ylabel="Samples", xlim=(0, 1))
    bottom.grid(axis="x", visible=False)
    _save(figure, path)


def precision_tradeoff_figure(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float, target: float, path: Path
) -> None:
    thresholds = np.linspace(0.0, 1.0, 201)
    predictions = probabilities[None, :] >= thresholds[:, None]
    positives = labels.astype(bool)
    true_positive = (predictions & positives).sum(axis=1)
    predicted = predictions.sum(axis=1)
    keep = predicted > 0
    precision = true_positive[keep] / predicted[keep]
    recall = true_positive[keep] / max(1, positives.sum())

    figure, axis = plt.subplots(figsize=(3.4, 3.0))
    axis.plot(thresholds[keep], precision, color=BLUE, linewidth=LINE_WIDTH, label="Precision")
    axis.plot(thresholds[keep], recall, color=ORANGE, linewidth=LINE_WIDTH, label="Recall")
    axis.axvline(threshold, color=MUTED, linewidth=1.0, linestyle="--")
    axis.annotate(
        f"{target:.0%} precision\nthreshold {threshold:.2f}", (threshold, 0.08),
        xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=7, color=MUTED,
    )
    axis.set(xlabel="Decision threshold", ylabel="Validation score", xlim=(0, 1), ylim=(0, 1.02), title="Threshold trade-off")
    axis.legend(loc="lower left", bbox_to_anchor=(0.0, 0.2))
    _save(figure, path)


def generator_figure(rows: list[dict], unseen: set[str], path: Path) -> None:
    """Horizontal bars, sorted. Unseen generators differ by hatch as well as colour."""
    rows = sorted(rows, key=lambda row: row["accuracy"])
    figure, axis = plt.subplots(figsize=(4.2, max(2.0, 0.3 * len(rows) + 1.2)))
    positions = np.arange(len(rows))
    is_unseen = [row["generator_name"] in unseen for row in rows]
    bars = axis.barh(
        positions,
        [row["accuracy"] for row in rows],
        height=0.62,
        color=[ORANGE if flag else BLUE for flag in is_unseen],
        edgecolor=SURFACE,
        linewidth=0.0,
        hatch=["///" if flag else "" for flag in is_unseen],
    )
    for bar, row in zip(bars, rows):
        axis.text(row["accuracy"] + 0.012, bar.get_y() + bar.get_height() / 2, f"{row['accuracy']:.1%}",
                  va="center", fontsize=7, color=MUTED)
    axis.set_yticks(positions, [row["generator_name"] for row in rows])
    axis.set(xlim=(0, 1.13), xlabel="Accuracy at the chosen threshold")
    axis.set_title("Accuracy by generator", pad=20 if unseen else 6)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.grid(axis="y", visible=False)
    if unseen:
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=BLUE),
            plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE, hatch="///", edgecolor=SURFACE, linewidth=0.0),
        ]
        axis.legend(handles, ["Seen in training", "Unseen generator"], loc="lower center",
                    bbox_to_anchor=(0.5, 1.0), ncol=2)
    _save(figure, path)
