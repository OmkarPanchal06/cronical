"""Figures for the evaluation report, drawn from real predictions.

Every function here takes the actual arrays produced by a fitted model and plots
them. Nothing generates a placeholder or illustrative graph: if there are no
predictions, there is no figure.

The matplotlib backend is forced to ``Agg`` so figures can be written on a
headless machine or in CI without a display.

A note on what a ROC or PR curve does and does not show
-------------------------------------------------------
A ROC curve plots sensitivity against specificity across *every* possible
threshold. That makes it threshold-independent, which is useful for comparing a
model's ranking ability — and useless for choosing an operating point. A curve
that looks excellent tells you nothing about what would happen at any particular
threshold. Selecting a threshold is a separate, later question, and these figures
deliberately do not answer it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402  (named type for annotations)

from cronical.models.errors import EvaluationError  # noqa: E402
from cronical.models.evaluate import (  # noqa: E402
    confusion_counts,
    pr_curve_points,
    roc_curve_points,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "plot_confusion_matrix",
    "plot_precision_recall_curve",
    "plot_roc_curve",
    "save_figure",
]

#: Shared figure geometry, so every plot in the report looks like one family.
_FIGURE_SIZE: Final[tuple[float, float]] = (6.0, 4.5)
_DPI: Final[int] = 150


def save_figure(figure: Figure, path: Path | str) -> Path:
    """Write a figure to disk, creating parent directories.

    Args:
        figure: The figure to write.
        path: Destination file. The parent directory is created if needed.

    Returns:
        The path written.

    Raises:
        EvaluationError: If the figure cannot be written.
    """
    destination = Path(path).expanduser()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=_DPI, bbox_inches="tight")
    except OSError as exc:
        raise EvaluationError(f"could not write figure to {destination}: {exc}") from exc
    finally:
        plt.close(figure)
    return destination


def plot_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    title: str = "Confusion matrix",
    path: Path | str | None = None,
) -> Figure:
    """Plot the four confusion-matrix cells.

    Args:
        y_true: Actual binary labels.
        y_pred: Predicted binary labels.
        title: Figure title. Should identify the model and threshold.
        path: If given, the figure is written here and closed.

    Returns:
        The figure. Returned even when ``path`` is supplied, so a caller can
        inspect it; use :func:`save_figure` to write it later.

    Raises:
        EvaluationError: If the counts cannot be computed.
    """
    counts = confusion_counts(y_true, y_pred)
    matrix = np.array(counts.as_matrix(), dtype=int)

    figure, axes = plt.subplots(figsize=_FIGURE_SIZE)
    image = axes.imshow(matrix, cmap="Blues")

    axes.set_xticks([0, 1], labels=["predicted negative", "predicted positive"])
    axes.set_yticks([0, 1], labels=["actually negative", "actually positive"])
    axes.set_xlabel("Predicted")
    axes.set_ylabel("Actual")
    axes.set_title(title)

    threshold = matrix.max() / 2 if matrix.max() else 0.5
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            axes.text(
                column,
                row,
                f"{value}\n({_share(value, counts.total)})",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.colorbar(image, ax=axes, fraction=0.046)
    figure.tight_layout()

    if path is not None:
        save_figure(figure, path)
    return figure


def _share(value: int, total: int) -> str:
    """Render a count as a percentage of the total, or a dash if undefined."""
    return f"{value / total:.1%}" if total else "n/a"


def plot_roc_curve(
    y_true: Sequence[int] | np.ndarray,
    y_probabilities: Sequence[float] | np.ndarray,
    *,
    auc: float | None = None,
    title: str = "ROC curve",
    path: Path | str | None = None,
) -> Figure:
    """Plot the ROC curve from positive-class probability scores.

    Args:
        y_true: Actual binary labels. Both classes must be present.
        y_probabilities: Positive-class scores.
        auc: The computed ROC-AUC, displayed in the legend when available.
        title: Figure title. Should identify the model.
        path: If given, the figure is written here and closed.

    Returns:
        The figure.

    Raises:
        EvaluationError: If both classes are not present, since a ROC curve is
            undefined in that case.
    """
    false_positive_rate, true_positive_rate, _ = roc_curve_points(y_true, y_probabilities)

    figure, axes = plt.subplots(figsize=_FIGURE_SIZE)
    label = "ROC (AUC = n/a)" if auc is None else f"ROC (AUC = {auc:.3f})"
    axes.plot(false_positive_rate, true_positive_rate, label=label, linewidth=2)
    axes.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="grey", linewidth=1)

    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_title(title)
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.01)
    axes.legend(loc="lower right")
    figure.tight_layout()

    if path is not None:
        save_figure(figure, path)
    return figure


def plot_precision_recall_curve(
    y_true: Sequence[int] | np.ndarray,
    y_probabilities: Sequence[float] | np.ndarray,
    *,
    average_precision: float | None = None,
    positive_rate: float | None = None,
    title: str = "Precision-Recall curve",
    path: Path | str | None = None,
) -> Figure:
    """Plot the precision-recall curve from probability scores.

    When ``positive_rate`` is supplied, a horizontal baseline is drawn at the
    prevalence of the positive class. A curve below that line is doing worse than
    always predicting positive, which is the reference that makes PR curves
    interpretable on imbalanced data.

    Args:
        y_true: Actual binary labels.
        y_probabilities: Positive-class scores.
        average_precision: Computed average precision, shown in the legend.
        positive_rate: Prevalence of the positive class, for the baseline.
        title: Figure title. Should identify the model.
        path: If given, the figure is written here and closed.

    Returns:
        The figure.

    Raises:
        EvaluationError: If the curve cannot be computed.
    """
    recall, precision, _ = pr_curve_points(y_true, y_probabilities)

    figure, axes = plt.subplots(figsize=_FIGURE_SIZE)
    label = "PR (AP = n/a)" if average_precision is None else f"PR (AP = {average_precision:.3f})"
    axes.plot(recall, precision, label=label, linewidth=2)
    if positive_rate is not None:
        axes.axhline(
            positive_rate,
            linestyle="--",
            color="grey",
            linewidth=1,
            label=f"positive rate = {positive_rate:.3f}",
        )
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_title(title)
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.01)
    axes.legend(loc="lower left")
    figure.tight_layout()

    if path is not None:
        save_figure(figure, path)
    return figure