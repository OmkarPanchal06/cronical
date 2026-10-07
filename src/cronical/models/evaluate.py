"""Metrics for binary classification, computed from real predictions.

Scope
-----
This module measures **model behaviour on a dataset**. It does not establish
clinical validity, and no figure it produces says anything about any patient.

Two rules shape the implementation:

**Nothing is invented.** Every number comes from the arrays passed in. There are
no placeholder values, no defaults, and no metric carried over from a previous
run.

**A metric that cannot be computed is omitted, not guessed.** Several metrics are
undefined for particular inputs — precision when nothing is predicted positive,
ROC-AUC when only one class is present. Those are reported as ``None`` and named
in :attr:`Metrics.undefined`, so a reader can never mistake "not computable" for
"zero".

Threshold
---------
Predictions default to :data:`DECISION_THRESHOLD` = 0.5, the conventional
classifier cut-off.

> **0.5 is a technical default, not a medically validated threshold.** It carries
> no clinical meaning and is not derived from this data, from any diagnostic
> criterion, or from any clinical guideline. Threshold selection belongs to a
> later stage and must be justified on validation data, never chosen by looking
> at test performance.

Rank metrics use **probability scores**, not the hard 0/1 predictions. An AUC
computed from thresholded classes loses all the ranking information that makes it
useful and would report a different quantity entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix as sklearn_confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from cronical.models.errors import EvaluationError

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "DECISION_THRESHOLD",
    "Metrics",
    "compute_metrics",
    "confusion_counts",
    "prediction_summary",
    "pr_curve_points",
    "roc_curve_points",
]

#: Conventional classifier cut-off. A technical default, not a clinical value.
DECISION_THRESHOLD: Final[float] = 0.5

#: Column index of the positive class in `predict_proba` output.
_POSITIVE_CLASS: Final[int] = 1


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """The four cells of a binary confusion matrix.

    Attributes:
        true_positives: Predicted positive, actually positive.
        false_positives: Predicted positive, actually negative.
        true_negatives: Predicted negative, actually negative.
        false_negatives: Predicted negative, actually positive.
    """

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def total(self) -> int:
        """Number of observations accounted for."""
        return (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )

    def as_matrix(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return the matrix in scikit-learn's row/column orientation."""
        return (
            (self.true_negatives, self.false_positives),
            (self.false_negatives, self.true_positives),
        )

    def as_dict(self) -> dict[str, int]:
        """Return the four counts keyed by name."""
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
        }


@dataclass(frozen=True, slots=True)
class Metrics:
    """Every metric computed for one model on one dataset.

    Attributes:
        accuracy: Share of all predictions that were correct. Always defined: the
            inputs are required to be non-empty.
        precision: Of those predicted positive, the share actually positive.
            ``None`` when nothing was predicted positive.
        recall: Of those actually positive, the share found. Also reported as
            ``sensitivity``; the two are the same quantity.
        f1: Harmonic mean of precision and recall. ``None`` when either is
            undefined.
        specificity: Of those actually negative, the share correctly rejected.
        sensitivity: Same quantity as ``recall``, named as clinicians do.
        roc_auc: Area under the ROC curve, from probability scores. ``None`` when
            only one class is present in the truth.
        pr_auc: Average precision. ``None`` when only one class is present, since
            scikit-learn flags the value as degenerate in that case.
        threshold: The cut-off used to produce the hard predictions.
        counts: The confusion-matrix cells.
        n_samples: Number of observations scored.
        undefined: Names of metrics that could not be computed, so an absent
            metric is always visible rather than silently missing.
    """

    accuracy: float
    precision: float | None
    recall: float | None
    f1: float | None
    specificity: float | None
    sensitivity: float | None
    roc_auc: float | None
    pr_auc: float | None
    threshold: float
    counts: ConfusionCounts
    n_samples: int
    undefined: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, with ``None`` preserved for gaps."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "specificity": self.specificity,
            "sensitivity": self.sensitivity,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "threshold": self.threshold,
            "confusion_matrix": [list(row) for row in self.counts.as_matrix()],
            **self.counts.as_dict(),
            "n_samples": self.n_samples,
            "undefined": list(self.undefined),
        }

    def as_row(self) -> dict[str, float | int | str | None]:
        """Return the comparison-table view of the metrics."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "specificity": self.specificity,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "n_samples": self.n_samples,
        }


def _as_array(values: ArrayLike) -> np.ndarray:
    """Return ``values`` as a 1-D integer array."""
    return np.asarray(values).ravel().astype(int)


def _as_scores(probabilities: ArrayLike) -> np.ndarray:
    """Return positive-class probability scores as a 1-D float array.

    Accepts either a 1-D array of positive-class scores or the 2-column output of
    ``predict_proba``, so a caller need not reshape it first.
    """
    array = np.asarray(probabilities, dtype=float)
    if array.ndim == 2:
        if array.shape[1] != 2:
            raise EvaluationError(
                f"expected 2 probability columns (negative, positive), got {array.shape[1]}"
            )
        array = array[:, _POSITIVE_CLASS]
    if array.ndim != 1:
        raise EvaluationError(f"expected 1-D probability scores, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise EvaluationError("probability scores contain non-finite values")
    if ((array < 0.0) | (array > 1.0)).any():
        raise EvaluationError("probability scores fall outside [0, 1]")
    return array


def _validate_binary(values: np.ndarray, *, label: str) -> np.ndarray:
    """Check an array holds only the documented binary labels."""
    unique = set(np.unique(values).tolist())
    if not unique <= {0, 1}:
        raise EvaluationError(f"{label} must contain only 0 and 1, found {sorted(unique)}")
    return values


def confusion_counts(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> ConfusionCounts:
    """Compute the four confusion-matrix cells.

    Args:
        y_true: Actual labels.
        y_pred: Predicted labels.

    Returns:
        The counts, verified against scikit-learn's own confusion matrix.

    Raises:
        EvaluationError: If the inputs are not binary or are different lengths.
    """
    truth = _validate_binary(_as_array(y_true), label="y_true")
    predicted = _validate_binary(_as_array(y_pred), label="y_pred")
    if truth.shape != predicted.shape:
        raise EvaluationError(
            f"y_true has {truth.shape[0]} row(s) but y_pred has {predicted.shape[0]}"
        )
    matrix = sklearn_confusion_matrix(truth, predicted, labels=[0, 1])
    true_negatives, false_positives = int(matrix[0, 0]), int(matrix[0, 1])
    false_negatives, true_positives = int(matrix[1, 0]), int(matrix[1, 1])
    return ConfusionCounts(
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    """Return the ratio, or ``None`` when the denominator is zero."""
    return float(numerator / denominator) if denominator else None


def compute_metrics(
    y_true: ArrayLike,
    y_probabilities: ArrayLike,
    *,
    threshold: float = DECISION_THRESHOLD,
) -> Metrics:
    """Compute every binary-classification metric from probability scores.

    Hard predictions are derived here from ``threshold``; the caller passes only
    probabilities, which removes any chance of an AUC being computed from
    thresholded labels.

    Args:
        y_true: Actual binary labels.
        y_probabilities: Positive-class scores, either 1-D or as the 2-column
            output of ``predict_proba``.
        threshold: Probability at which a prediction becomes positive. The
            technical default of 0.5 is not a medically validated cut-off.

    Returns:
        Every metric, with undefined ones set to ``None`` and named in
        :attr:`Metrics.undefined`.

    Raises:
        EvaluationError: If inputs are not binary, are different lengths, or the
            threshold is outside (0, 1).
    """
    if not 0.0 < threshold < 1.0:
        raise EvaluationError(f"threshold must lie strictly between 0 and 1, got {threshold}")

    truth = _validate_binary(_as_array(y_true), label="y_true")
    if truth.shape[0] == 0:
        # Caught here so that every metric below is well defined for the rest of
        # the function, rather than each one guarding against an empty input.
        raise EvaluationError("cannot compute metrics from zero observations")
    scores = _as_scores(y_probabilities)
    if truth.shape != scores.shape:
        raise EvaluationError(
            f"y_true has {truth.shape[0]} row(s) but probabilities have "
            f"{scores.shape[0]}"
        )

    predictions = (scores >= threshold).astype(int)
    counts = confusion_counts(truth, predictions)
    undefined: list[str] = []

    precision = _safe_ratio(counts.true_positives, counts.true_positives + counts.false_positives)
    recall = _safe_ratio(counts.true_positives, counts.true_positives + counts.false_negatives)
    specificity = _safe_ratio(
        counts.true_negatives, counts.true_negatives + counts.false_positives
    )
    if precision is None:
        undefined.append("precision")
    if recall is None:
        undefined.append("recall")
        undefined.append("sensitivity")
    if specificity is None:
        undefined.append("specificity")

    f1: float | None = None
    if precision is not None and recall is not None and (precision + recall) > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    elif precision is None or recall is None:
        undefined.append("f1")

    # The non-empty check above means this ratio can never divide by zero.
    accuracy = float((counts.true_positives + counts.true_negatives) / counts.total)

    roc_auc, pr_auc = _ranking_metrics(truth, scores, undefined)

    return Metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        sensitivity=recall,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        threshold=threshold,
        counts=counts,
        n_samples=int(truth.shape[0]),
        undefined=tuple(undefined),
    )


def _ranking_metrics(
    truth: np.ndarray, scores: np.ndarray, undefined: list[str]
) -> tuple[float | None, float | None]:
    """Return ``(roc_auc, pr_auc)``, or ``(None, None)`` when only one class exists.

    Both are ranking metrics over pairs of classes. With one class absent there is
    nothing to rank against: scikit-learn returns NaN for ROC-AUC and, for average
    precision, emits a warning that recall is set to one at every threshold. A
    number produced under that caveat would be reported as if it meant something,
    so both are marked undefined instead.
    """
    if len(np.unique(truth)) < 2:
        undefined.append("roc_auc")
        undefined.append("pr_auc")
        return None, None
    return float(roc_auc_score(truth, scores)), float(average_precision_score(truth, scores))


def prediction_summary(y_true: ArrayLike) -> dict[str, int]:
    """Return the class distribution of an outcome column."""
    values = _as_array(y_true)
    return {str(value): int(count) for value, count in zip(*np.unique(values, return_counts=True))}


def roc_curve_points(
    y_true: ArrayLike,
    y_probabilities: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(false_positive_rate, true_positive_rate, thresholds)``.

    For plotting a ROC curve. Raises when only one class is present, because a
    ROC curve is not defined in that case.

    Raises:
        EvaluationError: If the inputs cannot produce a curve.
    """
    truth = _validate_binary(_as_array(y_true), label="y_true")
    scores = _as_scores(y_probabilities)
    if truth.shape != scores.shape:
        raise EvaluationError("y_true and probabilities must have the same length")
    if len(np.unique(truth)) < 2:
        raise EvaluationError(
            "a ROC curve needs both classes present in y_true; only one was found"
        )
    return roc_curve(truth, scores)


def pr_curve_points(
    y_true: ArrayLike,
    y_probabilities: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(recall, precision, thresholds)``.

    For plotting a precision-recall curve.

    Raises:
        EvaluationError: If the inputs cannot produce a curve.
    """
    truth = _validate_binary(_as_array(y_true), label="y_true")
    scores = _as_scores(y_probabilities)
    if truth.shape != scores.shape:
        raise EvaluationError("y_true and probabilities must have the same length")
    if truth.shape[0] == 0:
        raise EvaluationError("a precision-recall curve needs at least one observation")
    precision, recall, thresholds = precision_recall_curve(truth, scores)
    return recall, precision, thresholds


def iter_metric_names() -> Iterator[str]:
    """Yield the metric names in the order they appear in a comparison table."""
    yield from ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "pr_auc")