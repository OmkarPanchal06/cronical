"""Tests for :mod:`cronical.models.evaluate`.

The metric functions are verified against hand-computed values on small,
constructed examples. That is a test of the *metric implementation*, not a claim
about any model's performance: no expected figure here describes a real dataset,
and none is derived from the Pima dataset.

The recurring checks are that every number comes from the arrays supplied, and
that an undefined metric is reported as a gap rather than as zero.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cronical.models.errors import EvaluationError
from cronical.models.evaluate import (
    DECISION_THRESHOLD,
    compute_metrics,
    confusion_counts,
    iter_metric_names,
    pr_curve_points,
    prediction_summary,
    roc_curve_points,
)


class TestConfusionCounts:
    """The four cells, against hand-computed examples."""

    def test_all_correct(self) -> None:
        counts = confusion_counts([0, 1], [0, 1])
        assert counts.as_dict() == {
            "true_positives": 1,
            "false_positives": 0,
            "true_negatives": 1,
            "false_negatives": 0,
        }

    def test_one_of_each_cell(self) -> None:
        """y_true=[0,0,1,1] against y_pred=[0,1,0,1] lands on TN, FP, FN, TP."""
        counts = confusion_counts([0, 0, 1, 1], [0, 1, 0, 1])
        assert counts.true_negatives == 1
        assert counts.false_positives == 1
        assert counts.false_negatives == 1
        assert counts.true_positives == 1
        assert counts.total == 4

    def test_matrix_orientation_matches_sklearn(self) -> None:
        counts = confusion_counts([0, 0, 1, 1], [0, 1, 0, 1])
        assert counts.as_matrix() == ((1, 1), (1, 1))

    def test_matrix_sums_to_total(self) -> None:
        counts = confusion_counts([0, 1, 1, 0, 1], [0, 1, 0, 1, 0])
        assert sum(sum(row) for row in counts.as_matrix()) == counts.total

    def test_accepts_numpy_input(self) -> None:
        assert confusion_counts(np.array([0, 1]), np.array([0, 1])).total == 2

    def test_rejects_non_binary_truth(self) -> None:
        with pytest.raises(EvaluationError, match="only 0 and 1"):
            confusion_counts([0, 2], [0, 1])

    def test_rejects_non_binary_predictions(self) -> None:
        with pytest.raises(EvaluationError, match="only 0 and 1"):
            confusion_counts([0, 1], [0, -1])

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(EvaluationError, match="but y_pred has"):
            confusion_counts([0, 1, 1], [0, 1])


class TestMetricValues:
    """Metrics against hand-computed values."""

    @staticmethod
    def _perfect() -> tuple[list[int], list[float]]:
        return [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]

    def test_a_perfect_prediction_scores_one(self) -> None:
        metrics = compute_metrics(*self._perfect())
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.specificity == 1.0
        assert metrics.roc_auc == 1.0
        assert metrics.pr_auc == 1.0

    def test_sensitivity_equals_recall(self) -> None:
        metrics = compute_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
        assert metrics.sensitivity == metrics.recall

    def test_hand_computed_confusion_matrix(self) -> None:
        """Scores [0.1,0.9,0.2,0.8] against [0,0,1,1] predict [0,1,0,1].

        That lands one observation in each of the four cells, so accuracy is 2/4,
        precision and recall are each 1/2, and specificity is TN/(TN+FP) = 1/2.
        """
        metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8])
        assert metrics.counts.true_negatives == 1
        assert metrics.counts.false_positives == 1
        assert metrics.counts.false_negatives == 1
        assert metrics.counts.true_positives == 1
        assert metrics.accuracy == 0.5
        assert metrics.precision == 0.5
        assert metrics.recall == 0.5
        assert metrics.f1 == 0.5
        assert metrics.specificity == 0.5

    def test_roc_auc_is_computed_from_ranking_not_from_the_threshold(self) -> None:
        """One positive outranks two negatives but not the highest-scoring one.

        AUC is 2/3, because two of the three negative comparisons are ordered
        correctly, while accuracy at the 0.5 cut-off is only 2/4: the row with the
        highest score is actually negative. An AUC computed from thresholded labels
        could not express that gap.
        """
        scores = [0.1, 0.2, 0.3, 0.9]
        truth = [0, 0, 1, 0]
        metrics = compute_metrics(truth, scores)
        assert metrics.roc_auc == pytest.approx(2 / 3)
        assert metrics.accuracy == 0.5
        assert metrics.roc_auc is not None
        assert metrics.roc_auc > metrics.accuracy

    def test_flipped_ranking_lowers_auc(self) -> None:
        truth = [0, 0, 1, 1]
        assert compute_metrics(truth, [0.1, 0.2, 0.8, 0.9]).roc_auc == 1.0
        assert compute_metrics(truth, [0.9, 0.8, 0.2, 0.1]).roc_auc == 0.0

    def test_probabilities_at_the_threshold_are_positive(self) -> None:
        """The default cut-off is inclusive of the score itself."""
        metrics = compute_metrics([0, 1], [0.5, 0.5])
        assert metrics.counts.true_positives == 1
        assert metrics.counts.false_positives == 1

    def test_threshold_changes_the_hard_predictions(self) -> None:
        truth = [0, 1]
        scores = [0.4, 0.6]
        assert compute_metrics(truth, scores, threshold=0.5).counts.true_positives == 1
        assert compute_metrics(truth, scores, threshold=0.7).counts.true_positives == 0

    def test_threshold_does_not_change_auc(self) -> None:
        truth = [0, 0, 1, 1]
        scores = [0.1, 0.4, 0.6, 0.9]
        assert compute_metrics(truth, scores, threshold=0.2).roc_auc == (
            compute_metrics(truth, scores, threshold=0.8).roc_auc
        )

    def test_default_threshold_is_the_conventional_half(self) -> None:
        assert DECISION_THRESHOLD == 0.5
        assert compute_metrics([1], [0.6]).threshold == 0.5

    def test_sample_count_is_recorded(self) -> None:
        assert compute_metrics([0, 1, 1], [0.1, 0.9, 0.8]).n_samples == 3

    def test_accepts_predict_proba_output(self) -> None:
        """A 2-column probability matrix is accepted directly."""
        truth = [0, 1]
        paired = np.array([[0.9, 0.1], [0.2, 0.8]])
        assert compute_metrics(truth, paired).counts.true_positives == 1


class TestUndefinedMetrics:
    """A metric that cannot be computed is a gap, not a zero."""

    def test_no_positive_predictions_makes_precision_undefined(self) -> None:
        metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4])
        assert metrics.precision is None
        assert "precision" in metrics.undefined

    def test_no_actual_positives_makes_recall_undefined(self) -> None:
        metrics = compute_metrics([0, 0], [0.1, 0.2])
        assert metrics.recall is None
        assert metrics.sensitivity is None
        assert "recall" in metrics.undefined

    def test_undefined_precision_propagates_to_f1(self) -> None:
        metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4])
        assert metrics.f1 is None
        assert "f1" in metrics.undefined

    def test_no_actual_negatives_makes_specificity_undefined(self) -> None:
        metrics = compute_metrics([1, 1], [0.9, 0.8])
        assert metrics.specificity is None
        assert "specificity" in metrics.undefined

    def test_single_class_makes_roc_auc_undefined(self) -> None:
        metrics = compute_metrics([1, 1, 1], [0.2, 0.5, 0.9])
        assert metrics.roc_auc is None
        assert "roc_auc" in metrics.undefined

    def test_single_class_still_reports_precision_and_recall(self) -> None:
        """Scores [0.2,0.5,0.9] against an all-positive truth predict [0,1,1].

        Only the ratio metrics degrade: precision and recall remain computable.
        """
        metrics = compute_metrics([1, 1, 1], [0.2, 0.5, 0.9])
        assert metrics.counts.true_positives == 2
        assert metrics.counts.false_negatives == 1
        assert metrics.precision == 1.0
        assert metrics.recall == pytest.approx(2 / 3)
        assert metrics.roc_auc is None
        assert metrics.pr_auc is None

    def test_a_perfect_single_class_prediction_is_reported_honestly(self) -> None:
        metrics = compute_metrics([1, 1, 1], [0.9, 0.9, 0.9])
        assert metrics.accuracy == 1.0
        assert metrics.roc_auc is None, "ROC-AUC is undefined for one class"

    def test_undefined_is_empty_for_a_normal_case(self) -> None:
        assert compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]).undefined == ()

    def test_accuracy_is_always_defined_for_valid_input(self) -> None:
        """Only the ratio-based metrics can go undefined, never accuracy."""
        for truth, scores in (
            ([0, 1], [0.1, 0.9]),
            ([1, 1], [0.9, 0.9]),
            ([0, 0], [0.1, 0.2]),
            ([0, 1, 1], [0.4, 0.6, 0.5]),
        ):
            metrics = compute_metrics(truth, scores)
            assert "accuracy" not in metrics.undefined
            assert 0.0 <= metrics.accuracy <= 1.0

    def test_single_class_leaves_pr_auc_undefined(self) -> None:
        """Single-class average precision is left undefined.

        scikit-learn warns that the value is degenerate without both classes, so
        reporting the number would overstate what it means.
        """
        metrics = compute_metrics([1, 1], [0.9, 0.8])
        assert metrics.pr_auc is None
        assert "pr_auc" in metrics.undefined

    def test_both_ranking_metrics_need_both_classes(self) -> None:
        """ROC-AUC and average precision are ranked over pairs of classes."""
        for truth in ([1, 1, 1], [0, 0, 0]):
            metrics = compute_metrics(truth, [0.2, 0.5, 0.9])
            assert "roc_auc" in metrics.undefined
            assert "pr_auc" in metrics.undefined


class TestInvalidInput:
    """Malformed metric inputs are rejected."""

    def test_non_binary_truth_is_rejected(self) -> None:
        with pytest.raises(EvaluationError, match="y_true must contain only 0 and 1"):
            compute_metrics([0, 2], [0.1, 0.9])

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(EvaluationError, match="probabilities have"):
            compute_metrics([0, 1, 1], [0.1, 0.9])

    @pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.5])
    def test_threshold_outside_the_open_unit_interval_is_rejected(self, threshold: float) -> None:
        with pytest.raises(EvaluationError, match="strictly between 0 and 1"):
            compute_metrics([0, 1], [0.1, 0.9], threshold=threshold)

    def test_scores_outside_the_unit_interval_are_rejected(self) -> None:
        with pytest.raises(EvaluationError, match=r"outside \[0, 1\]"):
            compute_metrics([0, 1], [0.1, 1.4])

    def test_non_finite_scores_are_rejected(self) -> None:
        with pytest.raises(EvaluationError, match="non-finite"):
            compute_metrics([0, 1], [0.1, float("nan")])

    def test_wrong_probability_width_is_rejected(self) -> None:
        with pytest.raises(EvaluationError, match="expected 2 probability columns"):
            compute_metrics([0, 1], np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]]))

    def test_higher_dimensional_scores_are_rejected(self) -> None:
        with pytest.raises(EvaluationError, match="expected 1-D probability scores"):
            compute_metrics([0, 1], np.zeros((2, 2, 2)))

    def test_zero_observations_are_rejected(self) -> None:
        with pytest.raises(EvaluationError, match="zero observations"):
            compute_metrics([], np.array([]))


class TestCurves:
    """Curve extraction for plotting."""

    def test_roc_points_are_monotonic(self) -> None:
        fpr, tpr, _ = roc_curve_points([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert (fpr >= 0).all() and (fpr <= 1).all()
        assert (tpr >= 0).all() and (tpr <= 1).all()
        assert np.all(np.diff(fpr) >= -1e-12)

    def test_roc_needs_both_classes(self) -> None:
        with pytest.raises(EvaluationError, match="both classes"):
            roc_curve_points([1, 1], [0.4, 0.6])

    def test_roc_rejects_a_length_mismatch(self) -> None:
        with pytest.raises(EvaluationError, match="same length"):
            roc_curve_points([0, 1, 1], [0.1, 0.9])

    def test_pr_points_are_returned_as_recall_precision_threshold(self) -> None:
        recall, precision, thresholds = pr_curve_points([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert recall.shape == precision.shape
        assert thresholds.shape[0] == recall.shape[0] - 1

    def test_pr_needs_at_least_one_observation(self) -> None:
        with pytest.raises(EvaluationError, match="at least one observation"):
            pr_curve_points([], [])

    def test_pr_rejects_non_binary_truth(self) -> None:
        with pytest.raises(EvaluationError, match="only 0 and 1"):
            pr_curve_points([0, 3], [0.1, 0.9])

    def test_pr_rejects_a_length_mismatch(self) -> None:
        with pytest.raises(EvaluationError, match="same length"):
            pr_curve_points([0, 1, 1], [0.1, 0.9])


class TestReportingHelpers:
    """Serialisation and small utilities."""

    def test_prediction_summary_counts_classes(self) -> None:
        assert prediction_summary([0, 1, 1, 1]) == {"0": 1, "1": 3}

    def test_prediction_summary_handles_a_single_class(self) -> None:
        assert prediction_summary([1, 1]) == {"1": 2}

    def test_metrics_are_json_serialisable(self) -> None:
        payload = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]).as_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_confusion_counts_are_serialised_as_a_matrix(self) -> None:
        """Scores [0.1,0.9,0.8,0.2] against [0,0,1,1] predict [0,1,1,0]."""
        payload = compute_metrics([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2]).as_dict()
        assert payload["confusion_matrix"] == [[1, 1], [1, 1]]

    def test_undefined_metrics_survive_serialisation(self) -> None:
        """An all-positive truth leaves specificity and both ranking metrics undefined."""
        payload = compute_metrics([1, 1], [0.9, 0.8]).as_dict()
        assert payload["undefined"] == ["specificity", "roc_auc", "pr_auc"]
        assert payload["specificity"] is None
        assert payload["roc_auc"] is None
        assert payload["pr_auc"] is None

    def test_row_view_covers_the_comparison_metrics(self) -> None:
        row = compute_metrics([0, 1], [0.1, 0.9]).as_row()
        assert set(row) == {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "roc_auc",
            "pr_auc",
            "n_samples",
        }

    def test_metric_names_are_stable(self) -> None:
        assert tuple(iter_metric_names()) == (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "roc_auc",
            "pr_auc",
        )
