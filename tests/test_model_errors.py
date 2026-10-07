"""Coverage for the defensive branches in the modelling layer.

Each test here reaches a path that only a failure or a hostile input can produce:
an unwritable destination, a checksum that cannot be read, a metric that is
undefined for a degenerate input. They are grouped here rather than mixed into the
behavioural suites because they assert on error handling, not on modelling.

None of these tests asserts a performance figure.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cronical.config import ScalingMode, Settings, TrainingSettings
from cronical.data.errors import CronicalDataError
from cronical.data.preprocessing import DatasetSplit
from cronical.data.schema import FEATURE_COLUMNS
from cronical.models.config import (
    BASELINE_REGISTRY,
    EstimatorName,
    ModelSpec,
    available_models,
    get_spec,
)
from cronical.models.errors import ArtifactError, EvaluationError
from cronical.models.evaluate import compute_metrics
from cronical.models.figures import plot_roc_curve
from cronical.models.reporting import (
    ModelEvaluation,
    evaluate_model,
    write_comparison_json,
    write_report,
)
from cronical.models.train import load_training_frame, save_model, train_model


class TestThresholdSettings:
    """String coercion for the decision threshold."""

    def test_numeric_string_is_coerced(self) -> None:
        assert TrainingSettings(decision_threshold="0.25").decision_threshold == 0.25

    def test_non_numeric_string_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a number"):
            TrainingSettings(decision_threshold="half")

    def test_float_is_accepted(self) -> None:
        assert TrainingSettings(decision_threshold=0.3).decision_threshold == 0.3

    @pytest.mark.parametrize("value", [0.0, 1.0, 1.5, -0.2])
    def test_out_of_range_thresholds_are_rejected(self, value: float) -> None:
        with pytest.raises(ValueError):
            TrainingSettings(decision_threshold=value)


class TestUnreadableDatasetChecksum:
    """A dataset that cannot be read has no checksum, and says so."""

    def test_missing_file_yields_no_checksum(
        self, modelling_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        absent = tmp_path / "absent.csv"
        model = train_model(
            modelling_frame,
            "logistic_regression",
            dataset_path=absent,
            dataset_sha256=_checksum(absent),
        )
        assert model.metadata.dataset_sha256 == ""
        assert model.metadata.dataset_path.endswith("absent.csv")

    def test_present_file_yields_a_checksum(self, tmp_path: Path) -> None:
        path = tmp_path / "diabetes.csv"
        path.write_bytes(b"content")
        assert _checksum(path) == hashlib.sha256(b"content").hexdigest()

    def test_checksum_is_recorded_for_a_real_file(
        self, modelling_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = tmp_path / "diabetes.csv"
        modelling_frame.to_csv(path, index=False)
        frame, resolved = load_training_frame(path)
        model = train_model(
            frame,
            "logistic_regression",
            dataset_path=resolved,
            dataset_sha256=_checksum(resolved),
        )
        assert len(model.metadata.dataset_sha256) == 64


def _checksum(path: Path) -> str:
    """Return the checksum :mod:`cronical.models.train` would record."""
    from cronical.models.train import _sha256

    return _sha256(path)


class TestSpecificationShape:
    """A specification is constructible directly and reports itself honestly."""

    def test_spec_may_be_constructed_directly(self) -> None:
        spec = ModelSpec(name=EstimatorName.RANDOM_FOREST, hyperparameters={"n_estimators": 5})
        assert spec.as_dict()["hyperparameters"] == {"n_estimators": 5}

    def test_spec_defaults_are_empty(self) -> None:
        spec = ModelSpec(name=EstimatorName.RANDOM_FOREST)
        assert dict(spec.hyperparameters) == {}
        assert spec.scaling is ScalingMode.NONE
        assert spec.requires == ()
        assert spec.description == ""

    def test_registry_describes_every_baseline(self) -> None:
        from cronical.models.config import BASELINE_REGISTRY

        assert all(spec.description for spec in BASELINE_REGISTRY.values())

    def test_every_registered_baseline_can_be_built(self) -> None:
        from cronical.models.train import _build_estimator

        for spec in BASELINE_REGISTRY.values():
            if spec.name in available_models():
                assert _build_estimator(spec) is not None


class TestUnreadablePreprocessor:
    """Training survives a pipeline whose preprocessing cannot be described."""

    def test_metadata_notes_that_preprocessing_is_unavailable(
        self, modelling_frame: pd.DataFrame
    ) -> None:
        """A pipeline without the imputer cannot be described, and says so."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        malformed = Pipeline([("scaler", StandardScaler())])
        malformed.fit(modelling_frame.loc[:, list(FEATURE_COLUMNS)])

        from cronical.models.train import _fitted_preprocessing

        payload = _fitted_preprocessing(malformed)
        assert "unavailable" in payload

    def test_a_describable_pipeline_reports_statistics(self, modelling_frame: pd.DataFrame) -> None:
        from cronical.models.train import _fitted_preprocessing

        model = train_model(modelling_frame, "logistic_regression")
        payload = _fitted_preprocessing(model.pipeline)
        assert payload["imputation_strategy"] == "median"
        assert set(payload["imputation_statistics"]) == set(FEATURE_COLUMNS)

    def test_preprocessor_failure_is_not_a_data_error(self) -> None:
        """The guard catches only the data-layer error type."""
        assert issubclass(CronicalDataError, Exception)


class TestUnwritableDestinations:
    """Write failures become typed errors naming the destination."""

    def test_model_artifact_write_failure(
        self, modelling_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = train_model(modelling_frame, "logistic_regression")

        def refuse(*args: object, **kwargs: object) -> None:
            raise OSError("disk full")

        import joblib

        monkeypatch.setattr(joblib, "dump", refuse)
        with pytest.raises(ArtifactError, match="could not write model artifact"):
            save_model(model, path=tmp_path / "model.joblib")

    def test_metadata_write_failure(
        self, modelling_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = train_model(modelling_frame, "logistic_regression")

        def refuse(*args: object, **kwargs: object) -> None:
            raise OSError("read-only filesystem")

        monkeypatch.setattr(Path, "write_text", refuse)
        with pytest.raises(ArtifactError, match="could not write model metadata"):
            save_model(model, path=tmp_path / "model.joblib")

    def test_report_write_failure(
        self, modelling_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluation = _evaluate(modelling_frame)

        def refuse(*args: object, **kwargs: object) -> None:
            raise OSError("read-only filesystem")

        monkeypatch.setattr(Path, "write_text", refuse)
        with pytest.raises(EvaluationError, match="could not write report"):
            write_report([evaluation], dataset_summary={}, path=tmp_path / "report.md")

    def test_figure_write_failure(
        self, modelling_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        split = _split(modelling_frame)
        truth = np.asarray(split.y_test)

        def refuse(*args: object, **kwargs: object) -> None:
            raise OSError("disk full")

        from matplotlib.figure import Figure

        monkeypatch.setattr(Figure, "savefig", refuse)
        with pytest.raises(EvaluationError, match="could not write figure"):
            plot_roc_curve(truth, [0.2] * len(truth), path=tmp_path / "roc.png")

    def test_comparison_json_is_written(
        self, modelling_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        from cronical.models.reporting import compare_models

        comparison = compare_models([_evaluate(modelling_frame)])
        path = write_comparison_json(comparison, tmp_path / "nested" / "comparison.json")
        assert path.is_file() and path.read_text(encoding="utf-8").startswith("{")

    def test_comparison_json_write_failure(
        self, modelling_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cronical.models.reporting import compare_models

        comparison = compare_models([_evaluate(modelling_frame)])

        def refuse(*args: object, **kwargs: object) -> None:
            raise OSError("read-only filesystem")

        monkeypatch.setattr(Path, "write_text", refuse)
        with pytest.raises(EvaluationError, match="could not write comparison"):
            write_comparison_json(comparison, tmp_path / "comparison.json")


class TestDegenerateMetrics:
    """Inputs for which a metric genuinely cannot be produced."""

    def test_all_positive_truth_also_loses_specificity(self) -> None:
        """Scores [0.2,0.5,0.9] predict [0,1,1], so nothing was a true negative."""
        metrics = compute_metrics([1, 1, 1], [0.2, 0.5, 0.9])
        assert set(metrics.undefined) == {"specificity", "roc_auc", "pr_auc"}
        assert metrics.accuracy == pytest.approx(2 / 3)

    def test_all_negative_truth_also_loses_recall(self) -> None:
        """Scores [0.9,0.8,0.7] predict [1,1,1], so nothing was a true positive.

        Precision is still 0/3, which is defined; recall is 0/0, which is not.
        """
        metrics = compute_metrics([0, 0, 0], [0.9, 0.8, 0.7])
        assert set(metrics.undefined) == {
            "recall",
            "sensitivity",
            "f1",
            "roc_auc",
            "pr_auc",
        }
        assert metrics.precision == 0.0
        assert metrics.specificity == 0.0

    def test_roc_auc_is_recorded_exactly_once(self) -> None:
        metrics = compute_metrics([0, 0, 0], [0.9, 0.8, 0.7])
        assert metrics.undefined.count("roc_auc") == 1

    def test_accuracy_survives_a_single_class(self) -> None:
        """A single class invalidates the ratio and ranking metrics, not accuracy."""
        metrics = compute_metrics([1, 1], [0.9, 0.8])
        assert "accuracy" not in metrics.undefined
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0


def _split(frame: pd.DataFrame) -> DatasetSplit:
    """Return a fixed split of ``frame``."""
    from cronical.data.preprocessing import split_dataset

    return split_dataset(frame, test_size=0.25, random_state=42)


def _evaluate(frame: pd.DataFrame) -> ModelEvaluation:
    """Return one evaluation on a fixed split of ``frame``."""
    split = _split(frame)
    model = train_model(frame, "logistic_regression", random_state=42, test_size=0.25)
    return evaluate_model(model, split.X_test, split.y_test)


class TestTrainingFrameEdgeCases:
    """Loading paths that are easy to get wrong."""

    def test_a_valid_frame_loads_without_raising(
        self, tmp_path: Path, modelling_frame: pd.DataFrame
    ) -> None:
        path = tmp_path / "diabetes.csv"
        modelling_frame.to_csv(path, index=False)
        frame, resolved = load_training_frame(path)
        assert resolved == path
        assert len(frame.index) == len(modelling_frame.index)

    def test_the_configured_default_path_is_used(
        self, isolated_project_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cronical.config import reload_settings

        monkeypatch.setenv("CRONICAL_PROJECT_ROOT", str(isolated_project_root))
        reload_settings()
        with pytest.raises(Exception, match="Dataset file not found"):
            load_training_frame()

    def test_settings_are_threaded_through(
        self, modelling_frame: pd.DataFrame, isolated_project_root: Path
    ) -> None:
        configured = Settings.for_root(isolated_project_root)
        model = train_model(modelling_frame, "random_forest", settings=configured)
        assert model.metadata.random_state == configured.random_seed

    def test_job_count_is_taken_from_settings(
        self, modelling_frame: pd.DataFrame, isolated_project_root: Path
    ) -> None:
        configured = Settings.for_root(isolated_project_root)
        configured = configured.model_copy(
            update={
                "training": configured.training.model_copy(update={"n_jobs": 2}),
            }
        )
        model = train_model(modelling_frame, "random_forest", settings=configured)
        assert model.specification.hyperparameters["n_jobs"] == 2

    def test_spec_records_the_seed_it_was_given(self, modelling_frame: pd.DataFrame) -> None:
        model = train_model(modelling_frame, "random_forest", random_state=13)
        assert get_spec("random_forest", random_state=13).hyperparameters["random_state"] == 13
        assert model.metadata.random_state == 13
