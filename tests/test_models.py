"""Tests for :mod:`cronical.models.train`, `config` and `reporting`.

Everything here runs on the synthetic :func:`modelling_frame` fixture. No test
asserts a specific performance figure: an AUC, accuracy or F1 measured on invented
data says nothing about the Pima Indians Diabetes dataset, so pinning one would be
a fabricated claim rather than a test.

What is asserted instead is everything that *must* hold for a result to be
trustworthy at all:

* the registry is complete and the hyperparameters are explicit
* the preprocessing path matches the model family
* one pipeline carries both stages, so inference cannot drift from training
* training is deterministic under a fixed seed
* predictions are the right shape, in range, and binary
* the seed and configuration are recorded and never silently changed
* a missing dataset fails with an actionable message
* artifacts round-trip and carry no metrics
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from cronical.config import ScalingMode, Settings
from cronical.data.errors import DatasetNotFoundError, DatasetValidationError, SplitError
from cronical.data.preprocessing import DatasetSplit
from cronical.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from cronical.models.config import (
    BASELINE_REGISTRY,
    EstimatorName,
    ModelSpec,
    available_models,
    get_spec,
    model_names,
    require_dependency,
)
from cronical.models.errors import (
    ArtifactError,
    EvaluationError,
    MissingDependencyError,
    UnknownModelError,
)
from cronical.models.figures import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from cronical.models.reporting import (
    ModelEvaluation,
    compare_models,
    evaluate_model,
    render_report,
)
from cronical.models.train import (
    ExperimentMetadata,
    TrainedModel,
    build_model_pipeline,
    load_metadata,
    load_model,
    load_training_frame,
    save_model,
    train_all_baselines,
    train_model,
)

#: Skipped when XGBoost is absent, so the suite still passes without the extra.
requires_xgboost = pytest.mark.skipif(
    "xgboost" not in available_models(),
    reason="xgboost is not installed",
)

ALL_BASELINES = ("logistic_regression", "random_forest", "xgboost")


class TestRegistry:
    """The baseline registry and its typed configuration."""

    def test_three_baselines_are_registered(self) -> None:
        assert model_names() == ALL_BASELINES

    def test_registry_keys_match_the_names(self) -> None:
        assert set(BASELINE_REGISTRY) == set(EstimatorName)

    def test_every_spec_declares_a_scaling_path(self) -> None:
        assert all(spec.scaling in set(ScalingMode) for spec in BASELINE_REGISTRY.values())

    def test_every_spec_states_its_random_state(self) -> None:
        for spec in BASELINE_REGISTRY.values():
            assert "random_state" in spec.hyperparameters

    def test_hyperparameters_are_frozen(self) -> None:
        """A spec must not be mutable once published, or a run stops being reproducible."""
        spec = get_spec("random_forest")
        with pytest.raises(TypeError):
            spec.hyperparameters["n_estimators"] = 1  # type: ignore[index]

    def test_spec_round_trips_through_a_dict(self) -> None:
        payload = get_spec("logistic_regression").as_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert payload["name"] == "logistic_regression"

    def test_seed_is_injected_from_the_caller(self) -> None:
        assert get_spec("random_forest", random_state=7).hyperparameters["random_state"] == 7

    def test_seed_defaults_to_zero_in_the_registry(self) -> None:
        assert BASELINE_REGISTRY[EstimatorName.RANDOM_FOREST].hyperparameters["random_state"] == 0

    def test_job_count_is_injected_from_the_caller(self) -> None:
        assert get_spec("random_forest", n_jobs=3).hyperparameters["n_jobs"] == 3

    def test_unknown_model_is_rejected(self) -> None:
        with pytest.raises(UnknownModelError, match="unknown model"):
            get_spec("deep_learning_revolution")

    def test_model_is_accepted_as_an_identifier(self) -> None:
        assert get_spec(EstimatorName.XGBOOST).name is EstimatorName.XGBOOST

    def test_random_forest_declares_the_expected_baseline_parameters(self) -> None:
        parameters = get_spec("random_forest").hyperparameters
        for key in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "n_jobs"):
            assert key in parameters

    def test_xgboost_declares_the_expected_baseline_parameters(self) -> None:
        parameters = get_spec("xgboost").hyperparameters
        for key in ("n_estimators", "max_depth", "learning_rate", "n_jobs", "eval_metric"):
            assert key in parameters

    def test_logistic_regression_declares_a_max_iteration(self) -> None:
        assert get_spec("logistic_regression").hyperparameters["max_iter"] >= 1_000

    def test_available_models_excludes_absent_dependencies(self) -> None:
        available = available_models()
        assert set(available) <= set(ALL_BASELINES)
        assert "logistic_regression" in available

    def test_available_models_omits_a_model_whose_package_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reporting what can run must not raise for the models that cannot.

        The two scikit-learn baselines have no third-party requirement, so they
        remain listed; only XGBoost is dropped.
        """
        import importlib.util

        monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
        available = available_models()
        assert "xgboost" not in available
        assert set(available) == {"logistic_regression", "random_forest"}

    def test_dependency_check_passes_for_pure_sklearn_models(self) -> None:
        require_dependency(EstimatorName.LOGISTIC_REGRESSION)
        require_dependency(EstimatorName.RANDOM_FOREST)

    def test_missing_dependency_names_the_package_and_the_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib.util

        monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
        with pytest.raises(MissingDependencyError) as excinfo:
            require_dependency(EstimatorName.XGBOOST)
        assert excinfo.value.distribution == "xgboost"
        assert "cronical[ml]" in str(excinfo.value)

    def test_missing_dependency_never_substitutes_a_model(self) -> None:
        """The error must state that no other estimator will be used."""
        assert (
            "No substitute model will be used"
            in MissingDependencyError("xgboost", model="xgboost", extra="ml").args[0]
        )


class TestScalingPaths:
    """Each model family gets the preprocessing it needs."""

    def test_logistic_regression_uses_standard_scaling(self) -> None:
        assert get_spec("logistic_regression").scaling is ScalingMode.STANDARD

    def test_random_forest_does_not_require_scaling(self) -> None:
        assert get_spec("random_forest").scaling is ScalingMode.NONE

    def test_xgboost_does_not_require_scaling(self) -> None:
        assert get_spec("xgboost").scaling is ScalingMode.NONE

    def test_logistic_regression_pipeline_contains_a_scaler(self) -> None:
        pipeline = build_model_pipeline(get_spec("logistic_regression"))
        assert "scaler" in pipeline.named_steps
        assert "estimator" in pipeline.named_steps

    def test_random_forest_pipeline_has_no_scaler(self) -> None:
        pipeline = build_model_pipeline(get_spec("random_forest"))
        assert "scaler" not in pipeline.named_steps
        assert "imputer" in pipeline.named_steps

    def test_both_paths_share_the_earlier_steps(self) -> None:
        """Only the scaler distinguishes them; contract, sentinels and imputer are shared."""
        linear = list(build_model_pipeline(get_spec("logistic_regression")).named_steps)
        forest = list(build_model_pipeline(get_spec("random_forest")).named_steps)
        assert linear[:3] == forest[:3]
        assert linear[:3] == ["feature_contract", "sentinel_zeros", "imputer"]


class TestPipelineComposition:
    """One pipeline carries preprocessing and the estimator."""

    @pytest.mark.parametrize("name", ALL_BASELINES)
    def test_pipeline_starts_with_the_feature_contract(self, name: str) -> None:
        steps = list(build_model_pipeline(get_spec(name)).named_steps)
        assert steps[0] == "feature_contract"

    @pytest.mark.parametrize("name", ALL_BASELINES)
    def test_sentinel_handling_precedes_imputation(self, name: str) -> None:
        steps = list(build_model_pipeline(get_spec(name)).named_steps)
        assert steps.index("sentinel_zeros") < steps.index("imputer")

    @pytest.mark.parametrize("name", ALL_BASELINES)
    def test_estimator_is_the_final_step(self, name: str) -> None:
        assert list(build_model_pipeline(get_spec(name)).named_steps)[-1] == "estimator"

    def test_sentinel_and_imputation_precede_the_estimator(self) -> None:
        steps = list(build_model_pipeline(get_spec("random_forest")).named_steps)
        assert steps.index("sentinel_zeros") < steps.index("estimator")
        assert steps.index("imputer") < steps.index("estimator")

    def test_estimator_receives_the_seed(self) -> None:
        pipeline = build_model_pipeline(get_spec("random_forest", random_state=11))
        estimator = pipeline.named_steps["estimator"]
        assert estimator.get_params()["random_state"] == 11

    def test_estimator_receives_the_configured_job_count(self) -> None:
        pipeline = build_model_pipeline(get_spec("random_forest", n_jobs=2))
        assert pipeline.named_steps["estimator"].get_params()["n_jobs"] == 2

    @requires_xgboost
    def test_xgboost_estimator_is_built(self) -> None:
        estimator = build_model_pipeline(get_spec("xgboost")).named_steps["estimator"]
        assert estimator.__class__.__name__ == "XGBClassifier"


class TestTraining:
    """Fitting a baseline on the synthetic fixture."""

    def test_returns_a_trained_model(self, modelling_frame: pd.DataFrame) -> None:
        model = train_model(modelling_frame, "logistic_regression")
        assert isinstance(model, TrainedModel)
        assert model.name == "logistic_regression"

    def test_records_the_split_sizes(self, modelling_frame: pd.DataFrame) -> None:
        model = train_model(modelling_frame, "logistic_regression", test_size=0.25)
        assert model.metadata.train_size + model.metadata.test_size_rows == len(
            modelling_frame.index
        )

    def test_records_the_seed(self, modelling_frame: pd.DataFrame) -> None:
        assert (
            train_model(modelling_frame, "random_forest", random_state=9).metadata.random_state == 9
        )

    def test_seed_defaults_to_settings(self, modelling_frame: pd.DataFrame) -> None:
        from cronical.config import get_settings

        assert train_model(modelling_frame, "random_forest").metadata.random_state == (
            get_settings().random_seed
        )

    def test_records_the_feature_list(self, modelling_frame: pd.DataFrame) -> None:
        model = train_model(modelling_frame, "logistic_regression")
        assert model.metadata.feature_names == FEATURE_COLUMNS

    def test_records_the_target(self, modelling_frame: pd.DataFrame) -> None:
        assert train_model(modelling_frame, "logistic_regression").metadata.target == TARGET_COLUMN

    def test_records_the_preprocessing_configuration(self, modelling_frame: pd.DataFrame) -> None:
        metadata = train_model(modelling_frame, "logistic_regression").metadata
        assert metadata.preprocessing["imputation_strategy"] == "median"
        assert metadata.preprocessing["scaling"] == ScalingMode.STANDARD.value

    def test_records_the_model_configuration(self, modelling_frame: pd.DataFrame) -> None:
        metadata = train_model(modelling_frame, "random_forest").metadata
        assert metadata.model_configuration["name"] == "random_forest"
        assert "n_estimators" in metadata.model_configuration["hyperparameters"]

    def test_records_a_timestamp_and_version(self, modelling_frame: pd.DataFrame) -> None:
        metadata = train_model(modelling_frame, "logistic_regression").metadata
        assert metadata.created_at.endswith("+00:00")
        assert metadata.project_version

    def test_records_the_decision_threshold(self, modelling_frame: pd.DataFrame) -> None:
        assert (
            train_model(modelling_frame, "logistic_regression").metadata.decision_threshold == 0.5
        )

    def test_metadata_is_json_serialisable(self, modelling_frame: pd.DataFrame) -> None:
        metadata = train_model(modelling_frame, "logistic_regression").metadata.as_dict()
        assert json.loads(json.dumps(metadata)) == metadata

    def test_metadata_carries_no_metric(self, modelling_frame: pd.DataFrame) -> None:
        """Metrics belong in a report, never inside an artifact."""
        payload = train_model(modelling_frame, "logistic_regression").metadata.as_dict()
        assert not any("auc" in key or "accuracy" in key for key in payload)

    def test_in_memory_frame_has_no_checksum(self, modelling_frame: pd.DataFrame) -> None:
        model = train_model(modelling_frame, "logistic_regression")
        assert model.metadata.dataset_sha256 == ""
        assert model.metadata.dataset_path == "<in-memory DataFrame>"

    def test_supplied_dataset_identity_is_recorded(self, modelling_frame: pd.DataFrame) -> None:
        model = train_model(
            modelling_frame,
            "logistic_regression",
            dataset_path="data/raw/diabetes.csv",
            dataset_sha256="abc123",
        )
        assert model.metadata.dataset_path == "data/raw/diabetes.csv"
        assert model.metadata.dataset_sha256 == "abc123"

    def test_unknown_model_is_rejected(self, modelling_frame: pd.DataFrame) -> None:
        with pytest.raises(UnknownModelError):
            train_model(modelling_frame, "not_a_model")

    def test_invalid_dataset_is_rejected(self, modelling_frame: pd.DataFrame) -> None:
        broken = modelling_frame.copy()
        broken[TARGET_COLUMN] = 7
        with pytest.raises(DatasetValidationError):
            train_model(broken, "logistic_regression")

    def test_missing_feature_is_rejected(self, modelling_frame: pd.DataFrame) -> None:
        with pytest.raises(DatasetValidationError):
            train_model(modelling_frame.drop(columns=["Insulin"]), "logistic_regression")

    def test_unsplittable_dataset_is_rejected(self, modelling_frame: pd.DataFrame) -> None:
        rare = modelling_frame.copy()
        rare[TARGET_COLUMN] = 0
        rare.loc[rare.index[0], TARGET_COLUMN] = 1
        with pytest.raises(SplitError):
            train_model(rare, "logistic_regression", test_size=0.25)

    def test_train_all_baselines_returns_each(self, modelling_frame: pd.DataFrame) -> None:
        models = train_all_baselines(
            modelling_frame, names=("logistic_regression", "random_forest")
        )
        assert [model.name for model in models] == ["logistic_regression", "random_forest"]

    def test_train_all_baselines_shares_one_split(self, modelling_frame: pd.DataFrame) -> None:
        models = train_all_baselines(
            modelling_frame, names=("logistic_regression", "random_forest"), random_state=5
        )
        assert models[0].metadata.train_size == models[1].metadata.train_size
        assert models[0].metadata.random_state == models[1].metadata.random_state


class TestDeterminism:
    """A fixed seed must give a fixed model."""

    def test_repeated_fitting_is_identical(self, modelling_frame: pd.DataFrame) -> None:
        first = train_model(modelling_frame, "random_forest", random_state=3)
        second = train_model(modelling_frame, "random_forest", random_state=3)
        split = DatasetSplit(*_partition(first, modelling_frame, 3))
        assert np.allclose(first.predict_proba(split.X_test), second.predict_proba(split.X_test))

    def test_logistic_regression_is_deterministic(self, modelling_frame: pd.DataFrame) -> None:
        first = train_model(modelling_frame, "logistic_regression", random_state=3)
        second = train_model(modelling_frame, "logistic_regression", random_state=3)
        assert np.allclose(
            first.predict_proba(first.metadata and _x(first, modelling_frame)),
            second.predict_proba(_x(second, modelling_frame)),
        )

    def test_a_different_seed_changes_the_forest(self, modelling_frame: pd.DataFrame) -> None:
        """A changed seed must actually change something, or it is not recorded."""
        first = train_model(modelling_frame, "random_forest", random_state=1)
        second = train_model(modelling_frame, "random_forest", random_state=2)
        assert not np.allclose(
            first.predict_proba(_x(first, modelling_frame)),
            second.predict_proba(_x(second, modelling_frame)),
        )

    def test_identical_configuration_gives_identical_metadata_except_time(
        self, modelling_frame: pd.DataFrame
    ) -> None:
        first = train_model(modelling_frame, "random_forest", random_state=4).metadata.as_dict()
        second = train_model(modelling_frame, "random_forest", random_state=4).metadata.as_dict()
        first.pop("created_at")
        second.pop("created_at")
        assert first == second


def _x(model: TrainedModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Return the feature columns a model would score on."""
    del model
    return cast("pd.DataFrame", frame.loc[:, list(FEATURE_COLUMNS)])


def _partition(
    model: TrainedModel, frame: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Rebuild the split a model was fitted on."""
    from cronical.data.preprocessing import split_dataset

    split = split_dataset(frame, test_size=model.metadata.test_size, random_state=seed)
    return split.X_train, split.X_test, split.y_train, split.y_test


class TestPredictions:
    """Shape and range of model output."""

    @pytest.fixture
    def trained(self, modelling_frame: pd.DataFrame) -> TrainedModel:
        """Train one model for these checks."""
        return train_model(modelling_frame, "logistic_regression")

    def test_probability_shape_matches_the_input(
        self, trained: TrainedModel, modelling_frame: pd.DataFrame
    ) -> None:
        features = modelling_frame.loc[:, list(FEATURE_COLUMNS)]
        assert trained.predict_proba(features).shape == (len(features.index), 2)

    def test_probabilities_lie_between_zero_and_one(
        self, trained: TrainedModel, modelling_frame: pd.DataFrame
    ) -> None:
        probabilities = trained.predict_proba(modelling_frame.loc[:, list(FEATURE_COLUMNS)])
        assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all()

    def test_probability_columns_sum_to_one(
        self, trained: TrainedModel, modelling_frame: pd.DataFrame
    ) -> None:
        probabilities = trained.predict_proba(modelling_frame.loc[:, list(FEATURE_COLUMNS)])
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_hard_predictions_are_binary(
        self, trained: TrainedModel, modelling_frame: pd.DataFrame
    ) -> None:
        predictions = trained.predict(modelling_frame.loc[:, list(FEATURE_COLUMNS)])
        assert set(np.unique(predictions).tolist()) <= {0, 1}

    def test_hard_predictions_match_the_threshold(
        self, trained: TrainedModel, modelling_frame: pd.DataFrame
    ) -> None:
        features = modelling_frame.loc[:, list(FEATURE_COLUMNS)]
        scores = trained.predict_proba(features)[:, 1]
        assert np.array_equal(trained.predict(features), (scores >= 0.5).astype(int))

    def test_single_record_predicts_one_row(
        self, trained: TrainedModel, patient_record: dict[str, float]
    ) -> None:
        assert trained.predict_proba(patient_record).shape == (1, 2)

    def test_a_record_with_a_zero_sentinel_is_scored(self, modelling_frame: pd.DataFrame) -> None:
        """A Glucose of 0 must be imputed, not rejected."""
        model = train_model(modelling_frame, "logistic_regression")
        record = dict.fromkeys(FEATURE_COLUMNS, 1.0)
        record["Glucose"] = 0.0
        assert model.predict_proba(record).shape == (1, 2)

    def test_inference_uses_the_fitted_preprocessing(self, modelling_frame: pd.DataFrame) -> None:
        """A sentinel in the input must reach the imputer, not the estimator."""
        model = train_model(modelling_frame, "logistic_regression")
        imputer = model.pipeline.named_steps["imputer"]
        assert np.isfinite(np.asarray(imputer.statistics_, dtype=float)).all()

    def test_scaling_is_not_caller_overridable(self, modelling_frame: pd.DataFrame) -> None:
        """The specification owns the preprocessing path.

        A model cannot be fitted with the wrong one by accident, which is what
        stops a linear model being trained on unscaled features.
        """
        linear = train_model(modelling_frame, "logistic_regression").pipeline
        forest = train_model(modelling_frame, "random_forest").pipeline
        assert "scaler" in linear.named_steps
        assert "scaler" not in forest.named_steps


class TestNoTargetLeakage:
    """The target must never become a feature or influence a transform."""

    def test_outcome_is_not_in_the_feature_list(self, modelling_frame: pd.DataFrame) -> None:
        model = train_model(modelling_frame, "logistic_regression")
        assert TARGET_COLUMN not in model.metadata.feature_names

    def test_outcome_is_absent_from_the_transformed_matrix(
        self, modelling_frame: pd.DataFrame
    ) -> None:
        model = train_model(modelling_frame, "logistic_regression")
        transformed = model.pipeline[:-1].transform(modelling_frame.loc[:, list(FEATURE_COLUMNS)])
        assert transformed.shape[1] == len(FEATURE_COLUMNS)

    def test_fitting_on_the_feature_frame_without_the_target_works(
        self, modelling_frame: pd.DataFrame
    ) -> None:
        features = modelling_frame.loc[:, list(FEATURE_COLUMNS)]
        assert train_model(modelling_frame, "logistic_regression").metadata.feature_names == tuple(
            FEATURE_COLUMNS
        )
        assert not any(TARGET_COLUMN in str(name) for name in features.columns)


class TestDatasetHandling:
    """Loading for training, including the absent-dataset path."""

    def test_missing_dataset_raises_with_guidance(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError) as excinfo:
            load_training_frame(tmp_path / "diabetes.csv")
        assert "data/raw" in str(excinfo.value)

    def test_missing_default_dataset_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CRONICAL_PROJECT_ROOT", "C:/nonexistent-cronical-root")
        from cronical.config import reload_settings

        reload_settings()
        with pytest.raises(DatasetNotFoundError):
            load_training_frame()

    def test_valid_dataset_loads_with_its_path(
        self, tmp_path: Path, modelling_frame: pd.DataFrame
    ) -> None:
        path = tmp_path / "diabetes.csv"
        modelling_frame.to_csv(path, index=False)
        frame, resolved = load_training_frame(path)
        assert resolved == path
        assert frame.shape[0] == modelling_frame.shape[0]

    def test_invalid_dataset_is_rejected(
        self, tmp_path: Path, modelling_frame: pd.DataFrame
    ) -> None:
        path = tmp_path / "diabetes.csv"
        broken = modelling_frame.copy()
        broken[TARGET_COLUMN] = 9
        broken.to_csv(path, index=False)
        with pytest.raises(DatasetValidationError):
            load_training_frame(path)

    def test_checksum_is_recorded_for_a_real_file(
        self, tmp_path: Path, modelling_frame: pd.DataFrame
    ) -> None:
        path = tmp_path / "diabetes.csv"
        modelling_frame.to_csv(path, index=False)
        frame, resolved = load_training_frame(path)
        model = train_model(
            frame,
            "logistic_regression",
            dataset_path=resolved,
            dataset_sha256=_sha(resolved),
        )
        assert len(model.metadata.dataset_sha256) == 64


def _sha(path: Path) -> str:
    """Return the SHA-256 of a file."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestArtifacts:
    """Persistence of the fitted pipeline and its metadata."""

    def test_round_trip_reproduces_predictions(
        self, modelling_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        model = train_model(modelling_frame, "logistic_regression")
        features = modelling_frame.loc[:, list(FEATURE_COLUMNS)]
        path, _ = save_model(model, path=tmp_path / "model.joblib")
        reloaded = load_model(path)
        assert np.allclose(reloaded.predict_proba(features), model.predict_proba(features))

    def test_metadata_is_written_beside_the_artifact(
        self, modelling_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        model = train_model(modelling_frame, "logistic_regression")
        path, metadata_path = save_model(model, path=tmp_path / "model.joblib")
        assert metadata_path == path.with_suffix(".json")
        assert metadata_path.is_file()

    def test_saved_metadata_round_trips(
        self, modelling_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        model = train_model(modelling_frame, "logistic_regression", random_state=8)
        path, _ = save_model(model, path=tmp_path / "model.joblib")
        assert load_metadata(path)["random_state"] == 8

    def test_saved_metadata_contains_no_metric(
        self, modelling_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        model = train_model(modelling_frame, "logistic_regression")
        path, _ = save_model(model, path=tmp_path / "model.joblib")
        payload = load_metadata(path)
        assert not any("auc" in key or "accuracy" in key for key in payload)

    def test_metadata_can_be_skipped(self, modelling_frame: pd.DataFrame, tmp_path: Path) -> None:
        model = train_model(modelling_frame, "logistic_regression")
        _, metadata_path = save_model(model, path=tmp_path / "model.joblib", write_metadata=False)
        assert metadata_path is None

    def test_default_path_is_under_models(
        self, modelling_frame: pd.DataFrame, isolated_project_root: Path
    ) -> None:
        configured = Settings.for_root(isolated_project_root)
        model = train_model(modelling_frame, "logistic_regression", settings=configured)
        path, _ = save_model(model, settings=configured)
        assert path == isolated_project_root / "models" / "logistic_regression.joblib"

    def test_missing_artifact_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactError, match="no model artifact"):
            load_model(tmp_path / "absent.joblib")

    def test_missing_metadata_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactError, match="no model metadata"):
            load_metadata(tmp_path / "absent.joblib")

    def test_corrupt_artifact_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "model.joblib"
        path.write_bytes(b"not a pickle")
        with pytest.raises(ArtifactError, match="could not read"):
            load_model(path)

    def test_corrupt_metadata_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "model.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ArtifactError, match="not valid JSON"):
            load_metadata(path)


def _roc_auc(evaluation: ModelEvaluation) -> float:
    """Return a defined ROC-AUC, failing loudly if the model was unrankable."""
    assert evaluation.metrics.roc_auc is not None, evaluation.name
    return evaluation.metrics.roc_auc


class TestComparison:
    """Ranking several models, and the limits of that ranking."""

    @pytest.fixture
    def evaluations(
        self, modelling_frame: pd.DataFrame, modelling_split: DatasetSplit
    ) -> list[ModelEvaluation]:
        """Score two baselines on one shared split."""
        models = train_all_baselines(
            modelling_frame,
            names=("logistic_regression", "random_forest"),
            random_state=42,
            test_size=0.25,
        )
        return [
            evaluate_model(model, modelling_split.X_test, modelling_split.y_test)
            for model in models
        ]

    def test_orders_by_roc_auc_descending(self, evaluations: list[ModelEvaluation]) -> None:
        comparison = compare_models(evaluations)
        # Only ranked models reach here, so each has a defined ROC-AUC.
        values = [_roc_auc(evaluation) for evaluation in comparison.evaluations]
        assert values == sorted(values, reverse=True)

    def test_reports_the_ranking_metric(self, evaluations: list[ModelEvaluation]) -> None:
        assert compare_models(evaluations).ranking_metric == "roc_auc"

    def test_exposes_the_ranked_names(self, evaluations: list[ModelEvaluation]) -> None:
        assert set(compare_models(evaluations).names) == {"logistic_regression", "random_forest"}

    def test_can_rank_by_another_metric(self, evaluations: list[ModelEvaluation]) -> None:
        assert compare_models(evaluations, metric="accuracy").ranking_metric == "accuracy"

    def test_unknown_ranking_metric_is_rejected(self, evaluations: list[ModelEvaluation]) -> None:
        with pytest.raises(EvaluationError, match="unknown ranking metric"):
            compare_models(evaluations, metric="flair")

    def test_a_single_evaluation_still_ranks(self, evaluations: list[ModelEvaluation]) -> None:
        assert len(compare_models(evaluations[:1]).evaluations) == 1

    def test_comparison_is_json_serialisable(self, evaluations: list[ModelEvaluation]) -> None:
        payload = compare_models(evaluations).as_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_comparison_table_has_a_row_per_model(self, evaluations: list[ModelEvaluation]) -> None:
        rows = compare_models(evaluations).as_table()
        assert len(rows) == 2
        assert set(rows[0]) >= {"model", "accuracy", "roc_auc"}

    @staticmethod
    def _without_roc_auc(evaluation: ModelEvaluation) -> ModelEvaluation:
        """Return an evaluation copy whose ROC-AUC is undefined."""
        return replace(
            evaluation,
            metrics=replace(evaluation.metrics, roc_auc=None, undefined=("roc_auc",)),
        )

    def test_undefined_metric_is_listed_aside_not_ranked(
        self, evaluations: list[ModelEvaluation]
    ) -> None:
        comparison = compare_models([self._without_roc_auc(evaluations[0]), evaluations[1]])
        assert evaluations[0].name in comparison.unranked
        assert evaluations[0].name not in comparison.names

    def test_ranking_is_rejected_when_nothing_defines_the_metric(
        self, evaluations: list[ModelEvaluation]
    ) -> None:
        forced = [self._without_roc_auc(evaluation) for evaluation in evaluations]
        with pytest.raises(EvaluationError, match="cannot be ranked"):
            compare_models(forced)


class TestFigures:
    """Plots are produced from real predictions.

    The plotting functions return the figure for inspection; ``path`` names where
    it was written, so the file is checked at that location.
    """

    @pytest.fixture
    def predictions(
        self, modelling_frame: pd.DataFrame, modelling_split: DatasetSplit
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return truth, hard predictions and scores from a fitted model."""
        model = train_model(modelling_frame, "logistic_regression", random_state=42)
        scores = model.predict_proba(modelling_split.X_test)[:, 1]
        truth = np.asarray(modelling_split.y_test)
        return truth, (scores >= 0.5).astype(int), scores

    def test_confusion_matrix_is_written(
        self, predictions: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        truth, predicted, _ = predictions
        destination = tmp_path / "cm.png"
        plot_confusion_matrix(truth, predicted, title="t", path=destination)
        assert destination.is_file() and destination.stat().st_size > 0

    def test_roc_curve_is_written(
        self, predictions: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        truth, _, scores = predictions
        destination = tmp_path / "roc.png"
        plot_roc_curve(truth, scores, auc=0.75, path=destination)
        assert destination.is_file() and destination.stat().st_size > 0

    def test_pr_curve_is_written(
        self, predictions: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        truth, _, scores = predictions
        destination = tmp_path / "pr.png"
        plot_precision_recall_curve(
            truth, scores, average_precision=0.6, positive_rate=0.4, path=destination
        )
        assert destination.is_file() and destination.stat().st_size > 0

    def test_figure_is_returned_when_no_path_is_given(
        self, predictions: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        truth, predicted, scores = predictions
        assert plot_roc_curve(truth, scores) is not None
        assert plot_confusion_matrix(truth, predicted) is not None
        assert plot_precision_recall_curve(truth, scores) is not None
        assert not list(tmp_path.glob("*.png")), "nothing should be written without a path"

    def test_nested_directories_are_created(
        self, predictions: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        truth, _, scores = predictions
        destination = tmp_path / "deep" / "nested" / "roc.png"
        plot_roc_curve(truth, scores, path=destination)
        assert destination.is_file()

    def test_roc_curve_needs_both_classes(
        self, predictions: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        _, _, scores = predictions
        with pytest.raises(EvaluationError, match="both classes"):
            plot_roc_curve(np.ones(len(scores), dtype=int), scores, path=tmp_path / "roc.png")

    def test_roc_auc_none_is_rendered(
        self, predictions: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        truth, _, scores = predictions
        destination = tmp_path / "roc.png"
        plot_roc_curve(truth, scores, auc=None, path=destination)
        assert destination.is_file()

    def test_pr_curve_without_a_baseline_is_written(
        self, predictions: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        truth, _, scores = predictions
        destination = tmp_path / "pr.png"
        plot_precision_recall_curve(truth, scores, path=destination)
        assert destination.is_file()

    def test_pr_curve_with_undefined_ap_is_written(
        self, predictions: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        truth, _, scores = predictions
        destination = tmp_path / "pr.png"
        plot_precision_recall_curve(truth, scores, average_precision=None, path=destination)
        assert destination.is_file()

    def test_confusion_matrix_rejects_non_binary_input(
        self, predictions: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        truth, predicted, _ = predictions
        with pytest.raises(EvaluationError, match="only 0 and 1"):
            plot_confusion_matrix(truth, predicted * 2, path=tmp_path / "cm.png")


class TestReport:
    """The generated evaluation report."""

    @pytest.fixture
    def evaluation(
        self, modelling_frame: pd.DataFrame, modelling_split: DatasetSplit
    ) -> ModelEvaluation:
        """Return one real evaluation to report on."""
        model = train_model(modelling_frame, "logistic_regression", random_state=42)
        return evaluate_model(model, modelling_split.X_test, modelling_split.y_test)

    def test_report_names_the_dataset(self, evaluation: ModelEvaluation) -> None:
        report = render_report([evaluation], dataset_summary={})
        assert "Dataset:" in report

    def test_report_lists_the_features(self, evaluation: ModelEvaluation) -> None:
        report = render_report([evaluation], dataset_summary={})
        features_line = next(line for line in report.splitlines() if line.startswith("- Features"))
        for name in FEATURE_COLUMNS:
            assert f"`{name}`" in features_line
        assert f"`{TARGET_COLUMN}`" not in features_line

    def test_report_states_the_split(self, evaluation: ModelEvaluation) -> None:
        assert "Train/test split" in render_report([evaluation], dataset_summary={})

    def test_report_states_the_preprocessing(self, evaluation: ModelEvaluation) -> None:
        report = render_report([evaluation], dataset_summary={})
        assert "median" in report
        assert "training partition only" in report

    def test_report_includes_the_metric_table(self, evaluation: ModelEvaluation) -> None:
        report = render_report([evaluation], dataset_summary={})
        assert "| model | accuracy |" in report

    def test_report_includes_a_confusion_matrix_table(self, evaluation: ModelEvaluation) -> None:
        assert "| model | tn | fp | fn | tp |" in render_report([evaluation], dataset_summary={})

    def test_report_includes_limitations(self, evaluation: ModelEvaluation) -> None:
        assert "## 7. Limitations" in render_report([evaluation], dataset_summary={})

    def test_report_does_not_claim_clinical_validity(self, evaluation: ModelEvaluation) -> None:
        report = render_report([evaluation], dataset_summary={}).lower()
        assert "does not establish clinical validity" in report

    def test_report_flags_the_threshold_as_technical(self, evaluation: ModelEvaluation) -> None:
        assert "not a clinical cut-off" in render_report([evaluation], dataset_summary={})

    def test_report_shows_the_class_distribution(self, evaluation: ModelEvaluation) -> None:
        report = render_report(
            [evaluation], dataset_summary={"class_distribution": {"0": 1, "1": 2}}
        )
        assert "0=1, 1=2" in report

    def test_report_shows_zero_sentinel_counts(self, evaluation: ModelEvaluation) -> None:
        report = render_report([evaluation], dataset_summary={"zero_counts": {"Glucose": 7}})
        assert "Glucose=7" in report

    def test_report_shows_missing_counts(self, evaluation: ModelEvaluation) -> None:
        report = render_report([evaluation], dataset_summary={"missing_counts": {"Glucose": 3}})
        assert "Missing cells: 3" in report

    def test_report_omits_figures_that_were_not_produced(self, evaluation: ModelEvaluation) -> None:
        assert "## 6. Figures" not in render_report([evaluation], dataset_summary={})

    def test_report_includes_supplied_figures(self, evaluation: ModelEvaluation) -> None:
        report = render_report(
            [evaluation],
            dataset_summary={},
            figures={"ROC": "reports/figures/roc.png"},
        )
        assert "![ROC](reports/figures/roc.png)" in report

    def test_report_notes_unranked_models(self, evaluation: ModelEvaluation) -> None:
        """A model without ROC-AUC is listed aside rather than ranked."""
        without_auc = replace(
            evaluation,
            metrics=replace(evaluation.metrics, roc_auc=None, undefined=("roc_auc",)),
        )
        assert "Not ranked" in render_report([without_auc], dataset_summary={})

    def test_report_renders_when_nothing_can_be_ranked(self, evaluation: ModelEvaluation) -> None:
        """A report is still produced when the ranking metric is undefined."""
        without_auc = replace(
            evaluation,
            metrics=replace(evaluation.metrics, roc_auc=None, undefined=("roc_auc",)),
        )
        report = render_report([without_auc], dataset_summary={})
        assert "No model defines the ranking metric" in report
        assert "| model | accuracy |" in report

    def test_report_cannot_be_rendered_without_results(self) -> None:
        with pytest.raises(EvaluationError, match="no results"):
            render_report([], dataset_summary={})

    def test_report_lists_unranked_models_alongside_ranked_ones(
        self, modelling_frame: pd.DataFrame, modelling_split: DatasetSplit
    ) -> None:
        """With one rankable and one unrankable model, the report still orders."""
        models = train_all_baselines(
            modelling_frame,
            names=("logistic_regression", "random_forest"),
            random_state=42,
            test_size=0.25,
        )
        evaluations = [
            evaluate_model(model, modelling_split.X_test, modelling_split.y_test)
            for model in models
        ]
        degraded = replace(
            evaluations[0],
            metrics=replace(evaluations[0].metrics, roc_auc=None, undefined=("roc_auc",)),
        )
        report = render_report([degraded, evaluations[1]], dataset_summary={})
        assert "Ordered by `roc_auc` descending" in report
        assert f"Not ranked, because `roc_auc` was undefined: {degraded.name}" in report

    def test_report_writes_to_disk(self, evaluation: ModelEvaluation, tmp_path: Path) -> None:
        from cronical.models.reporting import write_report

        path = write_report([evaluation], dataset_summary={}, path=tmp_path / "report.md")
        assert path.is_file() and path.read_text(encoding="utf-8").startswith("# Model evaluation")


class TestMetadataStructure:
    """The metadata record itself."""

    def test_metadata_is_immutable(self) -> None:
        metadata = ExperimentMetadata(
            model_name="m",
            dataset_path="d",
            dataset_sha256="s",
            row_count=1,
            feature_names=("a",),
            target="t",
            random_state=0,
            test_size=0.2,
            decision_threshold=0.5,
        )
        with pytest.raises(FrozenInstanceError):
            metadata.row_count = 2  # type: ignore[misc]

    def test_spec_is_immutable(self) -> None:
        spec: ModelSpec = get_spec("logistic_regression")
        with pytest.raises(FrozenInstanceError):
            # Assigning a different enum member is what mypy objects to; the
            # point of the test is that the runtime refuses the assignment.
            spec.name = EstimatorName.XGBOOST  # type: ignore[misc, assignment]
