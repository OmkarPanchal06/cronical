"""Tests for :mod:`cronical.data.preprocessing`.

Every value here comes from the synthetic rows in ``conftest.py``. No test loads
the real dataset, and no figure produced by these tests describes it.

Two themes run through the suite:

* **Nothing is mutated.** The raw frame, the frame handed to the pipeline and the
  raw CSV on disk all come out byte-for-byte unchanged.
* **Learned statistics come from training data only.** The leakage tests assert
  this against a dataset whose training median and whole-dataset median differ, so
  a regression that fitted on everything would be caught rather than hidden.
"""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from cronical.config import ScalingMode, Settings
from cronical.data.errors import (
    DatasetValidationError,
    FeatureContractError,
    PreprocessingError,
    SplitError,
)
from cronical.data.preprocessing import (
    DatasetSplit,
    FeatureContract,
    SentinelZeroHandler,
    build_preprocessor,
    build_training_pipeline,
    describe_preprocessor,
    dump_preprocessor,
    fit_preprocessor,
    load_preprocessor,
    save_preprocessor,
    split_dataset,
    transform_features,
    transform_patient,
)
from cronical.data.schema import FEATURE_COLUMNS, TARGET_COLUMN, ZERO_SENTINEL_COLUMNS
from cronical.data.validation import validate_dataframe

from .conftest import HEADER, PREPROCESSING_ROWS

TRAIN_ROWS = slice(0, 16)
HELD_OUT_ROWS = slice(16, 18)


def _post_imputation_mean(frame: pd.DataFrame) -> np.ndarray:
    """Return the per-feature mean after sentinel replacement and median filling.

    Mirrors the data the scaler actually sees, so a leakage assertion compares
    like with like.
    """
    from sklearn.impute import SimpleImputer

    sentinel_free = _sentinel_frame(frame)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    return np.asarray(imputer.fit_transform(sentinel_free), dtype=float).mean(axis=0)


def _sentinel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the sentinel handler, keeping the DataFrame return type explicit.

    The scikit-learn stubs declare ``TransformerMixin.fit_transform`` as returning
    an ``ndarray``, so the concrete type is restated here once instead of being
    cast at every call site.
    """
    return cast(pd.DataFrame, SentinelZeroHandler().fit_transform(frame))


class TestFixtureSanity:
    """The shared fixture must be usable, or every other test is meaningless."""

    def test_dataset_passes_validation(self, dataset_frame: pd.DataFrame) -> None:
        report = validate_dataframe(dataset_frame)
        assert report.is_valid, [issue.message for issue in report.errors]

    def test_dataset_exercises_the_zero_sentinels(self, dataset_frame: pd.DataFrame) -> None:
        report = validate_dataframe(dataset_frame)
        assert set(report.zero_counts) == set(ZERO_SENTINEL_COLUMNS)
        assert all(count > 0 for count in report.zero_counts.values())

    def test_dataset_also_carries_genuine_missing_values(self, dataset_frame: pd.DataFrame) -> None:
        assert int(dataset_frame.isna().sum().sum()) > 0

    def test_header_agrees_with_the_schema(self) -> None:
        assert HEADER.split(",") == [*FEATURE_COLUMNS, TARGET_COLUMN]


class TestSentinelZeroHandling:
    """Zero sentinels become missing values; real zeros do not."""

    @pytest.mark.parametrize("column", ZERO_SENTINEL_COLUMNS)
    def test_zero_becomes_nan_in_each_sentinel_column(self, column: str) -> None:
        frame = pd.DataFrame([[1, 5, 5, 5, 5, 5.0, 0.5, 30]], columns=list(FEATURE_COLUMNS))
        frame.loc[0, column] = 0
        handler = SentinelZeroHandler().fit(frame)
        assert math.isnan(float(handler.transform(frame)[column].iloc[0]))

    def test_zero_in_pregnancies_stays_zero(self, features_frame: pd.DataFrame) -> None:
        frame = features_frame.copy()
        frame.loc[frame.index[0], "Pregnancies"] = 0
        transformed = _sentinel_frame(frame)
        assert float(transformed["Pregnancies"].iloc[0]) == 0.0

    def test_zero_in_pedigree_function_stays_zero(self, features_frame: pd.DataFrame) -> None:
        frame = features_frame.copy()
        frame.loc[frame.index[0], "DiabetesPedigreeFunction"] = 0.0
        transformed = _sentinel_frame(frame)
        assert float(transformed["DiabetesPedigreeFunction"].iloc[0]) == 0.0

    def test_only_the_documented_columns_are_touched(self, features_frame: pd.DataFrame) -> None:
        frame = features_frame.copy()
        for column in FEATURE_COLUMNS:
            frame.loc[frame.index[0], column] = 0
        transformed = _sentinel_frame(frame)
        for column in ZERO_SENTINEL_COLUMNS:
            assert math.isnan(float(transformed[column].iloc[0])), column
        for column in ("Pregnancies", "DiabetesPedigreeFunction", "Age"):
            assert float(transformed[column].iloc[0]) == 0.0, column

    def test_only_exactly_zero_is_replaced(self, features_frame: pd.DataFrame) -> None:
        """A small but non-zero measurement must survive untouched."""
        frame = features_frame.copy()
        frame.loc[frame.index[0], "Insulin"] = 5
        transformed = _sentinel_frame(frame)
        assert float(transformed["Insulin"].iloc[0]) == 5.0

    def test_sentinel_handler_rejects_a_missing_column(self) -> None:
        frame = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "b", "c", "d"])
        with pytest.raises(FeatureContractError, match="not present"):
            SentinelZeroHandler().fit(frame)

    def test_transform_before_fit_is_rejected(self, features_frame: pd.DataFrame) -> None:
        with pytest.raises(FeatureContractError, match="before fit"):
            SentinelZeroHandler().transform(features_frame)

    def test_sentinel_columns_are_configurable(self, features_frame: pd.DataFrame) -> None:
        handler = SentinelZeroHandler(columns=("Insulin",)).fit(features_frame)
        frame = features_frame.copy()
        frame.loc[frame.index[0], "Glucose"] = 0
        frame.loc[frame.index[0], "Insulin"] = 0
        transformed = handler.transform(frame)
        assert float(transformed["Glucose"].iloc[0]) == 0.0
        assert math.isnan(float(transformed["Insulin"].iloc[0]))


class TestOutcomeIsNeverTransformed:
    """The target must not be a feature, nor be altered."""

    def test_outcome_column_is_absent_from_the_output(self, features_frame: pd.DataFrame) -> None:
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        transformed = transform_features(pipeline, features_frame)
        assert TARGET_COLUMN not in transformed.columns

    def test_pipeline_can_be_fed_a_frame_that_contains_the_target(
        self, dataset_frame: pd.DataFrame
    ) -> None:
        pipeline = fit_preprocessor(dataset_frame, scaling=ScalingMode.NONE)
        transformed = transform_features(pipeline, dataset_frame)
        assert TARGET_COLUMN not in transformed.columns

    def test_target_values_are_never_modified(self, dataset_frame: pd.DataFrame) -> None:
        before = dataset_frame[TARGET_COLUMN].tolist()
        split_dataset(dataset_frame)
        fit_preprocessor(dataset_frame, scaling=ScalingMode.NONE)
        assert dataset_frame[TARGET_COLUMN].tolist() == before

    def test_passing_the_target_to_fit_changes_nothing(self, dataset_frame: pd.DataFrame) -> None:
        features = dataset_frame.loc[:, list(FEATURE_COLUMNS)]
        target = dataset_frame[TARGET_COLUMN]
        without_y = fit_preprocessor(features, scaling=ScalingMode.NONE)
        with_y = fit_preprocessor(features, y_train=target, scaling=ScalingMode.NONE)
        assert np.allclose(
            without_y.named_steps["imputer"].statistics_,
            with_y.named_steps["imputer"].statistics_,
        ), "the target must not influence any fitted statistic"


class TestImputation:
    """Missing values are filled from the training partition."""

    def test_no_missing_values_survive_the_pipeline(self, dataset_frame: pd.DataFrame) -> None:
        pipeline = fit_preprocessor(dataset_frame, scaling=ScalingMode.NONE)
        transformed = transform_features(pipeline, dataset_frame)
        assert int(transformed.isna().sum().sum()) == 0

    def test_pipeline_copes_with_gaps_already_in_the_input(
        self, features_frame: pd.DataFrame
    ) -> None:
        assert int(features_frame.isna().sum().sum()) > 0
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        assert not transform_features(pipeline, features_frame).isna().any().any()

    def test_imputation_uses_the_configured_median_strategy(
        self, features_frame: pd.DataFrame
    ) -> None:
        pipeline = fit_preprocessor(features_frame, imputation="median")
        assert pipeline.named_steps["imputer"].get_params()["strategy"] == "median"

    def test_mean_strategy_can_be_selected(self, features_frame: pd.DataFrame) -> None:
        pipeline = fit_preprocessor(features_frame, imputation="mean")
        assert pipeline.named_steps["imputer"].get_params()["strategy"] == "mean"

    def test_imputed_value_equals_the_training_median(self, dataset_frame: pd.DataFrame) -> None:
        features = dataset_frame.loc[:, list(FEATURE_COLUMNS)]
        pipeline = fit_preprocessor(features, scaling=ScalingMode.NONE)
        expected = float(_sentinel_frame(features)["Insulin"].median(skipna=True))
        fitted = pipeline.named_steps["imputer"].statistics_[FEATURE_COLUMNS.index("Insulin")]
        assert float(fitted) == expected

    def test_a_zero_sentinel_row_is_imputed_not_treated_as_zero(
        self, dataset_frame: pd.DataFrame
    ) -> None:
        """The all-zero row must receive the median, proving 0 became missing."""
        features = dataset_frame.loc[:, list(FEATURE_COLUMNS)]
        pipeline = fit_preprocessor(features, scaling=ScalingMode.NONE)
        transformed = transform_features(pipeline, features)
        transformed.index = features.index
        all_zero_row = features.index[12]
        assert float(transformed.loc[all_zero_row, "Glucose"]) > 0.0

    def test_all_missing_column_keeps_the_feature_count(self, features_frame: pd.DataFrame) -> None:
        """An unusable column must not silently change the output width."""
        degenerate = features_frame.copy()
        degenerate["BMI"] = math.nan
        pipeline = fit_preprocessor(degenerate, scaling=ScalingMode.NONE)
        transformed = transform_features(pipeline, degenerate)
        assert transformed.shape[1] == len(FEATURE_COLUMNS)


class TestLeakagePrevention:
    """Statistics must be learned from training data only."""

    @staticmethod
    def _post_sentinel_median(frame: pd.DataFrame, column: str) -> float:
        """Return a column's median after sentinel replacement, ignoring gaps."""
        replaced = _sentinel_frame(frame)[column]
        return float(replaced.median(skipna=True))

    def test_imputer_uses_the_training_median_not_the_whole_dataset_median(
        self, dataset_frame: pd.DataFrame
    ) -> None:
        train = dataset_frame.iloc[TRAIN_ROWS].loc[:, list(FEATURE_COLUMNS)]
        whole = dataset_frame.loc[:, list(FEATURE_COLUMNS)]

        training_median = self._post_sentinel_median(train, "Glucose")
        whole_median = self._post_sentinel_median(whole, "Glucose")
        assert training_median != whole_median, "fixture must separate the two medians"

        pipeline = fit_preprocessor(train, scaling=ScalingMode.NONE)
        fitted = float(
            pipeline.named_steps["imputer"].statistics_[FEATURE_COLUMNS.index("Glucose")]
        )
        assert fitted == training_median
        assert fitted != whole_median, "held-out rows must not reach the imputer"

    def test_scaler_uses_training_mean_not_the_whole_dataset_mean(
        self, dataset_frame: pd.DataFrame
    ) -> None:
        train = dataset_frame.iloc[TRAIN_ROWS].loc[:, list(FEATURE_COLUMNS)]
        whole = dataset_frame.loc[:, list(FEATURE_COLUMNS)]

        pipeline = fit_preprocessor(train, scaling=ScalingMode.STANDARD)
        mean = np.asarray(pipeline.named_steps["scaler"].mean_, dtype=float)

        # The scaler is fitted on data that has already been through the earlier
        # steps, so the expectation is computed the same way.
        assert np.allclose(mean, _post_imputation_mean(train))
        assert not np.allclose(mean, _post_imputation_mean(whole))

    def test_held_out_rows_do_not_change_fitted_statistics(
        self, dataset_frame: pd.DataFrame
    ) -> None:
        train = dataset_frame.iloc[TRAIN_ROWS].loc[:, list(FEATURE_COLUMNS)]
        extremes = dataset_frame.iloc[HELD_OUT_ROWS].loc[:, list(FEATURE_COLUMNS)]
        baseline = fit_preprocessor(train, scaling=ScalingMode.STANDARD)
        augmented = fit_preprocessor(pd.concat([train, extremes]), scaling=ScalingMode.STANDARD)
        assert not np.allclose(
            baseline.named_steps["imputer"].statistics_,
            augmented.named_steps["imputer"].statistics_,
            equal_nan=True,
        )

    def test_scaler_fitted_on_train_centres_the_training_partition(
        self, dataset_frame: pd.DataFrame
    ) -> None:
        split = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        pipeline = fit_preprocessor(split.X_train, scaling=ScalingMode.STANDARD)
        transformed = transform_features(pipeline, split.X_train)
        assert np.allclose(transformed.mean().to_numpy(), 0.0, atol=1e-9)

    def test_scaler_fitted_on_train_does_not_centre_the_test_partition(
        self, dataset_frame: pd.DataFrame
    ) -> None:
        """Evidence the test partition never influenced the fitted parameters."""
        split = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        pipeline = fit_preprocessor(split.X_train, scaling=ScalingMode.STANDARD)
        transformed = transform_features(pipeline, split.X_test)
        assert not np.allclose(transformed.mean().to_numpy(), 0.0, atol=1e-9)

    def test_split_partitions_are_disjoint_and_exhaustive(
        self, dataset_frame: pd.DataFrame
    ) -> None:
        split = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        train = set(split.X_train.index)
        test = set(split.X_test.index)
        assert not train & test, "a row appeared in both partitions"
        assert train | test == set(dataset_frame.index), "a row was dropped"

    def test_split_keeps_targets_aligned_with_their_rows(self, dataset_frame: pd.DataFrame) -> None:
        split = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        assert list(split.y_train.index) == list(split.X_train.index)
        assert (
            split.y_train.tolist() == split.X_train.index.map(dataset_frame[TARGET_COLUMN]).tolist()
        )


class TestScaling:
    """Scaling is applied only where it is meaningful."""

    def test_standard_mode_centres_and_scales(self, features_frame: pd.DataFrame) -> None:
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.STANDARD)
        transformed = transform_features(pipeline, features_frame)
        assert np.allclose(transformed.mean().to_numpy(), 0.0, atol=1e-9)
        # StandardScaler uses the population standard deviation (ddof=0).
        assert np.allclose(transformed.std(ddof=0).to_numpy(), 1.0, atol=1e-9)

    def test_no_scaling_mode_preserves_relative_magnitudes(
        self, features_frame: pd.DataFrame
    ) -> None:
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        transformed = transform_features(pipeline, features_frame)
        assert "scaler" not in pipeline.named_steps
        assert float(transformed["Glucose"].median()) > 100.0

    def test_both_paths_share_every_other_step(self, features_frame: pd.DataFrame) -> None:
        scaled = build_preprocessor(scaling=ScalingMode.STANDARD)
        unscaled = build_preprocessor(scaling=ScalingMode.NONE)
        assert list(scaled.named_steps)[:3] == list(unscaled.named_steps)[:3]
        assert "scaler" in scaled.named_steps
        assert "scaler" not in unscaled.named_steps

    def test_describe_reports_the_scaling_mode(self, features_frame: pd.DataFrame) -> None:
        unscaled = describe_preprocessor(fit_preprocessor(features_frame, scaling=ScalingMode.NONE))
        scaled = describe_preprocessor(
            fit_preprocessor(features_frame, scaling=ScalingMode.STANDARD)
        )
        assert unscaled["scaling"] == ScalingMode.NONE.value
        assert scaled["scaling"] == ScalingMode.STANDARD.value
        assert "scaler_mean" not in unscaled
        assert "scaler_mean" in scaled


class TestFeatureOrdering:
    """Output order is fixed and matches the schema."""

    def test_output_columns_match_the_schema_order(self, features_frame: pd.DataFrame) -> None:
        pipeline = fit_preprocessor(features_frame)
        assert tuple(transform_features(pipeline, features_frame).columns) == FEATURE_COLUMNS

    def test_ordering_is_independent_of_input_column_order(
        self, features_frame: pd.DataFrame
    ) -> None:
        pipeline = fit_preprocessor(features_frame)
        shuffled = features_frame.loc[:, list(reversed(FEATURE_COLUMNS))]
        assert tuple(transform_features(pipeline, shuffled).columns) == FEATURE_COLUMNS

    def test_reordering_does_not_change_the_values(self, features_frame: pd.DataFrame) -> None:
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        shuffled = features_frame.loc[:, list(reversed(FEATURE_COLUMNS))]
        pd.testing.assert_frame_equal(
            transform_features(pipeline, shuffled),
            transform_features(pipeline, features_frame),
        )

    def test_transformed_feature_count_is_correct(self, features_frame: pd.DataFrame) -> None:
        pipeline = fit_preprocessor(features_frame)
        transformed = transform_features(pipeline, features_frame)
        assert transformed.shape == (len(features_frame.index), len(FEATURE_COLUMNS))

    def test_transformed_values_are_floats(self, features_frame: pd.DataFrame) -> None:
        transformed = transform_features(fit_preprocessor(features_frame), features_frame)
        assert all(dtype == np.float64 for dtype in transformed.dtypes)


class TestSplit:
    """Splitting is stratified, deterministic and configurable."""

    def test_split_is_deterministic_for_a_fixed_seed(self, dataset_frame: pd.DataFrame) -> None:
        first = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        second = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        assert list(first.X_train.index) == list(second.X_train.index)
        assert list(first.X_test.index) == list(second.X_test.index)
        assert first.y_train.tolist() == second.y_train.tolist()

    def test_a_different_seed_produces_a_different_partition(
        self, dataset_frame: pd.DataFrame
    ) -> None:
        first = split_dataset(dataset_frame, test_size=0.25, random_state=1)
        second = split_dataset(dataset_frame, test_size=0.25, random_state=2)
        assert set(first.X_test.index) != set(second.X_test.index)

    def test_seed_defaults_to_settings(self, dataset_frame: pd.DataFrame) -> None:
        from cronical.config import get_settings

        explicit = split_dataset(
            dataset_frame,
            test_size=0.25,
            random_state=get_settings().random_seed,
        )
        default = split_dataset(dataset_frame, test_size=0.25)
        assert list(default.X_test.index) == list(explicit.X_test.index)

    @pytest.mark.parametrize("test_size", [0.25, 0.5])
    def test_test_size_is_configurable(self, dataset_frame: pd.DataFrame, test_size: float) -> None:
        split = split_dataset(dataset_frame, test_size=test_size, random_state=3)
        # scikit-learn rounds the held-out count up.
        assert split.test_size == math.ceil(len(dataset_frame.index) * test_size)
        assert split.train_size + split.test_size == len(dataset_frame.index)

    def test_test_size_defaults_to_settings(
        self, dataset_frame: pd.DataFrame, settings: Settings
    ) -> None:
        configured = Settings.for_root(settings.project_root)
        split = split_dataset(dataset_frame, settings=configured)
        assert split.test_size == math.ceil(
            len(dataset_frame.index) * configured.preprocessing.test_size
        )

    def test_stratification_is_preserved(self, dataset_frame: pd.DataFrame) -> None:
        split = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        overall = dataset_frame[TARGET_COLUMN].value_counts(normalize=True)
        for partition in (split.y_train, split.y_test):
            observed = partition.value_counts(normalize=True)
            for value, share in overall.items():
                assert abs(observed[value] - share) < 0.1, value

    def test_class_balance_reports_both_partitions(self, dataset_frame: pd.DataFrame) -> None:
        balance = split_dataset(dataset_frame, test_size=0.25, random_state=7).class_balance()
        assert set(balance) == {"train", "test"}
        assert all(isinstance(count, int) for part in balance.values() for count in part.values())

    def test_targets_stay_binary(self, dataset_frame: pd.DataFrame) -> None:
        split = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        for target in (split.y_train, split.y_test):
            assert set(target.unique()) <= {0, 1}

    def test_row_indices_are_preserved(self, dataset_frame: pd.DataFrame) -> None:
        split = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        assert set(split.X_train.index) <= set(dataset_frame.index)
        assert set(split.y_test.index) == set(split.X_test.index)

    def test_features_exclude_the_target(self, dataset_frame: pd.DataFrame) -> None:
        split = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        assert tuple(split.X_train.columns) == FEATURE_COLUMNS

    def test_sizes_reports_both_partitions(self, dataset_frame: pd.DataFrame) -> None:
        split = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        assert split.sizes() == {"train": split.train_size, "test": split.test_size}

    def test_feature_names_are_exposed(self, dataset_frame: pd.DataFrame) -> None:
        split = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        assert split.feature_names == FEATURE_COLUMNS

    def test_a_rare_class_is_rejected(self, features_frame: pd.DataFrame) -> None:
        frame = features_frame.copy()
        frame[TARGET_COLUMN] = [1] * len(frame.index)
        frame.loc[frame.index[0], TARGET_COLUMN] = 0
        with pytest.raises(SplitError, match="stratified"):
            split_dataset(frame, test_size=0.25, random_state=7)

    def test_a_test_fraction_that_cannot_cover_both_classes_is_rejected(
        self, features_frame: pd.DataFrame
    ) -> None:
        """Both classes are common enough to stratify, but the hold-out is not."""
        frame = features_frame.copy()
        frame[TARGET_COLUMN] = [index % 2 for index in range(len(frame.index))]
        # 18 rows * 0.05 rounds up to 1 held-out row, which cannot hold both classes.
        with pytest.raises(SplitError, match="increase test_size"):
            split_dataset(frame, test_size=0.05, random_state=7)

    def test_a_non_binary_target_is_rejected(self, features_frame: pd.DataFrame) -> None:
        frame = features_frame.copy()
        frame[TARGET_COLUMN] = 5
        with pytest.raises(DatasetValidationError):
            split_dataset(frame, test_size=0.25, random_state=7)

    def test_a_missing_feature_is_rejected(self, features_frame: pd.DataFrame) -> None:
        frame = features_frame.copy()
        frame[TARGET_COLUMN] = [0, 1] * (len(frame.index) // 2)
        with pytest.raises(DatasetValidationError):
            split_dataset(frame.drop(columns=["Insulin"]), test_size=0.25)

    def test_split_is_immutable(self, dataset_frame: pd.DataFrame) -> None:
        split = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        with pytest.raises(FrozenInstanceError):
            split.X_train = pd.DataFrame()  # type: ignore[misc]


class _Unconvertible:
    """Stands in for input that numpy refuses to coerce into an array."""

    def __array__(self, dtype: object = None, copy: object = None) -> object:
        """Fail conversion, so the guard around ``np.asarray`` is exercised."""
        raise ValueError("cannot convert to an array")


class TestInvalidInput:
    """Bad input fails loudly with a typed error."""

    def test_input_numpy_cannot_convert_is_rejected(self, features_frame: pd.DataFrame) -> None:
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        with pytest.raises(FeatureContractError, match="unsupported input type"):
            transform_features(pipeline, _Unconvertible())

    def test_missing_feature_column_is_rejected(self, features_frame: pd.DataFrame) -> None:
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        with pytest.raises(FeatureContractError, match="missing required feature"):
            transform_features(pipeline, features_frame.drop(columns=["Insulin"]))

    def test_non_numeric_feature_is_rejected(self) -> None:
        frame = pd.DataFrame(
            [[1, "high", 70, 20, 90, 25.0, 0.4, 30]], columns=list(FEATURE_COLUMNS)
        )
        with pytest.raises(FeatureContractError, match="must be numeric"):
            transform_features(fit_preprocessor(frame, scaling=ScalingMode.NONE), frame)

    def test_wrong_width_array_is_rejected(self, features_frame: pd.DataFrame) -> None:
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        with pytest.raises(FeatureContractError, match="expected an array"):
            transform_features(pipeline, np.zeros((2, 3)))

    def test_unusable_input_is_rejected(self, features_frame: pd.DataFrame) -> None:
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        with pytest.raises(FeatureContractError):
            transform_features(pipeline, object())

    def test_contract_transform_before_fit_is_rejected(self, features_frame: pd.DataFrame) -> None:
        with pytest.raises(FeatureContractError, match="before fit"):
            FeatureContract().transform(features_frame)

    def test_describe_rejects_an_unfitted_pipeline(self) -> None:
        with pytest.raises(PreprocessingError, match="not been fitted"):
            describe_preprocessor(build_preprocessor())

    def test_describe_rejects_a_pipeline_without_an_imputer(
        self, features_frame: pd.DataFrame
    ) -> None:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer

        malformed = Pipeline([("noop", FunctionTransformer(feature_names_out="one-to-one"))])
        malformed.fit(features_frame)
        with pytest.raises(PreprocessingError, match="imputation step"):
            describe_preprocessor(malformed)

    def test_load_rejects_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(PreprocessingError, match="no saved preprocessor"):
            load_preprocessor(path=tmp_path / "absent.joblib")


class TestInference:
    """One patient record travels the same fitted pipeline."""

    def test_a_single_row_array_is_promoted_to_two_dimensions(
        self, features_frame: pd.DataFrame
    ) -> None:
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        row = features_frame.iloc[0].to_numpy()
        assert transform_features(pipeline, row).shape == (1, len(FEATURE_COLUMNS))

    def test_patient_mapping_is_transformed(
        self, features_frame: pd.DataFrame, patient_record: dict[str, float]
    ) -> None:
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        transformed = transform_patient(pipeline, patient_record)
        assert transformed.shape == (1, len(FEATURE_COLUMNS))
        assert tuple(transformed.columns) == FEATURE_COLUMNS

    def test_patient_with_a_zero_sentinel_is_imputed(
        self, features_frame: pd.DataFrame, patient_record: dict[str, float]
    ) -> None:
        assert patient_record["Glucose"] == 0
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        transformed = transform_patient(pipeline, patient_record)
        assert float(transformed["Glucose"].iloc[0]) > 0.0

    def test_patient_given_as_a_series(
        self, features_frame: pd.DataFrame, patient_record: dict[str, float]
    ) -> None:
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        as_series = pd.Series(patient_record)
        assert transform_patient(pipeline, as_series).shape == (1, len(FEATURE_COLUMNS))

    def test_patient_may_be_given_positionally(
        self, features_frame: pd.DataFrame, patient_record: dict[str, float]
    ) -> None:
        """Positional rows go through the general entry point."""
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        values = np.array([[patient_record[name] for name in FEATURE_COLUMNS]])
        transformed = transform_features(pipeline, values)
        assert transformed.shape == (1, len(FEATURE_COLUMNS))

    def test_unseen_rows_transform_identically(self, dataset_frame: pd.DataFrame) -> None:
        split = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        pipeline = fit_preprocessor(split.X_train, scaling=ScalingMode.STANDARD)
        batch = transform_features(pipeline, split.X_test)
        one_at_a_time = pd.concat(
            [
                transform_patient(pipeline, record)
                for record in split.X_test.to_dict(orient="records")
            ]
        )
        one_at_a_time.index = batch.index
        pd.testing.assert_frame_equal(batch, one_at_a_time)

    def test_inference_reuses_the_fitted_statistics(
        self, features_frame: pd.DataFrame, patient_record: dict[str, float]
    ) -> None:
        """A second call must not refit anything."""
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        first = transform_patient(pipeline, patient_record)
        second = transform_patient(pipeline, patient_record)
        statistics = pipeline.named_steps["imputer"].statistics_
        pd.testing.assert_frame_equal(first, second)
        assert np.allclose(pipeline.named_steps["imputer"].statistics_, statistics, equal_nan=True)

    def test_patient_missing_a_feature_is_rejected(
        self, features_frame: pd.DataFrame, patient_record: dict[str, float]
    ) -> None:
        incomplete = {k: v for k, v in patient_record.items() if k != "BMI"}
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        with pytest.raises(FeatureContractError, match="missing required feature"):
            transform_patient(pipeline, incomplete)


class TestNothingIsMutated:
    """Inputs come out unchanged."""

    def test_raw_frame_survives_split_and_fit(self, dataset_frame: pd.DataFrame) -> None:
        before = dataset_frame.copy(deep=True)
        split = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        fit_preprocessor(split.X_train, scaling=ScalingMode.STANDARD)
        transform_features(fit_preprocessor(split.X_train), split.X_test)
        pd.testing.assert_frame_equal(dataset_frame, before)

    def test_feature_frame_survives_transform(self, features_frame: pd.DataFrame) -> None:
        before = features_frame.copy(deep=True)
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.STANDARD)
        transform_features(pipeline, features_frame)
        pd.testing.assert_frame_equal(features_frame, before)

    def test_sentinel_handler_survives_transform(self, features_frame: pd.DataFrame) -> None:
        before = features_frame.copy(deep=True)
        _sentinel_frame(features_frame)
        pd.testing.assert_frame_equal(features_frame, before)

    def test_split_partitions_are_copies_not_views(self, dataset_frame: pd.DataFrame) -> None:
        split = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        transformed = transform_features(fit_preprocessor(split.X_train), split.X_train)
        transformed.iloc[0, 0] = -999.0
        assert dataset_frame.loc[split.X_train.index[0], FEATURE_COLUMNS[0]] != -999.0


class TestSerialization:
    """The fitted pipeline survives a round trip."""

    def test_round_trip_reproduces_output(
        self, features_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.STANDARD)
        saved = save_preprocessor(pipeline, path=tmp_path / "preprocessor.joblib")
        reloaded = load_preprocessor(path=saved)
        pd.testing.assert_frame_equal(
            transform_features(pipeline, features_frame),
            transform_features(reloaded, features_frame),
        )

    def test_save_returns_the_written_path(
        self, features_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        destination = tmp_path / "nested" / "preprocessor.joblib"
        assert save_preprocessor(pipeline, path=destination) == destination
        assert destination.is_file(), "parent directories should be created"

    def test_save_creates_the_configured_directory(
        self, features_frame: pd.DataFrame, isolated_project_root: Path
    ) -> None:
        configured = Settings.for_root(isolated_project_root)
        pipeline = fit_preprocessor(features_frame, settings=configured)
        written = save_preprocessor(pipeline, settings=configured)
        assert written == configured.preprocessor_path
        assert written.parent == isolated_project_root / "models"

    def test_save_rejects_an_unfitted_pipeline(self) -> None:
        with pytest.raises(PreprocessingError):
            save_preprocessor(build_preprocessor(), path=Path("unused.joblib"))

    def test_load_uses_the_configured_path(
        self, features_frame: pd.DataFrame, isolated_project_root: Path
    ) -> None:
        configured = Settings.for_root(isolated_project_root)
        pipeline = fit_preprocessor(features_frame, settings=configured)
        save_preprocessor(pipeline, settings=configured)
        assert load_preprocessor(settings=configured) is not None

    def test_dump_writes_a_json_description(
        self, features_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.STANDARD)
        destination = tmp_path / "description.json"
        written = dump_preprocessor(pipeline, destination)
        assert json.loads(written)["scaling"] == ScalingMode.STANDARD.value
        assert json.loads(destination.read_text(encoding="utf-8")) == json.loads(written)


class TestDescribePreprocessor:
    """The audit trail of a fitted pipeline."""

    def test_reports_every_feature(self, features_frame: pd.DataFrame) -> None:
        info = describe_preprocessor(fit_preprocessor(features_frame))
        assert set(info["imputation_statistics"]) == set(FEATURE_COLUMNS)

    def test_reports_the_feature_order(self, features_frame: pd.DataFrame) -> None:
        info = describe_preprocessor(fit_preprocessor(features_frame))
        assert info["feature_order"] == list(FEATURE_COLUMNS)

    def test_reports_the_sentinel_columns(self, features_frame: pd.DataFrame) -> None:
        info = describe_preprocessor(fit_preprocessor(features_frame))
        assert info["sentinel_columns"] == list(ZERO_SENTINEL_COLUMNS)

    def test_reports_the_imputation_strategy(self, features_frame: pd.DataFrame) -> None:
        info = describe_preprocessor(fit_preprocessor(features_frame, imputation="mean"))
        assert info["imputation_strategy"] == "mean"

    def test_is_json_serialisable(self, features_frame: pd.DataFrame) -> None:
        info = describe_preprocessor(fit_preprocessor(features_frame))
        assert json.loads(json.dumps(info)) == info

    def test_reports_scaler_parameters(self, features_frame: pd.DataFrame) -> None:
        info = describe_preprocessor(fit_preprocessor(features_frame))
        assert len(info["scaler_mean"]) == len(FEATURE_COLUMNS)
        assert len(info["scaler_scale"]) == len(FEATURE_COLUMNS)

    def test_an_unusable_column_is_reported_as_a_finite_fill(
        self, features_frame: pd.DataFrame
    ) -> None:
        """keep_empty_features fills rather than drops, so the summary stays valid JSON."""
        degenerate = features_frame.copy()
        degenerate["BMI"] = math.nan
        info = describe_preprocessor(fit_preprocessor(degenerate, scaling=ScalingMode.NONE))
        value = info["imputation_statistics"]["BMI"]
        assert math.isfinite(float(value))
        assert json.loads(json.dumps(info)) == info

    def test_every_fitted_statistic_is_finite(self, features_frame: pd.DataFrame) -> None:
        info = describe_preprocessor(fit_preprocessor(features_frame))
        assert all(math.isfinite(float(value)) for value in info["imputation_statistics"].values())


class TestSettingsIntegration:
    """Configuration drives behaviour; nothing is hard-coded."""

    def test_default_scaling_comes_from_settings(
        self, features_frame: pd.DataFrame, isolated_project_root: Path
    ) -> None:
        configured = Settings.for_root(isolated_project_root)
        configured = configured.model_copy(
            update={
                "preprocessing": configured.preprocessing.model_copy(
                    update={"default_scaling": ScalingMode.NONE}
                )
            }
        )
        pipeline = build_preprocessor(settings=configured)
        assert "scaler" not in pipeline.named_steps

    def test_default_imputation_comes_from_settings(
        self, features_frame: pd.DataFrame, isolated_project_root: Path
    ) -> None:
        configured = Settings.for_root(isolated_project_root)
        configured = configured.model_copy(
            update={
                "preprocessing": configured.preprocessing.model_copy(
                    update={"imputation_strategy": "most_frequent"}
                )
            }
        )
        pipeline = build_preprocessor(settings=configured)
        assert pipeline.named_steps["imputer"].get_params()["strategy"] == "most_frequent"

    def test_explicit_arguments_override_settings(self, isolated_project_root: Path) -> None:
        configured = Settings.for_root(isolated_project_root)
        configured = configured.model_copy(
            update={
                "preprocessing": configured.preprocessing.model_copy(
                    update={"default_scaling": ScalingMode.NONE}
                )
            }
        )
        assert (
            "scaler"
            in build_preprocessor(scaling=ScalingMode.STANDARD, settings=configured).named_steps
        )

    def test_string_scaling_is_accepted(self) -> None:
        assert "scaler" in build_preprocessor(scaling="standard").named_steps

    def test_string_imputation_is_accepted(self) -> None:
        assert (
            build_preprocessor(imputation="mean").named_steps["imputer"].get_params()["strategy"]
            == "mean"
        )

    def test_preprocessor_path_is_derived_from_settings(self, isolated_project_root: Path) -> None:
        configured = Settings.for_root(isolated_project_root)
        assert configured.preprocessor_path.name == "preprocessor.joblib"
        assert configured.preprocessor_path.parent == configured.paths.models_dir

    def test_custom_artifact_name_is_honoured(self, isolated_project_root: Path) -> None:
        configured = Settings.for_root(isolated_project_root)
        configured = configured.model_copy(
            update={
                "preprocessing": configured.preprocessing.model_copy(
                    update={"artifact_name": "linear_prep.joblib"}
                )
            }
        )
        assert configured.preprocessor_path.name == "linear_prep.joblib"


class TestTrainingPipelineComposition:
    """The preprocessing steps can be composed with a final estimator."""

    def test_passthrough_estimator_composes(self, features_frame: pd.DataFrame) -> None:
        pipeline = build_training_pipeline("passthrough", scaling=ScalingMode.NONE)
        pipeline.fit(features_frame)
        assert np.asarray(pipeline.transform(features_frame)).shape == (
            len(features_frame.index),
            len(FEATURE_COLUMNS),
        )

    def test_scaling_reaches_the_composed_pipeline(self, features_frame: pd.DataFrame) -> None:
        pipeline = build_training_pipeline("passthrough", scaling=ScalingMode.STANDARD)
        pipeline.fit(features_frame)
        transformed = pd.DataFrame(pipeline.transform(features_frame))
        assert np.allclose(transformed.mean().to_numpy(), 0.0, atol=1e-9)

    def test_settings_apply_to_the_composed_pipeline(self, isolated_project_root: Path) -> None:
        configured = Settings.for_root(isolated_project_root)
        configured = configured.model_copy(
            update={
                "preprocessing": configured.preprocessing.model_copy(
                    update={"default_scaling": ScalingMode.NONE}
                )
            }
        )
        pipeline = build_training_pipeline("passthrough", settings=configured)
        assert "scaler" not in pipeline.named_steps

    def test_estimator_is_last(self, features_frame: pd.DataFrame) -> None:
        pipeline = build_training_pipeline("passthrough", scaling=ScalingMode.NONE)
        assert list(pipeline.steps)[-1][0] == "estimator"


class TestPimaSchema:
    """The exact published header is handled end to end."""

    def test_header_order_from_the_documented_schema_is_accepted(
        self, features_frame: pd.DataFrame
    ) -> None:
        reordered = pd.DataFrame(features_frame.to_numpy(), columns=list(reversed(FEATURE_COLUMNS)))
        transformed = transform_features(
            fit_preprocessor(reordered, scaling=ScalingMode.NONE), reordered
        )
        assert tuple(transformed.columns) == FEATURE_COLUMNS

    def test_all_eight_documented_features_are_produced(self, features_frame: pd.DataFrame) -> None:
        transformed = transform_features(
            fit_preprocessor(features_frame, scaling=ScalingMode.NONE), features_frame
        )
        assert set(transformed.columns) == set(FEATURE_COLUMNS)
        assert len(FEATURE_COLUMNS) == 8

    def test_full_dataset_runs_end_to_end(self, dataset_frame: pd.DataFrame) -> None:
        split = split_dataset(dataset_frame, test_size=0.25, random_state=7)
        pipeline = fit_preprocessor(split.X_train, scaling=ScalingMode.STANDARD)
        transformed = transform_features(pipeline, split.X_test)
        assert transformed.shape == (split.test_size, len(FEATURE_COLUMNS))
        assert not transformed.isna().any().any()

    def test_dataset_split_is_constructed_by_the_module(self, dataset_frame: pd.DataFrame) -> None:
        assert isinstance(split_dataset(dataset_frame, test_size=0.25), DatasetSplit)

    def test_values_with_nothing_missing_are_passed_through_unchanged(
        self, features_frame: pd.DataFrame
    ) -> None:
        """Preprocessing only fills gaps; it never reclassifies a real measurement."""
        pipeline = fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        transformed = transform_features(pipeline, features_frame)

        sentinel_free = _sentinel_frame(features_frame)
        untouched = sentinel_free["Glucose"].notna()
        pd.testing.assert_series_equal(
            transformed.loc[untouched, "Glucose"],
            sentinel_free.loc[untouched, "Glucose"],
            check_names=False,
        )

    def test_no_threshold_appears_in_the_fitted_pipeline(
        self, features_frame: pd.DataFrame
    ) -> None:
        """Each statistic is derived from observed data, not a fixed constant."""
        info = describe_preprocessor(fit_preprocessor(features_frame))
        measured = _sentinel_frame(features_frame)
        for name, value in info["imputation_statistics"].items():
            observed = measured[name].dropna()
            assert float(observed.min()) <= float(value) <= float(observed.max()), name

    def test_statistics_change_when_the_data_changes(self, features_frame: pd.DataFrame) -> None:
        """A fitted statistic is a measurement, not a hard-coded cut-off.

        Uses ``DiabetesPedigreeFunction`` because it carries no zero sentinels, so
        a uniform shift moves the median by exactly the shift.
        """
        column = "DiabetesPedigreeFunction"
        baseline = describe_preprocessor(
            fit_preprocessor(features_frame, scaling=ScalingMode.NONE)
        )["imputation_statistics"][column]
        shifted = features_frame.copy()
        shifted[column] = shifted[column] + 1.0
        after = describe_preprocessor(fit_preprocessor(shifted, scaling=ScalingMode.NONE))[
            "imputation_statistics"
        ][column]
        assert float(after) == pytest.approx(float(baseline) + 1.0)


def test_rows_constant_agrees_with_the_shared_fixture() -> None:
    """Guard the fixture against drift in the leakage tests above."""
    assert len(PREPROCESSING_ROWS) == 18
