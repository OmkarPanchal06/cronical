"""Leakage-safe feature preprocessing for the diabetes dataset.

This module turns a validated raw dataset into model-ready features using an
ordinary :class:`sklearn.pipeline.Pipeline`, so the *same fitted object* is used
during training, cross-validation, testing and single-patient inference.

The pipeline, in order
----------------------

1. :class:`FeatureContract` — validate against the schema and fix column order.
2. :class:`SentinelZeroHandler` — turn documented "not recorded" zeros into
   missing values.
3. :class:`sklearn.impute.SimpleImputer` — fill the gaps from statistics learned
   on the **training partition only**.
4. :class:`sklearn.preprocessing.StandardScaler` — **optional**, see below.

Why zero sentinels become missing
---------------------------------
Five columns in this dataset use ``0`` to mean *no measurement was recorded*
rather than a genuine zero: ``Glucose``, ``BloodPressure``, ``SkinThickness``,
``Insulin`` and ``BMI``. Nobody's blood pressure is zero, so a recorded zero in
one of these columns is a placeholder for absent data.

Leaving them as numbers would let the pipeline learn that "zero glucose" is a
very low value, quietly treating a missing measurement as an extreme
observation. Converting them to ``NaN`` first, and imputing afterwards, means
every derived statistic is computed from measurements that were actually taken.

``Pregnancies`` is deliberately excluded: a pregnancy count of zero is a real
observation, not a placeholder. ``Outcome`` is never passed through this
pipeline at all.

Why raw data is left untouched
------------------------------
The raw CSV is the only record of what was actually supplied. This module
therefore never writes to it and never mutates the frame it is given: every
transform copies first. Cleaning decisions live in code that can be read,
reviewed and re-run, rather than in an edited data file whose history cannot be
reconstructed.

Why statistics are fitted only on training data
----------------------------------------------
Any statistic learned from data — a median, a mean, a scale — carries information
about every row it saw. If the test partition contributes to those statistics,
the model is indirectly exposed to the answers it will be scored against, and the
resulting metrics are optimistic for reasons that have nothing to do with
predictive skill.

Consequently :func:`fit_preprocessor` accepts only the training partition, and
:func:`split_dataset` exists so the two partitions never meet. The tests assert
this directly by fitting on a partition whose medians differ measurably from the
full dataset.

Why median imputation
---------------------
Missing measurements are unlikely to be missing at random: they may be absent
precisely because they were hard to obtain. The median is pulled towards the
middle of the observed distribution and, unlike the mean, is not dragged around
by a small number of unusually large values.

It is preferred here over the mean because imputation must not manufacture an
observation that sits outside the plausible range of what was recorded, and over
model-based or iterative methods because it introduces no additional fitted
model that would need its own validation. It is a **conservative default, not a
claim that it is optimal.** Whether imputation should happen at all, and whether
these rows should instead be dropped or the missingness modelled explicitly,
remains an open modelling decision documented in the README.

Why scaling is optional
-----------------------
Linear models such as logistic regression learn a coefficient per feature, so
features measured on wildly different scales dominate the fit purely because of
their units. Standardisation puts them on a comparable footing there.

Tree ensembles split on thresholds and are indifferent to monotone rescaling of a
feature: the ordering of values is all that matters. Scaling them would add a
transformation, a set of fitted parameters and a serialisation dependency while
changing nothing about the model — so it is skipped for tree-based learners.

The choice is explicit via :class:`cronical.config.ScalingMode` rather than
applied unconditionally, and the two paths share every other step, so the
difference is genuinely one step.

Scope
-----
This module prepares features. It does not select a model, fit one, tune one, or
interpret one, and it produces no clinical output. Nothing here has been
validated for clinical use.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cronical.config import (
    ImputationStrategy,
    ScalingMode,
    Settings,
    get_settings,
)
from cronical.data.errors import FeatureContractError, PreprocessingError, SplitError
from cronical.data.schema import FEATURE_COLUMNS, TARGET_COLUMN, ZERO_SENTINEL_COLUMNS
from cronical.data.validation import raise_for_errors, validate_dataframe
from cronical.utils.logging import get_logger

__all__ = [
    "DatasetSplit",
    "FeatureContract",
    "SentinelZeroHandler",
    "build_preprocessor",
    "build_training_pipeline",
    "describe_preprocessor",
    "dump_preprocessor",
    "fit_preprocessor",
    "load_preprocessor",
    "save_preprocessor",
    "split_dataset",
    "transform_features",
    "transform_patient",
]

_LOG = get_logger(__name__)

#: Name of the step that enforces the feature contract.
_CONTRACT_STEP = "feature_contract"
#: Name of the step that replaces zero sentinels.
_SENTINEL_STEP = "sentinel_zeros"
#: Name of the imputer step.
_IMPUTER_STEP = "imputer"
#: Name of the optional scaler step.
_SCALER_STEP = "scaler"


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Two disjoint partitions of a dataset, features separated from target.

    Row indices are preserved from the input frame so any transformed row can be
    traced back to the record it came from.

    Attributes:
        X_train: Feature rows for fitting.
        X_test: Feature rows held back for scoring.
        y_train: Outcome values for fitting.
        y_test: Outcome values held back for scoring.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def train_size(self) -> int:
        """Number of rows in the training partition."""
        return len(self.X_train.index)

    @property
    def test_size(self) -> int:
        """Number of rows in the test partition."""
        return len(self.X_test.index)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Feature column names, shared by both partitions."""
        return tuple(str(name) for name in self.X_train.columns)

    def sizes(self) -> dict[str, int]:
        """Return row counts per partition, for logging and assertions."""
        return {"train": self.train_size, "test": self.test_size}

    def class_balance(self) -> dict[str, dict[str, int]]:
        """Return the outcome distribution in each partition.

        Useful for confirming that stratification actually held.
        """
        return {
            "train": {str(key): int(value) for key, value in self.y_train.value_counts().items()},
            "test": {str(key): int(value) for key, value in self.y_test.value_counts().items()},
        }


def _as_feature_frame(data: Any, required: Sequence[str]) -> pd.DataFrame:
    """Coerce supported inputs into a feature frame.

    Accepts a :class:`~pandas.DataFrame`, a single record given as a mapping or a
    :class:`~pandas.Series`, or a positional array. Anything else, or an array
    whose width does not match the schema, is rejected with a typed error rather
    than silently mis-aligned.

    Args:
        data: The input to coerce.
        required: Column names the frame must end up carrying.

    Returns:
        A DataFrame. It may be the original object, so callers that intend to
        mutate must copy.

    Raises:
        FeatureContractError: If the input cannot be aligned with the schema.
    """
    columns = list(required)
    if isinstance(data, pd.DataFrame):
        # Returned untouched so that a caller-supplied frame keeps its own dtypes;
        # a non-numeric column here must be *reported*, not silently coerced.
        return data
    if isinstance(data, pd.Series):
        return data.to_frame().T.infer_objects()
    if isinstance(data, Mapping):
        return pd.DataFrame([dict(data)]).infer_objects()
    try:
        array = np.asarray(data, dtype=object)
    except (TypeError, ValueError) as exc:
        raise FeatureContractError(f"unsupported input type {type(data).__name__}") from exc
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != len(columns):
        raise FeatureContractError(
            f"expected an array with {len(columns)} column(s) named {columns}, "
            f"got shape {array.shape}"
        )
    # Positional input carries no dtypes, so infer them. A genuinely non-numeric
    # entry stays object and is reported by the contract rather than coerced away.
    return pd.DataFrame(array, columns=columns).infer_objects()


class FeatureContract(BaseEstimator, TransformerMixin):
    """Validate a feature frame against the schema and fix its column order.

    This is the first pipeline step, and the only place that knows the schema.
    Rejecting bad input here rather than downstream means a malformed record
    fails at the boundary with a message naming the offending column, instead of
    producing a plausible-looking row of the wrong width.

    Extra columns are dropped, with the dropped names logged. Silent column
    dropping would let an outcome column or a future leaky feature pass through
    unnoticed.

    Args:
        required: Column names the frame must carry, in output order.
    """

    def __init__(self, required: tuple[str, ...] = FEATURE_COLUMNS) -> None:
        self.required = required

    def fit(self, X: Any, y: Any = None) -> Self:
        """Validate the frame and record the fitted contract.

        Args:
            X: Feature rows.
            y: Ignored. The contract is unsupervised; the target is never a
                feature, so it must not influence what is fitted.

        Returns:
            The fitted transformer.
        """
        self._validated(X)
        self.n_features_in_ = len(self.required)
        _LOG.debug("feature contract fitted", extra={"features": len(self.required)})
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        """Return the frame reduced to the required columns, as ``float64``.

        Args:
            X: Feature rows.

        Returns:
            A new DataFrame. The input is never modified.

        Raises:
            FeatureContractError: If the frame does not satisfy the schema, or if
                this transformer has not been fitted.
        """
        if not hasattr(self, "n_features_in_"):
            raise FeatureContractError("FeatureContract.transform called before fit")
        # pandas annotates DataFrame.astype as returning Any.
        return cast(pd.DataFrame, self._validated(X).loc[:, list(self.required)].astype("float64"))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the output column names, for scikit-learn's output API."""
        return np.asarray(self.required, dtype=object)

    def _validated(self, X: Any) -> pd.DataFrame:
        """Return ``X`` as a frame after checking it against the schema."""
        frame = _as_feature_frame(X, self.required)
        missing = [name for name in self.required if name not in frame.columns]
        if missing:
            raise FeatureContractError(f"missing required feature column(s): {missing}")
        non_numeric = [
            name for name in self.required if not pd.api.types.is_numeric_dtype(frame[name])
        ]
        if non_numeric:
            types = {name: str(frame[name].dtype) for name in non_numeric}
            raise FeatureContractError(f"feature column(s) must be numeric: {types}")
        unexpected = [str(name) for name in frame.columns if str(name) not in self.required]
        if unexpected:
            _LOG.warning("dropping unexpected columns", extra={"columns": unexpected})
        return frame


class SentinelZeroHandler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Replace documented zero placeholders with missing values.

    Only the columns named in :data:`cronical.data.schema.ZERO_SENTINEL_COLUMNS`
    are touched. Everything else, including ``Pregnancies``, passes through
    unchanged — including a genuine zero.

    The incoming frame is copied before anything is written, so the caller's data
    is left exactly as it was.

    Args:
        columns: Columns whose zeros mean "not recorded". Never mutate this; use
            :meth:`fit` to see the columns actually in use.
    """

    def __init__(self, columns: tuple[str, ...] = ZERO_SENTINEL_COLUMNS) -> None:
        self.columns = columns

    def fit(self, X: Any, y: Any = None) -> Self:
        """Confirm the sentinel columns are present.

        Args:
            X: Feature rows.
            y: Ignored; this transform uses no labels.

        Returns:
            The fitted transformer.

        Raises:
            FeatureContractError: If a configured sentinel column is absent.
        """
        frame = X if isinstance(X, pd.DataFrame) else _as_feature_frame(X, ZERO_SENTINEL_COLUMNS)
        missing = [name for name in self.columns if name not in frame.columns]
        if missing:
            raise FeatureContractError(f"sentinel column(s) not present: {missing}")
        self.columns_ = tuple(self.columns)
        # OneToOneFeatureMixin reads both of these to decide the transformer is
        # fitted and to recover the incoming column names. Without
        # `feature_names_in_` it would relabel every column x0..xn, silently
        # destroying the schema names that downstream steps and SHAP rely on.
        self.feature_names_in_ = np.asarray([str(name) for name in frame.columns], dtype=object)
        self.n_features_in_ = len(frame.columns)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        """Return a copy with zero sentinels replaced by ``NaN``.

        Args:
            X: Feature rows.

        Returns:
            A new DataFrame in which the configured columns hold ``NaN`` wherever
            they held ``0``.

        Raises:
            FeatureContractError: If this transformer has not been fitted.
        """
        if not hasattr(self, "columns_"):
            raise FeatureContractError("SentinelZeroHandler.transform called before fit")
        frame = X if isinstance(X, pd.DataFrame) else _as_feature_frame(X, ZERO_SENTINEL_COLUMNS)
        result = frame.copy(deep=True)
        for column in self.columns_:
            series = result[column]
            result[column] = series.where(series != 0, other=np.nan)
        return result


def build_preprocessor(
    *,
    scaling: ScalingMode | str | None = None,
    imputation: ImputationStrategy | None = None,
    settings: Settings | None = None,
) -> Pipeline:
    """Assemble an unfitted preprocessing pipeline.

    The same steps are always used; ``scaling`` decides only whether the final
    step exists. That keeps the linear and tree paths from drifting apart.

    Args:
        scaling: Scaling to apply. Defaults to
            :attr:`~cronical.config.PreprocessingSettings.default_scaling`.
            Use :attr:`ScalingMode.NONE` for tree-based models.
        imputation: Imputation strategy. Defaults to the configured strategy.
        settings: Configuration source. Defaults to :func:`get_settings`.

    Returns:
        An unfitted pipeline whose output is a pandas DataFrame carrying the
        schema's column names.
    """
    config = _resolve_settings(settings).preprocessing
    mode = config.default_scaling if scaling is None else ScalingMode(scaling)
    strategy = config.imputation_strategy if imputation is None else imputation

    steps: list[tuple[str, Any]] = [
        (_CONTRACT_STEP, FeatureContract()),
        (_SENTINEL_STEP, SentinelZeroHandler()),
        # keep_empty_features stops an all-missing column from silently
        # changing the output width between training and inference.
        (_IMPUTER_STEP, SimpleImputer(strategy=strategy, keep_empty_features=True)),
    ]
    if mode is ScalingMode.STANDARD:
        steps.append((_SCALER_STEP, StandardScaler()))

    pipeline = Pipeline(steps)
    _LOG.info(
        "preprocessor built",
        extra={"imputation": strategy, "scaling": mode.value, "steps": [name for name, _ in steps]},
    )
    # The stubs type set_output as returning BaseEstimator; it returns self.
    return cast(Pipeline, pipeline.set_output(transform="pandas"))


def build_training_pipeline(
    estimator: Any,
    *,
    scaling: ScalingMode | str | None = None,
    imputation: ImputationStrategy | None = None,
    settings: Settings | None = None,
) -> Pipeline:
    """Compose the preprocessing pipeline with an estimator.

    Provided for the modelling stage, which must not be able to bypass
    preprocessing. No estimator is defined in this project yet, so nothing here
    fits or selects one; the caller supplies it.

    The result deliberately omits ``set_output``: once an estimator is appended,
    the meaning of the pipeline's output belongs to that estimator rather than to
    the preprocessing steps.

    Args:
        estimator: A scikit-learn compatible estimator, or the string
            ``"passthrough"``.
        scaling: Scaling to apply. Defaults to the configured mode.
        imputation: Imputation strategy. Defaults to the configured strategy.
        settings: Configuration source. Defaults to :func:`get_settings`.

    Returns:
        An unfitted pipeline ending in ``estimator``.
    """
    config = _resolve_settings(settings).preprocessing
    mode = config.default_scaling if scaling is None else ScalingMode(scaling)
    strategy = config.imputation_strategy if imputation is None else imputation

    steps: list[tuple[str, Any]] = [
        (_CONTRACT_STEP, FeatureContract()),
        (_SENTINEL_STEP, SentinelZeroHandler()),
        (_IMPUTER_STEP, SimpleImputer(strategy=strategy, keep_empty_features=True)),
    ]
    if mode is ScalingMode.STANDARD:
        steps.append((_SCALER_STEP, StandardScaler()))
    steps.append(("estimator", estimator))
    return Pipeline(steps)


def fit_preprocessor(
    X_train: Any,
    *,
    y_train: Any = None,
    scaling: ScalingMode | str | None = None,
    imputation: ImputationStrategy | None = None,
    settings: Settings | None = None,
) -> Pipeline:
    """Fit preprocessing on the training partition and return the fitted pipeline.

    Pass **only** the training partition here. Fitting on the full dataset would
    leak information about held-out rows into the learned medians and scaling
    parameters, inflating every metric computed afterwards.

    Args:
        X_train: Training feature rows only.
        y_train: Training outcomes, accepted for symmetry with scikit-learn. It is
            never used to fit anything, and never reaches the target. Fitting any
            preprocessing step against the outcome would be target leakage.
        scaling: Scaling to apply. Defaults to the configured mode.
        imputation: Imputation strategy. Defaults to the configured strategy.
        settings: Configuration source. Defaults to :func:`get_settings`.

    Returns:
        The fitted pipeline, ready to transform further data.
    """
    pipeline = build_preprocessor(scaling=scaling, imputation=imputation, settings=settings)
    pipeline.fit(X_train, y_train)
    fitted = describe_preprocessor(pipeline)
    _LOG.info(
        "preprocessor fitted",
        extra={
            "imputation": fitted["imputation_strategy"],
            "scaling": fitted["scaling"],
            "imputed_values": fitted["imputation_statistics"],
            "rows": len(X_train.index) if isinstance(X_train, pd.DataFrame) else None,
        },
    )
    return pipeline


def transform_features(pipeline: Pipeline, X: Any) -> pd.DataFrame:
    """Apply a fitted pipeline to feature rows.

    Args:
        pipeline: A pipeline fitted by :func:`fit_preprocessor`.
        X: Rows to transform. Accepts a DataFrame, a mapping or Series for a
            single record, or a positional array.

    Returns:
        A DataFrame of transformed features, with the schema's column names and
        no target column.

    Raises:
        FeatureContractError: If the rows do not satisfy the feature schema.
    """
    # The stubs do not model set_output, so the DataFrame return is invisible.
    return cast(pd.DataFrame, pipeline.transform(X))


def transform_patient(pipeline: Pipeline, record: Mapping[str, Any] | pd.Series) -> pd.DataFrame:
    """Transform one patient record with an already-fitted pipeline.

    This is the inference path. Using the pipeline fitted during training — rather
    than refitting on the incoming record — is what guarantees that inference sees
    exactly the transformations the model was trained on.

    No prediction is made here; only the features are prepared.

    Args:
        pipeline: A pipeline fitted by :func:`fit_preprocessor`.
        record: One record as a mapping from feature name to value, or as a
            :class:`~pandas.Series`, using the schema's names.

    Returns:
        A single-row DataFrame of transformed features.

    Raises:
        FeatureContractError: If the record is missing required features or holds
            a non-numeric value.
    """
    frame = transform_features(pipeline, record)
    _LOG.debug("patient record transformed", extra={"features": len(frame.columns)})
    return frame


def describe_preprocessor(pipeline: Pipeline) -> dict[str, Any]:
    """Report what a fitted pipeline learned.

    Exposes the fitted medians and scaling parameters so a run can be audited and
    a result reproduced. Every value is read from the fitted object, never
    recomputed from data.

    Args:
        pipeline: A fitted preprocessing pipeline.

    Returns:
        A JSON-serialisable summary including per-feature imputation statistics,
        the scaling mode, and the column order the pipeline will produce.

    Raises:
        PreprocessingError: If the pipeline has not been fitted.
    """
    if _IMPUTER_STEP not in pipeline.named_steps:
        raise PreprocessingError("pipeline does not contain the expected imputation step")
    imputer: SimpleImputer = pipeline.named_steps[_IMPUTER_STEP]
    if not hasattr(imputer, "statistics_"):
        raise PreprocessingError("preprocessor has not been fitted; call fit_preprocessor first")

    statistics = np.asarray(imputer.statistics_, dtype=float)
    names = list(FEATURE_COLUMNS)
    summary: dict[str, Any] = {
        # Read via get_params: the stubs do not expose `strategy` as an
        # attribute, but it is a declared constructor parameter.
        "imputation_strategy": str(imputer.get_params()["strategy"]),
        "scaling": _scaling_mode(pipeline).value,
        # `keep_empty_features=True` means an unusable column is filled rather
        # than dropped, so every statistic is a finite float and the summary is
        # always valid JSON.
        "imputation_statistics": {
            name: float(value) for name, value in zip(names, statistics, strict=True)
        },
        "sentinel_columns": list(ZERO_SENTINEL_COLUMNS),
        "feature_order": names,
    }

    if _SCALER_STEP in pipeline.named_steps:
        scaler: StandardScaler = pipeline.named_steps[_SCALER_STEP]
        summary["scaler_mean"] = [float(value) for value in np.asarray(scaler.mean_, dtype=float)]
        summary["scaler_scale"] = [float(value) for value in np.asarray(scaler.scale_, dtype=float)]
    return summary


def _scaling_mode(pipeline: Pipeline) -> ScalingMode:
    """Return the scaling a pipeline applies, inferred from its steps."""
    return ScalingMode.STANDARD if _SCALER_STEP in pipeline.named_steps else ScalingMode.NONE


def split_dataset(
    frame: pd.DataFrame,
    *,
    test_size: float | None = None,
    random_state: int | None = None,
    settings: Settings | None = None,
) -> DatasetSplit:
    """Split a validated dataset into stratified train and test partitions.

    The frame is validated first, so a missing column or a non-binary outcome
    stops the split before anything is produced. Stratification keeps the outcome
    balance of both partitions close to the original, which matters when the
    classes are uneven.

    Row indices are preserved, so any row can be traced back to the record it
    came from. Neither partition is modified.

    Args:
        frame: The full dataset, including the outcome column.
        test_size: Fraction held out for testing. Defaults to the configured
            value.
        random_state: Seed for the split. Defaults to
            :attr:`~cronical.config.Settings.random_seed`.
        settings: Configuration source. Defaults to :func:`get_settings`.

    Returns:
        A :class:`DatasetSplit` with features and target already separated.

    Raises:
        DatasetValidationError: If the dataset fails validation, including a
            non-binary or non-numeric outcome.
        FeatureContractError: If a required feature column is missing.
        SplitError: If a class has too few rows to appear in both partitions.
    """
    resolved = _resolve_settings(settings)
    config = resolved.preprocessing
    size = config.test_size if test_size is None else test_size
    seed = resolved.random_seed if random_state is None else random_state

    raise_for_errors(validate_dataframe(frame))

    # Reuse the contract step's own logic rather than duplicating it: this
    # selects the feature columns, drops the outcome, and raises a typed error
    # if a required column is missing or non-numeric.
    contract = FeatureContract()
    contract.fit(frame)
    features = contract.transform(frame)
    target = frame[TARGET_COLUMN]
    _assert_stratifiable(target, size)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=size,
        random_state=seed,
        stratify=target,
    )
    split = DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    _LOG.info(
        "dataset split",
        extra={
            "train": split.train_size,
            "test": split.test_size,
            "test_size": size,
            "random_state": seed,
            "class_balance": split.class_balance(),
        },
    )
    return split


def _assert_stratifiable(target: pd.Series, test_size: float) -> None:
    """Reject a target that cannot be split in a stratified way.

    Args:
        target: The outcome column.
        test_size: Fraction to hold out.

    Raises:
        SplitError: If a class is too rare to appear in both partitions, or if the
            requested fraction cannot accommodate every class.
    """
    counts = target.value_counts()
    smallest = int(counts.min()) if len(counts) else 0
    if smallest < 2:
        raise SplitError(
            f"the rarest outcome value appears {smallest} time(s); stratified "
            "splitting needs at least 2 of every class"
        )
    rows = len(target.index)
    # scikit-learn rounds the held-out count up, so match it exactly.
    held_out = math.ceil(rows * test_size)
    if held_out < len(counts):
        raise SplitError(
            f"holding out {held_out} row(s) cannot cover all {len(counts)} outcome "
            "values; increase test_size or use more data"
        )


def save_preprocessor(
    pipeline: Pipeline,
    *,
    path: Path | str | None = None,
    settings: Settings | None = None,
) -> Path:
    """Persist a fitted preprocessing pipeline.

    Saving the fitted preprocessing is what lets inference reuse it verbatim. No
    model is written by this project.

    Args:
        pipeline: A fitted preprocessing pipeline.
        path: Destination. Defaults to the configured location under ``models/``.
        settings: Configuration source. Defaults to :func:`get_settings`.

    Returns:
        The path written to.

    Raises:
        PreprocessingError: If the pipeline has not been fitted.
    """
    describe_preprocessor(pipeline)
    resolved_settings = _resolve_settings(settings)
    destination = Path(path) if path is not None else resolved_settings.preprocessor_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    # joblib ships no type information and provides no stub package.
    joblib.dump(pipeline, destination)  # type: ignore[no-untyped-call]
    _LOG.info("preprocessor saved", extra={"path": str(destination)})
    return destination


def load_preprocessor(
    *,
    path: Path | str | None = None,
    settings: Settings | None = None,
) -> Pipeline:
    """Load a previously saved preprocessing pipeline.

    Args:
        path: File to read. Defaults to the configured location under
            ``models/``.
        settings: Configuration source. Defaults to :func:`get_settings`.

    Returns:
        The fitted pipeline, ready to transform new data.

    Raises:
        PreprocessingError: If no file exists at ``path``.
    """
    source = Path(path) if path is not None else _resolve_settings(settings).preprocessor_path
    if not source.is_file():
        raise PreprocessingError(
            f"no saved preprocessor at {source}; fit and save one with "
            "fit_preprocessor() and save_preprocessor() first"
        )
    pipeline = cast(Pipeline, joblib.load(source))  # type: ignore[no-untyped-call]
    _LOG.info("preprocessor loaded", extra={"path": str(source)})
    return pipeline


def _resolve_settings(settings: Settings | None) -> Settings:
    """Return ``settings`` or the process-wide default."""
    return get_settings() if settings is None else settings


def dump_preprocessor(pipeline: Pipeline, path: Path | str) -> str:
    """Return a JSON description of a fitted pipeline and write it to ``path``.

    Useful for recording, alongside a report, exactly which transformations a set
    of results was produced under.

    Args:
        pipeline: A fitted preprocessing pipeline.
        path: Destination for the JSON description.

    Returns:
        The JSON text that was written.
    """
    description = json.dumps(describe_preprocessor(pipeline), indent=2, sort_keys=True)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(description, encoding="utf-8")
    return description
