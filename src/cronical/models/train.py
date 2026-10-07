"""Baseline training: raw features to a fitted, reproducible estimator pipeline.

The central design decision is that **one fitted pipeline carries both stages**.
Preprocessing and the estimator are composed into a single scikit-learn
``Pipeline``, so raw features go in and predictions come out with no opportunity
for a caller to preprocess differently at training time and at inference time.

Consequences that matter:

* Every learned statistic — medians, means, scales — is fitted inside the
  pipeline, therefore on the training rows only.
* Re-fitting under cross-validation cannot leak a validation fold into the
  transforms, because the fold never reaches the fitted state.
* Serialising that one object is enough to reproduce inference exactly. There is
  no second, separately fitted preprocessor that could drift out of step.

Reproducibility
---------------
Every run records the dataset identity (path plus SHA-256), the feature list, the
target, the seed, the test fraction, the preprocessing configuration and the full
model configuration. Nothing is inferred later, and no seed is changed implicitly:
:func:`train_model` takes the seed it is given and records that seed.

Scope
-----
These are **baselines**. No hyperparameter search happens here, no metric is
reported for this dataset until an experiment actually runs, and nothing
produced here is evidence of clinical validity. Model behaviour on one dataset
says nothing about any patient.

Safety
-----
No clinical rule, diagnostic threshold or treatment recommendation appears
anywhere in this module. The only threshold involved is the conventional 0.5
classifier cut-off defined in :mod:`cronical.models.evaluate`, which is a
technical default and carries no clinical meaning.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd
from sklearn.pipeline import Pipeline

from cronical.config import Settings, get_settings
from cronical.data.loader import resolve_dataset_path
from cronical.data.preprocessing import (
    build_training_pipeline,
    describe_preprocessor,
    split_dataset,
)
from cronical.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from cronical.data.validation import validate_dataframe
from cronical.models.config import EstimatorName, ModelSpec, get_spec, require_dependency
from cronical.models.errors import ArtifactError
from cronical.utils.logging import get_logger

__all__ = [
    "ExperimentMetadata",
    "TrainedModel",
    "artifact_path",
    "build_model_pipeline",
    "load_metadata",
    "load_model",
    "load_training_frame",
    "save_model",
    "train_all_baselines",
    "train_model",
]

_LOG = get_logger(__name__)

#: File extension used for persisted pipelines.
_ARTIFACT_SUFFIX: Final[str] = ".joblib"


@dataclass(frozen=True, slots=True)
class ExperimentMetadata:
    """Everything needed to reproduce or audit one training run.

    Attributes:
        model_name: Estimator identifier.
        dataset_path: Where the CSV came from.
        dataset_sha256: Content hash of that file, so a result can be tied to an
            exact input even if the file is later replaced.
        row_count: Rows in the full dataset.
        feature_names: Predictor columns, in the order the pipeline consumes them.
        target: Outcome column name.
        random_state: Seed used for both the split and the estimator.
        test_size: Fraction held out.
        decision_threshold: Classifier cut-off used when scoring.
        preprocessing: What the fitted preprocessing actually learned.
        model_configuration: The full :class:`~cronical.models.config.ModelSpec`.
        created_at: UTC timestamp of the run.
        project_version: Version of the package that produced this artifact.
        train_size: Rows the model was fitted on.
        test_size_rows: Rows held back for scoring.
    """

    model_name: str
    dataset_path: str
    dataset_sha256: str
    row_count: int
    feature_names: tuple[str, ...]
    target: str
    random_state: int
    test_size: float
    decision_threshold: float
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    model_configuration: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    project_version: str = ""
    train_size: int = 0
    test_size_rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the metadata."""
        return {
            "model_name": self.model_name,
            "dataset_path": self.dataset_path,
            "dataset_sha256": self.dataset_sha256,
            "row_count": self.row_count,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "test_size_rows": self.test_size_rows,
            "feature_names": list(self.feature_names),
            "target": self.target,
            "random_state": self.random_state,
            "decision_threshold": self.decision_threshold,
            "preprocessing": dict(self.preprocessing),
            "model_configuration": dict(self.model_configuration),
            "created_at": self.created_at,
            "project_version": self.project_version,
        }


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """A fitted pipeline together with the metadata describing how it was fitted.

    Attributes:
        name: Estimator identifier.
        pipeline: The fitted ``Pipeline``: preprocessing followed by the estimator.
        metadata: Reproducibility record for this run.
        specification: The :class:`~cronical.models.config.ModelSpec` used.
    """

    name: str
    pipeline: Pipeline
    metadata: ExperimentMetadata
    specification: ModelSpec

    def predict_proba(self, features: Any) -> Any:
        """Return positive-class probabilities for ``features``.

        The features are preprocessed by the very pipeline that was fitted, so
        inference cannot drift from training.
        """
        return self.pipeline.predict_proba(features)

    def predict(self, features: Any) -> Any:
        """Return hard predictions for ``features``."""
        return self.pipeline.predict(features)

    def artifact_name(self) -> str:
        """Return the default artifact filename for this model."""
        return f"{self.name}{_ARTIFACT_SUFFIX}"


def _sha256(path: Path) -> str:
    """Return the SHA-256 of a file, or an empty string if it is unreadable."""
    digest = hashlib.sha256()
    try:
        payload = path.read_bytes()
    except OSError:
        return ""
    digest.update(payload)
    return digest.hexdigest()


def load_training_frame(
    path: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Load and validate the dataset for training.

    Validation is not optional: a dataset with a missing column, a non-numeric
    feature or a non-binary outcome is rejected here rather than being fitted
    against.

    Args:
        path: Location of the CSV. Defaults to the configured raw dataset path.
        settings: Configuration source. Defaults to :func:`get_settings`.

    Returns:
        The validated frame and the resolved path it came from.

    Raises:
        DatasetNotFoundError: If the CSV is absent. The message names the
            expected location. Nothing is ever downloaded.
        DatasetValidationError: If the dataset fails validation, carrying the full
            report of every problem found.
    """
    from cronical.data.errors import DatasetValidationError
    from cronical.data.loader import load_dataset

    resolved = resolve_dataset_path(path)
    frame = load_dataset(resolved)
    report = validate_dataframe(frame, source=str(resolved))
    if report.errors:
        raise DatasetValidationError(report)
    _LOG.info(
        "training dataset ready",
        extra={
            "path": str(resolved),
            "rows": report.row_count,
            "warnings": len(report.warnings),
        },
    )
    return frame, resolved


def build_model_pipeline(
    specification: ModelSpec,
    *,
    settings: Settings | None = None,
) -> Pipeline:
    """Compose preprocessing and the estimator into one unfitted pipeline.

    The preprocessing path comes from the specification, so a linear model gets
    standardisation and a tree ensemble does not. That is the only difference
    between the three baselines.

    Args:
        specification: The model to compose with.
        settings: Configuration source. Defaults to :func:`get_settings`.

    Returns:
        An unfitted pipeline ending in the requested estimator.

    Raises:
        MissingDependencyError: If the estimator's package is not installed.
        UnknownModelError: If the specification's estimator cannot be built.
    """
    require_dependency(specification.name)
    estimator = _build_estimator(specification)
    pipeline = build_training_pipeline(
        estimator, scaling=specification.scaling, settings=settings
    )
    _LOG.info(
        "model pipeline composed",
        extra={"model": specification.name.value, "scaling": specification.scaling.value},
    )
    return pipeline


def _build_estimator(specification: ModelSpec) -> Any:
    """Instantiate the estimator described by a specification.

    Imports happen here so an absent optional dependency surfaces as a typed
    error rather than an import failure at module load. ``require_dependency``
    already checks, so this is a second line of defence.
    """
    from cronical.models.errors import MissingDependencyError

    parameters = dict(specification.hyperparameters)
    try:
        if specification.name is EstimatorName.LOGISTIC_REGRESSION:
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(**parameters)
        if specification.name is EstimatorName.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(**parameters)
        from xgboost import XGBClassifier

        return XGBClassifier(**parameters)
    except ImportError as exc:  # pragma: no cover - guarded by require_dependency
        raise MissingDependencyError(
            exc.name or "unknown", model=specification.name.value, extra="ml"
        ) from exc


def train_model(
    frame: pd.DataFrame,
    name: EstimatorName | str,
    *,
    random_state: int | None = None,
    test_size: float | None = None,
    settings: Settings | None = None,
    dataset_path: Path | str | None = None,
    dataset_sha256: str = "",
    frame_label: str = "<in-memory DataFrame>",
) -> TrainedModel:
    """Split, fit and return one baseline model.

    The split happens first, and the pipeline is then fitted on the training
    partition alone. The test partition is never seen during fitting.

    Args:
        frame: The validated dataset, including the outcome column.
        name: Which baseline to train.
        random_state: Seed for the split and the estimator. Defaults to
            :attr:`cronical.config.Settings.random_seed`.
        test_size: Fraction held out. Defaults to the configured value.
        settings: Configuration source. Defaults to :func:`get_settings`.
        dataset_path: Where the frame came from, recorded in the metadata.
        dataset_sha256: Content hash of the source file, when known.
        frame_label: Human-readable dataset label for the metadata.

    Returns:
        The fitted pipeline with its reproducibility record.

    Raises:
        UnknownModelError: If ``name`` is not a registered baseline.
        MissingDependencyError: If the estimator's package is not installed.
        DatasetValidationError: If the frame fails validation.
        SplitError: If the frame cannot be split as requested.
    """
    resolved = get_settings() if settings is None else settings
    seed = resolved.random_seed if random_state is None else random_state
    specification = get_spec(
        name,
        random_state=seed,
        n_jobs=resolved.training.n_jobs,
    )
    split = split_dataset(frame, test_size=test_size, random_state=seed, settings=resolved)
    pipeline = build_model_pipeline(specification, settings=resolved)

    _LOG.info(
        "fitting baseline",
        extra={
            "model": specification.name.value,
            "train_rows": split.train_size,
            "random_state": seed,
        },
    )
    pipeline.fit(split.X_train, split.y_train)

    from cronical import __version__

    metadata = ExperimentMetadata(
        model_name=specification.name.value,
        dataset_path=str(dataset_path) if dataset_path is not None else frame_label,
        dataset_sha256=dataset_sha256,
        row_count=int(len(frame.index)),
        feature_names=tuple(str(name) for name in split.feature_names),
        target=TARGET_COLUMN,
        random_state=seed,
        test_size=split.test_size / (split.train_size + split.test_size),
        decision_threshold=resolved.training.decision_threshold,
        preprocessing=_fitted_preprocessing(pipeline),
        model_configuration=specification.as_dict(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        project_version=__version__,
        train_size=split.train_size,
        test_size_rows=split.test_size,
    )
    return TrainedModel(
        name=specification.name.value,
        pipeline=pipeline,
        metadata=metadata,
        specification=specification,
    )


def _fitted_preprocessing(pipeline: Pipeline) -> dict[str, Any]:
    """Return what the fitted preprocessing learned, or a reason it cannot be read.

    A model that fitted successfully should still be reportable if its
    preprocessing cannot be described, so this degrades rather than raising.
    """
    from cronical.data.errors import PreprocessingError as DataPreprocessingError
    from cronical.models.errors import CronicalModelError

    try:
        return describe_preprocessor(pipeline)
    except (CronicalModelError, DataPreprocessingError) as exc:
        return {"unavailable": str(exc)}


def train_all_baselines(
    frame: pd.DataFrame,
    *,
    names: Sequence[EstimatorName | str] | None = None,
    random_state: int | None = None,
    test_size: float | None = None,
    settings: Settings | None = None,
    dataset_path: Path | str | None = None,
    dataset_sha256: str = "",
) -> list[TrainedModel]:
    """Train every baseline on the same split.

    The seed and test fraction are passed to each model, so all three are scored
    against exactly the same partition. That is what makes the comparison fair:
    a difference between two models is then a difference between the models, not
    between the data each happened to see.

    Args:
        frame: The validated dataset.
        names: Baselines to train. Defaults to all three.
        random_state: Shared seed. Defaults to the configured value.
        test_size: Shared test fraction. Defaults to the configured value.
        settings: Configuration source. Defaults to :func:`get_settings`.
        dataset_path: Source path, recorded in each metadata block.
        dataset_sha256: Source content hash.

    Returns:
        One :class:`TrainedModel` per requested baseline, in registry order.
    """
    from cronical.models.config import BASELINE_REGISTRY

    selected = tuple(names) if names is not None else tuple(BASELINE_REGISTRY)
    return [
        train_model(
            frame,
            name,
            random_state=random_state,
            test_size=test_size,
            settings=settings,
            dataset_path=dataset_path,
            dataset_sha256=dataset_sha256,
        )
        for name in selected
    ]


def artifact_path(
    model: TrainedModel,
    *,
    settings: Settings | None = None,
    directory: Path | None = None,
) -> Path:
    """Return where a model's artifact belongs, under ``models/``."""
    resolved = get_settings() if settings is None else settings
    base = resolved.paths.models_dir if directory is None else Path(directory)
    return base / model.artifact_name()


def save_model(
    model: TrainedModel,
    *,
    path: Path | str | None = None,
    settings: Settings | None = None,
    write_metadata: bool = True,
) -> tuple[Path, Path | None]:
    """Persist a fitted pipeline and, separately, its metadata.

    The pipeline and its metadata are written as two files. Keeping them apart
    means the JSON can be read, diffed and reviewed without unpickling anything,
    which matters when the provenance of a result is the question.

    No metric is written here. Metrics are produced by evaluation and belong in a
    report, not in the artifact.

    Args:
        model: The fitted model to write.
        path: Destination for the pipeline. Defaults under ``models/``.
        settings: Configuration source. Defaults to :func:`get_settings`.
        write_metadata: Also write a sibling ``.json`` metadata file.

    Returns:
        The pipeline path, and the metadata path when one was written.

    Raises:
        ArtifactError: If the artifact cannot be written.
    """
    import joblib

    destination = (
        artifact_path(model, settings=settings) if path is None else Path(path).expanduser()
    )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # joblib ships no type information and provides no stub package.
        joblib.dump(model.pipeline, destination)  # type: ignore[no-untyped-call]
    except OSError as exc:
        raise ArtifactError(f"could not write model artifact to {destination}: {exc}") from exc

    metadata_path: Path | None = None
    if write_metadata:
        metadata_path = destination.with_suffix(".json")
        try:
            metadata_path.write_text(
                json.dumps(model.metadata.as_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ArtifactError(
                f"could not write model metadata to {metadata_path}: {exc}"
            ) from exc

    _LOG.info(
        "model artifact saved",
        extra={"model": model.name, "path": str(destination), "metadata": str(metadata_path)},
    )
    return destination, metadata_path


def load_model(path: Path | str) -> Pipeline:
    """Load a persisted preprocessing-plus-estimator pipeline.

    Args:
        path: Artifact to read.

    Returns:
        The fitted pipeline.

    Raises:
        ArtifactError: If the file is absent or cannot be unpickled.
    """
    import joblib

    source = Path(path).expanduser()
    if not source.is_file():
        raise ArtifactError(f"no model artifact at {source}")
    try:
        pipeline = cast(Pipeline, joblib.load(source))  # type: ignore[no-untyped-call]
    except Exception as exc:  # noqa: BLE001 - joblib raises many unpickling types
        raise ArtifactError(f"could not read model artifact at {source}: {exc}") from exc
    return pipeline


def load_metadata(path: Path | str) -> dict[str, Any]:
    """Read a saved metadata sidecar.

    Args:
        path: Artifact path or metadata path; a ``.json`` suffix is added if
            absent.

    Returns:
        The decoded metadata.

    Raises:
        ArtifactError: If the file is absent or is not valid JSON.
    """
    source = Path(path).expanduser()
    if source.suffix != ".json":
        source = source.with_suffix(".json")
    if not source.is_file():
        raise ArtifactError(f"no model metadata at {source}")
    try:
        decoded: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"model metadata at {source} is not valid JSON: {exc}") from exc
    return decoded