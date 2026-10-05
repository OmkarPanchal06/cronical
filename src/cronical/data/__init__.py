"""Data layer: acquisition, validation and preprocessing.

Scope
-----
This package owns everything between "a CSV exists somewhere" and "a clean,
modelling-ready table exists". It deliberately contains **no** model code, so the
same preprocessing can be reused for training, cross-validation and inference
without dragging estimators along.

Planned responsibilities
------------------------
* Dataset provenance: record the source, licence, retrieval date and checksum of
  every raw file so results stay reproducible.
* Schema validation: expected feature names, dtypes, units and missing-value
  conventions, checked with an explicit contract rather than assumed.
* A single, deterministic :func:`preprocess` entry point encapsulating cleaning,
  encoding and scaling, fitted on the training split only.
* Serialised preprocessing artifacts, so inference applies exactly the
  transformations the model was trained on.

Current contents
----------------
``schema``
    The explicit column contract for the Pima Indians Diabetes dataset, including
    which columns use zero as a "not recorded" placeholder.
``errors``
    A typed exception hierarchy, so callers can distinguish a missing file from a
    malformed one from a schema violation without parsing message strings.
``loader``
    Reads the CSV from a configurable local path. No network access: the dataset
    is placed in ``data/raw/`` by a developer, never downloaded automatically.
``validation``
    Read-only checks over a loaded frame, returning a
    :class:`~cronical.data.report.DataQualityReport`. Nothing is modified and
    nothing is imputed here.
``report``
    Typed structures describing what validation found.
``preprocessing``
    Leakage-safe feature preparation: sentinel handling, imputation, and
    model-appropriate scaling, composed as a scikit-learn pipeline so the same
    fitted object can serve training, validation and inference.

Data handling rules
-------------------
* **Raw data is immutable.** ``data/raw/`` is never edited in place. Cleaning
  decisions belong to a later preprocessing stage.
* **Nothing is invented.** Every figure in a report is computed from the dataset
  that was actually inspected.
* **Zero sentinels are surfaced, not resolved.** A recorded glucose of ``0`` is
  not a measurement. Validation reports those cells; preprocessing converts them
  to missing values and imputes from the training partition only.
* **Learned statistics come from training data only.** Medians and scaling
  parameters fitted on the full dataset would leak information about held-out
  rows and make reported performance optimistic for the wrong reason.
"""

from __future__ import annotations

from cronical.data.errors import (
    CronicalDataError,
    DatasetFormatError,
    DatasetNotFoundError,
    DatasetPathError,
    DatasetReadError,
    DatasetValidationError,
    FeatureContractError,
    PreprocessingError,
    SplitError,
)
from cronical.data.loader import (
    default_dataset_path,
    load_and_validate,
    load_dataset,
    resolve_dataset_path,
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
from cronical.data.report import (
    ColumnQuality,
    DataQualityReport,
    IssueCode,
    Severity,
    ValidationIssue,
)
from cronical.data.schema import (
    COLUMN_SPECS,
    COLUMN_SPECS_BY_NAME,
    DATASET_FILENAME,
    FEATURE_COLUMNS,
    FEATURE_SPECS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    VALID_TARGET_VALUES,
    ZERO_SENTINEL_COLUMNS,
    ColumnSpec,
)
from cronical.data.validation import (
    documented_predictors,
    documented_target,
    missing_required_columns,
    raise_for_errors,
    validate_dataframe,
)

__all__ = [
    "COLUMN_SPECS",
    "COLUMN_SPECS_BY_NAME",
    "DATASET_FILENAME",
    "FEATURE_COLUMNS",
    "FEATURE_SPECS",
    "REQUIRED_COLUMNS",
    "TARGET_COLUMN",
    "VALID_TARGET_VALUES",
    "ZERO_SENTINEL_COLUMNS",
    "ColumnQuality",
    "ColumnSpec",
    "CronicalDataError",
    "DataQualityReport",
    "DatasetFormatError",
    "DatasetNotFoundError",
    "DatasetPathError",
    "DatasetReadError",
    "DatasetSplit",
    "DatasetValidationError",
    "FeatureContract",
    "FeatureContractError",
    "IssueCode",
    "PreprocessingError",
    "SentinelZeroHandler",
    "Severity",
    "SplitError",
    "ValidationIssue",
    "build_preprocessor",
    "build_training_pipeline",
    "default_dataset_path",
    "describe_preprocessor",
    "documented_predictors",
    "documented_target",
    "dump_preprocessor",
    "fit_preprocessor",
    "load_and_validate",
    "load_dataset",
    "load_preprocessor",
    "missing_required_columns",
    "raise_for_errors",
    "resolve_dataset_path",
    "save_preprocessor",
    "split_dataset",
    "transform_features",
    "transform_patient",
    "validate_dataframe",
]
