"""Typed exceptions for the Cronical data layer.

Every failure mode the data layer can produce is represented here, so callers can
distinguish "the file is missing" from "the file is the wrong format" from "the
contents violate the schema" without parsing message strings.

Hierarchy::

    CronicalDataError
    +-- DatasetPathError
    |   +-- DatasetNotFoundError
    +-- DatasetFormatError
    +-- DatasetReadError
    +-- DatasetValidationError
    +-- PreprocessingError
        +-- FeatureContractError
        +-- SplitError

The split matters: :class:`DatasetValidationError` carries a full
:class:`~cronical.data.report.DataQualityReport`, so a caller can inspect every
problem found in one pass instead of fixing issues one exception at a time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from cronical.data.report import DataQualityReport

__all__ = [
    "CronicalDataError",
    "DatasetFormatError",
    "DatasetNotFoundError",
    "DatasetPathError",
    "DatasetReadError",
    "DatasetValidationError",
    "FeatureContractError",
    "PreprocessingError",
    "SplitError",
]

#: Where a developer should look when a dataset is missing.
_PLACEMENT_HINT = (
    "Place the CSV in the project's data/raw/ directory (see data/raw/README.md). "
    "The expected filename is 'diabetes.csv'. This project never downloads data "
    "automatically."
)


class CronicalDataError(Exception):
    """Base class for every error raised by :mod:`cronical.data`."""


class DatasetPathError(CronicalDataError):
    """The supplied dataset path cannot be used."""


class DatasetNotFoundError(DatasetPathError):
    """No dataset file exists at the requested location.

    Args:
        path: The resolved path that was checked.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Dataset file not found: {path}. {_PLACEMENT_HINT}")


class DatasetFormatError(CronicalDataError):
    """The supplied file is not a supported dataset format.

    Raised on the *request*, before any filesystem access beyond a suffix check,
    so the message describes what the caller asked for rather than what happens
    to be on disk.
    """

    def __init__(self, path: Path, *, expected_suffix: str) -> None:
        self.path = path
        self.expected_suffix = expected_suffix
        actual = path.suffix or "<none>"
        super().__init__(
            f"Unsupported dataset format for '{path.name}': expected "
            f"'{expected_suffix}' but found '{actual}'. Only delimited text "
            f"({expected_suffix}) is supported at this stage."
        )


class DatasetReadError(CronicalDataError):
    """The dataset file exists but could not be parsed into a table.

    Args:
        path: The file that failed to parse.
        reason: Human-readable explanation taken from the underlying parser.
    """

    def __init__(self, path: Path, *, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Could not read dataset '{path}': {reason}")


class DatasetValidationError(CronicalDataError):
    """A dataset violates the schema contract in :mod:`cronical.data.schema`.

    This exception is raised only when a caller explicitly asks for failures to
    be raised; plain validation returns a report instead.

    Args:
        report: The full quality report, including every error and warning.
    """

    def __init__(self, report: DataQualityReport) -> None:
        self.report = report
        errors = report.errors
        summary = "; ".join(f"{issue.code}: {issue.message}" for issue in errors)
        super().__init__(f"Dataset failed {len(errors)} validation check(s): {summary}")


class PreprocessingError(CronicalDataError):
    """A feature-preparation step could not be completed."""


class FeatureContractError(PreprocessingError):
    """The feature frame does not satisfy the documented schema.

    Raised for a missing required column, a column that is not numeric, or input
    whose shape cannot be aligned with the schema at all.

    Args:
        message: Explanation naming the offending columns or shape.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(f"Feature contract violated: {message}")


class SplitError(PreprocessingError):
    """A train/test split cannot be produced from the supplied dataset.

    Raised before any splitting is attempted, for conditions such as a class with
    too few members to appear in both partitions.

    Args:
        message: Explanation of what makes the split impossible.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(f"Cannot split dataset: {message}")
