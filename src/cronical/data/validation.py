"""Validation of a dataset against the contract in :mod:`cronical.data.schema`.

Checks performed
----------------
Structure
    The dataset is non-empty; every required column is present; the outcome
    column has the expected name rather than a variant spelling; no unexpected
    columns are present.

Types
    Every predictor column is numeric, and the outcome column is numeric.

Content
    The outcome column contains only its documented values; no predictor holds a
    value that cannot be a real observation.

Reported, not fatal
    Duplicate rows, null cells, and ``0`` values standing in for unrecorded
    measurements in the columns where the dataset uses that sentinel.

Design rules
------------
* **Nothing is modified.** The input frame is never mutated, no value is
  imputed, and no row is dropped. Deciding what to do about a sentinel zero is a
  preprocessing decision, made later and recorded there.
* **Nothing is invented.** Every count in the report is computed from the frame
  that was passed in.
* **Severity is explicit.** Anything that makes the dataset unusable is an error;
  anything a human should know about is a warning. Because zero-sentinel values
  are a well-known property of this dataset, they are warnings rather than
  errors: a dataset containing them is still loadable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import pandas as pd

from cronical.data.errors import DatasetValidationError
from cronical.data.report import (
    ColumnQuality,
    DataQualityReport,
    IssueCode,
    Severity,
    ValidationIssue,
)
from cronical.data.schema import (
    COLUMN_SPECS_BY_NAME,
    FEATURE_SPECS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TARGET_COLUMN_ALIASES,
    TARGET_SPEC,
    VALID_TARGET_VALUES,
    ZERO_SENTINEL_COLUMNS,
)
from cronical.utils.logging import get_logger

__all__ = [
    "raise_for_errors",
    "validate_dataframe",
]

_LOG = get_logger(__name__)


def _issue(
    code: IssueCode,
    severity: Severity,
    message: str,
    *,
    column: str | None = None,
    count: int | None = None,
) -> ValidationIssue:
    """Build a single finding."""
    return ValidationIssue(
        code=code, severity=severity, message=message, column=column, count=count
    )


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return ``column`` as a series, typed for strict type checking."""
    extracted = frame[column]
    if not isinstance(extracted, pd.Series):  # pragma: no cover - defensive
        raise TypeError(f"Expected a Series for column '{column}', got {type(extracted)!r}")
    return extracted


def _missing_count(series: pd.Series) -> int:
    """Return the number of null cells."""
    return int(series.isna().sum())


def _count_equals(series: pd.Series, value: float) -> int:
    """Return how many cells hold exactly ``value``.

    Null cells are never equal to a number, so they are correctly excluded.
    """
    return int((series == value).sum())


def _extreme(series: pd.Series, *, maximum: bool) -> float | None:
    """Return the largest or smallest non-null value, or ``None`` if there is none."""
    if series.isna().all():
        return None
    value = series.max() if maximum else series.min()
    return float(value)


def _render_number(value: float) -> str:
    """Render an observed value for a message, without a spurious ``.0``.

    ``2.0`` reads as ``2`` in an error message; ``1.5`` stays ``1.5`` so the
    value really present in the file is what the reader sees.
    """
    return str(int(value)) if float(value).is_integer() else str(value)


def _column_quality(
    series: pd.Series,
    *,
    row_count: int,
    count_zeros: bool,
) -> ColumnQuality:
    """Compute the per-column statistics for one column."""
    missing = _missing_count(series)
    is_numeric = bool(pd.api.types.is_numeric_dtype(series))
    return ColumnQuality(
        name=str(series.name),
        dtype=str(series.dtype),
        missing_count=missing,
        missing_fraction=(missing / row_count) if row_count else 0.0,
        unique_count=int(series.nunique(dropna=True)),
        zero_count=_count_equals(series, 0) if count_zeros else 0,
        minimum=_extreme(series, maximum=False) if is_numeric else None,
        maximum=_extreme(series, maximum=True) if is_numeric else None,
        is_numeric=is_numeric,
    )


def _check_structure(frame: pd.DataFrame, issues: list[ValidationIssue]) -> bool:
    """Check that the dataset has a usable shape and the required columns.

    Returns:
        ``True`` when per-column checks may proceed.
    """
    if frame.empty or len(frame.columns) == 0:
        issues.append(
            _issue(
                IssueCode.EMPTY_DATASET,
                Severity.ERROR,
                "Dataset contains no rows or no columns.",
                count=len(frame.index),
            )
        )
        return False

    missing_columns = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    for name in missing_columns:
        issues.append(
            _issue(
                IssueCode.MISSING_REQUIRED_COLUMN,
                Severity.ERROR,
                f"Required column '{name}' is absent from the dataset.",
                column=name,
            )
        )

    if TARGET_COLUMN not in frame.columns:
        _report_unexpected_target(frame, issues)

    for name in frame.columns:
        if str(name) not in COLUMN_SPECS_BY_NAME:
            issues.append(
                _issue(
                    IssueCode.UNEXPECTED_COLUMN,
                    Severity.WARNING,
                    f"Column '{name}' is not part of the documented schema. "
                    "Unexpected columns can leak information into a model and "
                    "should be removed deliberately, not by accident.",
                    column=str(name),
                )
            )

    return True


def _report_unexpected_target(frame: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    """Explain that the outcome column is named unexpectedly, if we can tell."""
    aliases = sorted(
        str(name) for name in frame.columns if str(name).strip().lower() in TARGET_COLUMN_ALIASES
    )
    detail = (
        f" Found a likely outcome column named {', '.join(repr(name) for name in aliases)}; "
        f"rename it to '{TARGET_COLUMN}' rather than accepting a variant."
        if aliases
        else ""
    )
    issues.append(
        _issue(
            IssueCode.UNEXPECTED_TARGET_COLUMN,
            Severity.ERROR,
            f"Expected the outcome column to be named '{TARGET_COLUMN}'.{detail}",
            column=TARGET_COLUMN,
        )
    )


def _check_missing_values(
    series: pd.Series,
    *,
    issues: list[ValidationIssue],
) -> None:
    """Report null cells, escalating to an error when nothing is usable."""
    missing = _missing_count(series)
    if missing == 0:
        return
    name = str(series.name)
    if missing == len(series.index):
        issues.append(
            _issue(
                IssueCode.COLUMN_WITHOUT_USABLE_VALUES,
                Severity.ERROR,
                f"Column '{name}' has no non-null values, so it cannot be used.",
                column=name,
                count=missing,
            )
        )
        return
    issues.append(
        _issue(
            IssueCode.MISSING_VALUES,
            Severity.WARNING,
            f"Column '{name}' contains {missing} null value(s).",
            column=name,
            count=missing,
        )
    )


def _check_zero_sentinels(
    series: pd.Series,
    *,
    issues: list[ValidationIssue],
) -> int:
    """Report zero values in a column where zero means "not recorded".

    Returns:
        The number of zero cells, for inclusion in the report.
    """
    name = str(series.name)
    zeros = _count_equals(series, 0)
    usable = len(series.index) - _missing_count(series)

    if zeros == 0:
        return 0

    if usable > 0 and zeros == usable:
        issues.append(
            _issue(
                IssueCode.COLUMN_WITHOUT_USABLE_MEASUREMENTS,
                Severity.ERROR,
                f"Every value in '{name}' is the zero placeholder used by this "
                "dataset for a missing measurement, so the column carries no "
                "usable measurement at all.",
                column=name,
                count=zeros,
            )
        )
        return zeros

    issues.append(
        _issue(
            IssueCode.ZERO_SENTINEL_VALUE,
            Severity.WARNING,
            f"Column '{name}' contains {zeros} zero value(s). In this dataset a "
            "zero here means the measurement was not recorded, not that the "
            "measured value is zero. It must be handled explicitly during "
            "preprocessing; it is left untouched here.",
            column=name,
            count=zeros,
        )
    )
    return zeros


def _check_invalid_values(
    series: pd.Series,
    *,
    requires_positive: bool,
    issues: list[ValidationIssue],
) -> None:
    """Report values that cannot be real observations for this column.

    Negative values are impossible for every column in this schema, so they are
    always reported. Zero is additionally reported for the columns whose
    published definition requires a strictly positive quantity; elsewhere zero is
    ordinary data and is left alone.
    """
    name = str(series.name)

    negative = int((series < 0).sum())
    if negative:
        issues.append(
            _issue(
                IssueCode.NEGATIVE_VALUE,
                Severity.ERROR,
                f"Column '{name}' contains {negative} negative value(s), which is "
                "not possible for this quantity.",
                column=name,
                count=negative,
            )
        )

    if not requires_positive:
        return

    zeros = _count_equals(series, 0)
    if zeros:
        issues.append(
            _issue(
                IssueCode.NON_POSITIVE_VALUE,
                Severity.ERROR,
                f"Column '{name}' contains {zeros} zero value(s); this quantity "
                "must be greater than zero.",
                column=name,
                count=zeros,
            )
        )


def _check_target(frame: pd.DataFrame, issues: list[ValidationIssue]) -> dict[float, int]:
    """Validate the outcome column and return its observed distribution.

    Distribution keys are floats so that an unexpected value such as ``1.5`` is
    represented exactly as observed. Truncating it to ``1`` here would hide a
    real problem behind a plausible-looking class count.
    """
    if TARGET_COLUMN not in frame.columns:
        return {}

    series = _series(frame, TARGET_COLUMN)
    if not pd.api.types.is_numeric_dtype(series):
        issues.append(
            _issue(
                IssueCode.NON_NUMERIC_FEATURE,
                Severity.ERROR,
                f"Outcome column '{TARGET_COLUMN}' must be numeric but is '{series.dtype}'.",
                column=TARGET_COLUMN,
            )
        )
        return {}

    counts: dict[float, int] = {}
    for value, count in series.value_counts().items():
        counts[float(cast("float", value))] = int(count)

    unexpected = sorted(value for value in counts if value not in VALID_TARGET_VALUES)
    if unexpected:
        rendered = ", ".join(_render_number(value) for value in unexpected)
        issues.append(
            _issue(
                IssueCode.NON_BINARY_TARGET,
                Severity.ERROR,
                f"Outcome column '{TARGET_COLUMN}' must contain only "
                f"{sorted(VALID_TARGET_VALUES)} but also contains {rendered}.",
                column=TARGET_COLUMN,
                count=sum(counts[value] for value in unexpected),
            )
        )
    return counts


def _as_int_mapping(mapping: Mapping[str, Any]) -> dict[str, int]:
    """Normalise a count mapping to plain ``str`` keys and ``int`` values."""
    return {str(key): int(value) for key, value in mapping.items()}


def _inspect_column(
    series: pd.Series,
    *,
    row_count: int,
    expected: bool,
    proceed: bool,
    issues: list[ValidationIssue],
) -> tuple[ColumnQuality, int | None]:
    """Run every applicable check for one column.

    Args:
        series: The column to inspect.
        row_count: Row count of the whole dataset, for fractions.
        expected: Whether the column is part of the documented schema.
        proceed: Whether structural checks passed far enough to judge content.
        issues: Collector for findings.

    Returns:
        The column's statistics, and its zero count when the column is a
        zero-sentinel column (``None`` otherwise).
    """
    name = str(series.name)
    is_numeric = bool(pd.api.types.is_numeric_dtype(series))
    spec = COLUMN_SPECS_BY_NAME.get(name)
    is_sentinel = name in ZERO_SENTINEL_COLUMNS

    # Null counts are always worth reporting, even when the shape is wrong.
    _check_missing_values(series, issues=issues)

    zero_count: int | None = None
    if proceed and expected and spec is not None:
        _check_column_types(series, issues=issues)
        if is_numeric:
            if is_sentinel:
                zero_count = _check_zero_sentinels(series, issues=issues)
            _check_invalid_values(series, requires_positive=spec.must_be_positive, issues=issues)

    quality = _column_quality(
        series,
        row_count=row_count,
        count_zeros=is_sentinel and is_numeric,
    )
    return quality, zero_count


def validate_dataframe(frame: pd.DataFrame, *, source: str | None = None) -> DataQualityReport:
    """Validate ``frame`` against the documented schema and describe its quality.

    The frame is only ever read: no column is added, removed or changed, and no
    value is imputed.

    Args:
        frame: The dataset to validate.
        source: Optional provenance label recorded in the report, such as a file
            path, so a report can be traced back to its input.

    Returns:
        A :class:`~cronical.data.report.DataQualityReport` holding the observed
        shape, per-column statistics and every finding. Inspect
        :attr:`~cronical.data.report.DataQualityReport.is_valid` to decide
        whether the dataset may be used.
    """
    issues: list[ValidationIssue] = []
    row_count = len(frame.index)
    column_names = tuple(str(name) for name in frame.columns)

    dtypes = {str(name): str(dtype) for name, dtype in frame.dtypes.items()}
    missing_counts = _as_int_mapping(frame.isna().sum().to_dict())

    duplicates = int(frame.duplicated().sum())
    if duplicates:
        issues.append(
            _issue(
                IssueCode.DUPLICATE_ROW,
                Severity.WARNING,
                f"Dataset contains {duplicates} fully duplicated row(s).",
                count=duplicates,
            )
        )

    proceed = _check_structure(frame, issues)

    columns: list[ColumnQuality] = []
    zero_counts: dict[str, int] = {}
    for name in column_names:
        quality, zero_count = _inspect_column(
            _series(frame, name),
            row_count=row_count,
            expected=str(name) in COLUMN_SPECS_BY_NAME,
            proceed=proceed,
            issues=issues,
        )
        columns.append(quality)
        if zero_count is not None:
            zero_counts[name] = zero_count

    target_distribution = _check_target(frame, issues) if proceed else {}

    report = DataQualityReport(
        source=source,
        row_count=row_count,
        column_count=len(column_names),
        column_names=column_names,
        dtypes=dtypes,
        missing_counts=missing_counts,
        zero_counts=zero_counts,
        duplicate_count=duplicates,
        target_distribution=target_distribution,
        columns=tuple(columns),
        issues=tuple(issues),
    )
    _LOG.info(
        "dataset validated",
        extra={
            "rows": report.row_count,
            "columns": report.column_count,
            "errors": len(report.errors),
            "warnings": len(report.warnings),
            "duplicates": report.duplicate_count,
            "source": source,
        },
    )
    return report


def _check_column_types(series: pd.Series, issues: list[ValidationIssue]) -> None:
    """Report a column whose dtype cannot be numeric."""
    if pd.api.types.is_numeric_dtype(series):
        return
    name = str(series.name)
    issues.append(
        _issue(
            IssueCode.NON_NUMERIC_FEATURE,
            Severity.ERROR,
            f"Column '{name}' must be numeric but is '{series.dtype}'.",
            column=name,
        )
    )


def raise_for_errors(report: DataQualityReport) -> DataQualityReport:
    """Return ``report`` unchanged, or raise if it recorded any error.

    Validation itself never raises, so a caller can inspect all findings first.
    This helper exists for call sites that would rather fail fast.

    Args:
        report: The report to assert on.

    Returns:
        The same report, when it contains no errors.

    Raises:
        DatasetValidationError: If the report records at least one error. The
            exception carries the report, so the caller still sees every finding.
    """
    if report.errors:
        raise DatasetValidationError(report)
    return report


def missing_required_columns(frame: pd.DataFrame) -> Sequence[str]:
    """Return the required columns absent from ``frame``, in schema order.

    A small convenience for callers that want to report a shortfall without
    building a full report.
    """
    present = {str(name) for name in frame.columns}
    return [name for name in REQUIRED_COLUMNS if name not in present]


def documented_predictors() -> Sequence[str]:
    """Return the predictor column names, in published order."""
    return tuple(spec.name for spec in FEATURE_SPECS)


def documented_target() -> str:
    """Return the expected outcome column name."""
    return TARGET_SPEC.name
