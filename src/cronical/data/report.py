"""Typed structures describing the quality of a dataset.

Every value in a report is computed from the dataset that was actually
inspected. Nothing is defaulted, estimated or carried over from a previous run,
so a report can always be regenerated and compared.

The report deliberately does not raise. Validation returns a report so that a
single pass surfaces *every* problem; a caller that prefers exceptions can opt in
via :func:`cronical.data.validation.raise_for_errors`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "ColumnQuality",
    "DataQualityReport",
    "IssueCode",
    "Severity",
    "ValidationIssue",
]


class Severity(StrEnum):
    """How much a finding should worry the caller."""

    ERROR = "error"
    WARNING = "warning"


def _render_key(value: float) -> str:
    """Render an outcome value as a JSON-friendly key without a trailing ``.0``."""
    return str(int(value)) if float(value).is_integer() else str(value)


class IssueCode(StrEnum):
    """Stable identifiers for every check, for grepping and for trend analysis.

    These strings are part of the project's internal contract: they are logged,
    counted and eventually surfaced in reports, so they must stay stable even if
    the wording of a message changes.
    """

    EMPTY_DATASET = "EMPTY_DATASET"
    MISSING_REQUIRED_COLUMN = "MISSING_REQUIRED_COLUMN"
    UNEXPECTED_TARGET_COLUMN = "UNEXPECTED_TARGET_COLUMN"
    UNEXPECTED_COLUMN = "UNEXPECTED_COLUMN"
    NON_NUMERIC_FEATURE = "NON_NUMERIC_FEATURE"
    NON_BINARY_TARGET = "NON_BINARY_TARGET"
    MISSING_VALUES = "MISSING_VALUES"
    COLUMN_WITHOUT_USABLE_VALUES = "COLUMN_WITHOUT_USABLE_VALUES"
    COLUMN_WITHOUT_USABLE_MEASUREMENTS = "COLUMN_WITHOUT_USABLE_MEASUREMENTS"
    ZERO_SENTINEL_VALUE = "ZERO_SENTINEL_VALUE"
    NEGATIVE_VALUE = "NEGATIVE_VALUE"
    NON_POSITIVE_VALUE = "NON_POSITIVE_VALUE"
    DUPLICATE_ROW = "DUPLICATE_ROW"


@dataclass(frozen=True, slots=True, order=True)
class ValidationIssue:
    """A single finding produced by a validation check.

    Attributes:
        code: Stable machine-readable identifier.
        severity: Whether the finding is fatal or advisory.
        message: Human-readable explanation, including the affected counts.
        column: The column the finding applies to, or ``None`` for whole-table
            findings such as duplicate rows.
        count: Number of affected cells or rows, when the check can quantify it.
    """

    code: IssueCode
    severity: Severity
    message: str
    column: str | None = None
    count: int | None = None

    def __str__(self) -> str:
        """Return a compact single-line rendering for logs."""
        location = f" [{self.column}]" if self.column else ""
        quantity = f" (n={self.count})" if self.count is not None else ""
        return f"[{self.severity.value.upper()}] {self.code}{location}{quantity}: {self.message}"


@dataclass(frozen=True, slots=True)
class ColumnQuality:
    """Per-column statistics computed from the dataset.

    Attributes:
        name: Column name as it appears in the dataset.
        dtype: pandas dtype, rendered as a string.
        missing_count: Number of null cells.
        missing_fraction: ``missing_count`` divided by the row count; ``0.0``
            when the dataset has no rows.
        unique_count: Number of distinct non-null values.
        zero_count: Number of cells equal to zero. Only meaningful for columns
            flagged as zero-sentinels; ``0`` elsewhere.
        minimum: Smallest non-null value, or ``None`` when the column is empty
            or wholly null.
        maximum: Largest non-null value, or ``None`` when the column is empty or
            wholly null.
        is_numeric: Whether pandas considers the column numeric.
    """

    name: str
    dtype: str
    missing_count: int
    missing_fraction: float
    unique_count: int
    zero_count: int
    minimum: float | None
    maximum: float | None
    is_numeric: bool = True


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """Everything learned about a dataset in a single validation pass.

    Attributes:
        source: Where the dataset came from, or ``None`` when validated in memory.
        row_count: Number of rows read.
        column_count: Number of columns read.
        column_names: Column names in the order they appear in the dataset.
        dtypes: Column name to rendered pandas dtype.
        missing_counts: Column name to null-cell count, for every column.
        zero_counts: Column name to zero-cell count, for the columns flagged as
            zero-sentinel columns.
        duplicate_count: Number of fully duplicated rows.
        target_distribution: Count of rows per distinct outcome value, for every
            value actually present. Values other than 0 or 1 are included so the
            report shows what is really in the file. Keys are floats so a
            value such as ``1.5`` is represented rather than truncated to ``1``.
        columns: Per-column statistics, aligned with ``column_names``.
        issues: Every finding, errors and warnings together, in check order.
    """

    source: str | None
    row_count: int
    column_count: int
    column_names: tuple[str, ...]
    dtypes: Mapping[str, str] = field(default_factory=dict)
    missing_counts: Mapping[str, int] = field(default_factory=dict)
    zero_counts: Mapping[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    target_distribution: Mapping[float, int] = field(default_factory=dict)
    columns: tuple[ColumnQuality, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Findings that make the dataset unusable without correction."""
        return tuple(issue for issue in self.issues if issue.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Findings a human should know about but which do not block use."""
        return tuple(issue for issue in self.issues if issue.severity is Severity.WARNING)

    @property
    def is_valid(self) -> bool:
        """Whether the dataset passed every fatal check."""
        return not self.errors

    @property
    def has_warnings(self) -> bool:
        """Whether any advisory finding was recorded."""
        return bool(self.warnings)

    def issue_counts(self) -> dict[str, int]:
        """Return the number of findings per issue code."""
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.code.value] = counts.get(issue.code.value, 0) + 1
        return counts

    def column(self, name: str) -> ColumnQuality | None:
        """Return the statistics recorded for ``name``, or ``None`` if absent."""
        for stats in self.columns:
            if stats.name == name:
                return stats
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the report.

        Suitable for writing to a report file or returning from an API. Keys and
        structure mirror the dataclass exactly.
        """
        return {
            "source": self.source,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "column_names": list(self.column_names),
            "dtypes": dict(self.dtypes),
            "missing_counts": dict(self.missing_counts),
            "zero_counts": dict(self.zero_counts),
            "duplicate_count": self.duplicate_count,
            "target_distribution": {
                _render_key(value): count for value, count in self.target_distribution.items()
            },
            "columns": [
                {
                    "name": column.name,
                    "dtype": column.dtype,
                    "missing_count": column.missing_count,
                    "missing_fraction": column.missing_fraction,
                    "unique_count": column.unique_count,
                    "zero_count": column.zero_count,
                    "minimum": column.minimum,
                    "maximum": column.maximum,
                    "is_numeric": column.is_numeric,
                }
                for column in self.columns
            ],
            "issues": [
                {
                    "code": issue.code.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "column": issue.column,
                    "count": issue.count,
                }
                for issue in self.issues
            ],
            "is_valid": self.is_valid,
        }

    def render(self) -> str:
        """Return a fixed-width text summary suitable for a log or a console.

        Returns:
            A multi-line report. No severity beyond ``ERROR``/``WARNING`` is
            implied, and no figure is added that is not already in the report.
        """
        lines = [
            "Data quality report",
            f"  source:      {self.source or '<in-memory DataFrame>'}",
            f"  rows:        {self.row_count}",
            f"  columns:     {self.column_count}",
            f"  duplicates:  {self.duplicate_count}",
            f"  valid:       {self.is_valid}",
        ]
        if self.target_distribution:
            distribution = ", ".join(
                f"{value}={count}" for value, count in sorted(self.target_distribution.items())
            )
            lines.append(f"  target:      {distribution}")

        if self.column_names:
            lines.append("  columns:")
            for column in self.columns:
                lines.append(
                    f"    - {column.name}: {column.dtype}, "
                    f"missing={column.missing_count}, zero={column.zero_count}, "
                    f"min={column.minimum}, max={column.maximum}"
                )

        if self.zero_counts:
            sentinel = ", ".join(f"{name}={count}" for name, count in self.zero_counts.items())
            lines.append(f"  zero-valued sentinel columns: {sentinel}")

        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)
