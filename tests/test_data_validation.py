"""Tests for :mod:`cronical.data.validation`.

Every expected count below is derived from rows written out in full in this
module or in ``conftest.py``. No test loads the real dataset, and no statistic
here describes it.

The theme running through these tests: validation *reports*, it never *repairs*.
Where a rule could plausibly have been implemented by silently fixing the data,
the test asserts the opposite.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import pandas as pd
import pytest

from cronical.data.errors import DatasetValidationError
from cronical.data.report import (
    ColumnQuality,
    DataQualityReport,
    IssueCode,
    Severity,
    ValidationIssue,
)
from cronical.data.schema import (
    DATASET_FILENAME,
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    VALID_TARGET_VALUES,
    ZERO_SENTINEL_COLUMNS,
)
from cronical.data.validation import (
    documented_predictors,
    documented_target,
    missing_required_columns,
    raise_for_errors,
    validate_dataframe,
)

from .conftest import CLEAN_ROWS, HEADER

#: A cell value a test may deliberately supply. Strings are included because
#: several tests need a non-numeric column on purpose.
Cell = int | float | str | None


def _rows_to_frame(rows: Sequence[Sequence[Cell]], columns: Sequence[str]) -> pd.DataFrame:
    """Build a frame from row values against an explicit column list."""
    return pd.DataFrame(list(rows), columns=list(columns))


def _rename_target(rows: tuple[tuple[int | float, ...], ...], alias: str) -> pd.DataFrame:
    """Build a valid frame whose outcome column carries a different name.

    Used to exercise the "unexpected target column" rule. Only the header
    changes; the values are the shared clean rows.
    """
    columns = (*[name for name in REQUIRED_COLUMNS if name != TARGET_COLUMN], alias)
    return pd.DataFrame(rows, columns=list(columns))


def _codes(report: DataQualityReport, severity: Severity | None = None) -> set[IssueCode]:
    """Return the issue codes in a report, optionally filtered by severity."""
    issues = (
        report.issues
        if severity is None
        else (report.errors if severity is Severity.ERROR else report.warnings)
    )
    return {issue.code for issue in issues}


def _issue_for(report: DataQualityReport, code: IssueCode) -> ValidationIssue:
    """Return the single finding with ``code``, failing loudly if absent or duplicated."""
    matches = [issue for issue in report.issues if issue.code is code]
    assert len(matches) == 1, f"expected exactly one {code}, got {matches}"
    return matches[0]


class TestCleanDataset:
    """A dataset with no findings at all."""

    def test_clean_frame_has_no_errors(self, clean_frame: pd.DataFrame) -> None:
        assert validate_dataframe(clean_frame).errors == ()

    def test_clean_frame_has_no_warnings(self, clean_frame: pd.DataFrame) -> None:
        assert validate_dataframe(clean_frame).warnings == ()

    def test_clean_frame_is_valid(self, clean_frame: pd.DataFrame) -> None:
        report = validate_dataframe(clean_frame)
        assert report.is_valid
        assert not report.has_warnings

    def test_shape_is_recorded(self, clean_frame: pd.DataFrame) -> None:
        report = validate_dataframe(clean_frame)
        assert report.row_count == len(CLEAN_ROWS)
        assert report.column_count == len(REQUIRED_COLUMNS)

    def test_column_names_are_recorded_in_file_order(self, clean_frame: pd.DataFrame) -> None:
        assert validate_dataframe(clean_frame).column_names == REQUIRED_COLUMNS

    def test_dtypes_are_recorded(self, clean_frame: pd.DataFrame) -> None:
        report = validate_dataframe(clean_frame)
        assert set(report.dtypes) == set(REQUIRED_COLUMNS)
        assert all(isinstance(value, str) for value in report.dtypes.values())

    def test_missing_counts_are_recorded(self, clean_frame: pd.DataFrame) -> None:
        report = validate_dataframe(clean_frame)
        assert set(report.missing_counts.values()) == {0}

    def test_duplicate_count_is_recorded(self, clean_frame: pd.DataFrame) -> None:
        assert validate_dataframe(clean_frame).duplicate_count == 0

    def test_target_distribution_is_recorded(self, clean_frame: pd.DataFrame) -> None:
        distribution = validate_dataframe(clean_frame).target_distribution
        assert distribution == {0: 2, 1: 2}

    def test_zero_counts_are_zero_for_a_clean_sentinel_column(
        self, clean_frame: pd.DataFrame
    ) -> None:
        assert set(validate_dataframe(clean_frame).zero_counts.values()) == {0}

    def test_zero_counts_cover_every_sentinel_column(self, clean_frame: pd.DataFrame) -> None:
        report = validate_dataframe(clean_frame)
        assert set(report.zero_counts) == set(ZERO_SENTINEL_COLUMNS)

    def test_per_column_statistics_are_produced(self, clean_frame: pd.DataFrame) -> None:
        columns = validate_dataframe(clean_frame).columns
        assert len(columns) == len(REQUIRED_COLUMNS)
        assert all(isinstance(column, ColumnQuality) for column in columns)

    def test_extremes_are_computed_from_the_data(self, clean_frame: pd.DataFrame) -> None:
        glucose = validate_dataframe(clean_frame).column("Glucose")
        assert glucose is not None
        assert glucose.minimum == 85.0
        assert glucose.maximum == 183.0
        assert glucose.unique_count == 4
        assert glucose.missing_fraction == 0.0

    def test_source_is_recorded(self, clean_frame: pd.DataFrame) -> None:
        assert validate_dataframe(clean_frame, source="unit-test").source == "unit-test"

    def test_source_defaults_to_none(self, clean_frame: pd.DataFrame) -> None:
        assert validate_dataframe(clean_frame).source is None


class TestNonMutating:
    """Validation is read-only."""

    def test_frame_is_unchanged_by_validation(self, clean_frame: pd.DataFrame) -> None:
        before = clean_frame.copy(deep=True)
        validate_dataframe(clean_frame)
        pd.testing.assert_frame_equal(clean_frame, before)

    def test_zero_sentinels_are_left_in_place(self) -> None:
        frame = _rows_to_frame(
            [
                [1, 0, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, 100, 0, 0, 0, 0.0, 0.5, 40, 0],
            ],
            REQUIRED_COLUMNS,
        )
        before = frame.copy(deep=True)
        validate_dataframe(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_nulls_are_not_imputed(self) -> None:
        frame = _rows_to_frame(
            [
                [1, None, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, 100, 65, 15, 85, 24.0, 0.5, 40, 0],
            ],
            REQUIRED_COLUMNS,
        )
        validate_dataframe(frame)
        assert frame["Glucose"].isna().sum() == 1


class TestStructuralChecks:
    """Shape and column-name rules."""

    def test_empty_frame_is_an_error(self) -> None:
        report = validate_dataframe(pd.DataFrame())
        assert IssueCode.EMPTY_DATASET in _codes(report)
        assert not report.is_valid

    def test_header_only_frame_is_an_error(self) -> None:
        report = validate_dataframe(pd.DataFrame(columns=list(REQUIRED_COLUMNS)))
        assert IssueCode.EMPTY_DATASET in _codes(report)

    def test_empty_frame_reports_zero_rows(self) -> None:
        report = validate_dataframe(pd.DataFrame())
        assert report.row_count == 0
        assert report.column_count == 0
        assert report.target_distribution == {}

    def test_missing_fraction_is_zero_without_rows(self) -> None:
        report = validate_dataframe(pd.DataFrame())
        assert all(column.missing_fraction == 0.0 for column in report.columns)

    def test_missing_required_column_is_an_error(self, clean_frame: pd.DataFrame) -> None:
        frame = clean_frame.drop(columns=["Insulin"])
        report = validate_dataframe(frame)
        assert IssueCode.MISSING_REQUIRED_COLUMN in _codes(report)
        assert not report.is_valid

    def test_missing_column_is_named_in_the_finding(self, clean_frame: pd.DataFrame) -> None:
        report = validate_dataframe(clean_frame.drop(columns=["Insulin"]))
        assert _issue_for(report, IssueCode.MISSING_REQUIRED_COLUMN).column == "Insulin"

    def test_every_missing_column_is_reported(self, clean_frame: pd.DataFrame) -> None:
        frame = clean_frame.drop(columns=["Insulin", "BMI", "Age"])
        report = validate_dataframe(frame)
        missing = [
            issue.column
            for issue in report.issues
            if issue.code is IssueCode.MISSING_REQUIRED_COLUMN
        ]
        assert missing == ["Insulin", "BMI", "Age"]

    @pytest.mark.parametrize("alias", ["class", "ClassVariable", "target", "DIABETES"])
    def test_unexpected_target_name_is_an_error(self, alias: str) -> None:
        report = validate_dataframe(_rename_target(CLEAN_ROWS, alias))
        assert IssueCode.UNEXPECTED_TARGET_COLUMN in _codes(report)
        assert not report.is_valid

    def test_unexpected_target_finding_suggests_the_alias_found(self) -> None:
        report = validate_dataframe(_rename_target(CLEAN_ROWS, "class"))
        assert "'class'" in _issue_for(report, IssueCode.UNEXPECTED_TARGET_COLUMN).message

    def test_unexpected_target_without_a_known_alias(self) -> None:
        report = validate_dataframe(_rename_target(CLEAN_ROWS, "label_42"))
        assert IssueCode.UNEXPECTED_TARGET_COLUMN in _codes(report)

    def test_extra_column_is_a_warning(self, clean_frame: pd.DataFrame) -> None:
        frame = clean_frame.assign(SiteCode="ABC")
        report = validate_dataframe(frame)
        assert IssueCode.UNEXPECTED_COLUMN in _codes(report, Severity.WARNING)
        assert report.is_valid, "an unknown column is reported but not fatal"

    def test_extra_column_is_named(self, clean_frame: pd.DataFrame) -> None:
        report = validate_dataframe(clean_frame.assign(SiteCode="ABC"))
        assert _issue_for(report, IssueCode.UNEXPECTED_COLUMN).column == "SiteCode"


class TestTypeChecks:
    """Dtype rules."""

    def test_non_numeric_feature_is_an_error(self) -> None:
        frame = _rows_to_frame(
            [[1, "high", 70, 20, 90, 25.0, 0.4, 30, 1]],
            REQUIRED_COLUMNS,
        )
        report = validate_dataframe(frame)
        assert IssueCode.NON_NUMERIC_FEATURE in _codes(report, Severity.ERROR)
        assert not report.is_valid

    def test_non_numeric_feature_is_named(self) -> None:
        frame = _rows_to_frame(
            [[1, "high", 70, 20, 90, 25.0, 0.4, 30, 1]],
            REQUIRED_COLUMNS,
        )
        report = validate_dataframe(frame)
        assert _issue_for(report, IssueCode.NON_NUMERIC_FEATURE).column == "Glucose"

    def test_non_numeric_column_has_no_extremes(self) -> None:
        frame = _rows_to_frame(
            [["low", "high", 70, 20, 90, 25.0, 0.4, 30, 1]],
            REQUIRED_COLUMNS,
        )
        glucose = validate_dataframe(frame).column("Glucose")
        assert glucose is not None
        assert not glucose.is_numeric
        assert glucose.minimum is None
        assert glucose.maximum is None

    def test_non_numeric_target_is_an_error(self) -> None:
        frame = _rows_to_frame(
            [[1, 100, 70, 20, 90, 25.0, 0.4, 30, "positive"]],
            REQUIRED_COLUMNS,
        )
        report = validate_dataframe(frame)
        assert IssueCode.NON_NUMERIC_FEATURE in _codes(report)
        assert report.target_distribution == {}


class TestTargetChecks:
    """Outcome-column rules."""

    @pytest.mark.parametrize("bad_value", [2, 7, -1])
    def test_unexpected_target_values_are_an_error(self, bad_value: int) -> None:
        frame = _rows_to_frame(
            [
                [1, 100, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, 110, 65, 15, 85, 24.0, 0.5, 40, bad_value],
            ],
            REQUIRED_COLUMNS,
        )
        report = validate_dataframe(frame)
        assert IssueCode.NON_BINARY_TARGET in _codes(report)
        assert not report.is_valid

    def test_fractional_target_is_reported_exactly(self) -> None:
        """1.5 must not be silently rounded into a valid class of 1."""
        frame = _rows_to_frame(
            [
                [1, 100, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, 110, 65, 15, 85, 24.0, 0.5, 40, 1.5],
            ],
            REQUIRED_COLUMNS,
        )
        report = validate_dataframe(frame)
        assert 1.5 in report.target_distribution
        assert 1.5 not in {0.0, 1.0}
        assert "1.5" in _issue_for(report, IssueCode.NON_BINARY_TARGET).message

    def test_invalid_target_count_is_reported(self) -> None:
        frame = _rows_to_frame(
            [
                [1, 100, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, 110, 65, 15, 85, 24.0, 0.5, 40, 5],
                [3, 120, 60, 25, 95, 26.0, 0.6, 50, 5],
            ],
            REQUIRED_COLUMNS,
        )
        assert _issue_for(validate_dataframe(frame), IssueCode.NON_BINARY_TARGET).count == 2

    def test_integral_float_target_is_accepted(self) -> None:
        """A CSV that parses the outcome as float is still a valid 0/1 column."""
        frame = _rows_to_frame(
            [
                [1, 100, 70, 20, 90, 25.0, 0.4, 30, 1.0],
                [2, 110, 65, 15, 85, 24.0, 0.5, 40, 0.0],
            ],
            REQUIRED_COLUMNS,
        )
        report = validate_dataframe(frame)
        assert IssueCode.NON_BINARY_TARGET not in _codes(report)
        assert report.target_distribution == {0.0: 1, 1.0: 1}

    def test_valid_target_values_are_the_documented_pair(self) -> None:
        assert set(VALID_TARGET_VALUES) == {0, 1}


class TestMissingValues:
    """Null-cell rules."""

    def test_missing_values_are_a_warning(self) -> None:
        frame = _rows_to_frame(
            [
                [1, None, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, 110, 65, 15, 85, 24.0, 0.5, 40, 0],
            ],
            REQUIRED_COLUMNS,
        )
        report = validate_dataframe(frame)
        assert IssueCode.MISSING_VALUES in _codes(report, Severity.WARNING)
        assert report.is_valid, "a null in an otherwise usable column is not fatal"

    def test_missing_count_is_accurate(self) -> None:
        frame = _rows_to_frame(
            [
                [1, None, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, None, 65, 15, 85, 24.0, 0.5, 40, 0],
                [3, 120, 60, 25, 95, 26.0, 0.6, 50, 1],
            ],
            REQUIRED_COLUMNS,
        )
        report = validate_dataframe(frame)
        assert report.missing_counts["Glucose"] == 2
        assert _issue_for(report, IssueCode.MISSING_VALUES).count == 2

    def test_missing_fraction_is_accurate(self) -> None:
        frame = _rows_to_frame(
            [
                [1, None, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, 110, 65, 15, 85, 24.0, 0.5, 40, 0],
            ],
            REQUIRED_COLUMNS,
        )
        glucose = validate_dataframe(frame).column("Glucose")
        assert glucose is not None
        assert glucose.missing_fraction == 0.5

    def test_entirely_null_column_is_an_error(self) -> None:
        frame = _rows_to_frame(
            [
                [1, None, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, None, 65, 15, 85, 24.0, 0.5, 40, 0],
            ],
            REQUIRED_COLUMNS,
        )
        report = validate_dataframe(frame)
        assert IssueCode.COLUMN_WITHOUT_USABLE_VALUES in _codes(report, Severity.ERROR)
        assert not report.is_valid

    def test_entirely_null_column_has_no_extremes(self) -> None:
        frame = _rows_to_frame(
            [
                [1, None, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, None, 65, 15, 85, 24.0, 0.5, 40, 0],
            ],
            REQUIRED_COLUMNS,
        )
        glucose = validate_dataframe(frame).column("Glucose")
        assert glucose is not None
        assert glucose.minimum is None
        assert glucose.maximum is None
        assert glucose.unique_count == 0

    def test_numeric_column_of_only_nan_reports_no_extremes(self) -> None:
        """A column that parsed as numeric but holds nothing must not claim 0.0."""
        nan = math.nan
        frame = _rows_to_frame(
            [
                [1, nan, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, nan, 65, 15, 85, 24.0, 0.5, 40, 0],
            ],
            REQUIRED_COLUMNS,
        )
        assert pd.api.types.is_numeric_dtype(frame["Glucose"])
        glucose = validate_dataframe(frame).column("Glucose")
        assert glucose is not None
        assert glucose.is_numeric
        assert glucose.minimum is None
        assert glucose.maximum is None
        assert glucose.missing_count == 2


class TestZeroSentinels:
    """Zero used as a "not recorded" placeholder."""

    def test_zero_in_sentinel_column_is_a_warning(self) -> None:
        frame = _rows_to_frame(
            [
                [1, 0, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, 110, 65, 15, 85, 24.0, 0.5, 40, 0],
            ],
            REQUIRED_COLUMNS,
        )
        report = validate_dataframe(frame)
        assert IssueCode.ZERO_SENTINEL_VALUE in _codes(report, Severity.WARNING)
        assert report.is_valid

    @pytest.mark.parametrize("column", ZERO_SENTINEL_COLUMNS)
    def test_every_sentinel_column_is_checked(self, column: str) -> None:
        rows = [list(row) for row in CLEAN_ROWS]
        rows[0][FEATURE_COLUMNS.index(column)] = 0
        report = validate_dataframe(_rows_to_frame(rows, REQUIRED_COLUMNS))
        assert report.zero_counts[column] == 1
        assert _issue_for(report, IssueCode.ZERO_SENTINEL_VALUE).column == column

    def test_zero_count_is_accurate(self) -> None:
        frame = _rows_to_frame(
            [
                [1, 0, 70, 20, 0, 25.0, 0.4, 30, 1],
                [2, 110, 0, 15, 85, 0.0, 0.5, 40, 0],
            ],
            REQUIRED_COLUMNS,
        )
        report = validate_dataframe(frame)
        assert report.zero_counts == {
            "Glucose": 1,
            "BloodPressure": 1,
            "SkinThickness": 0,
            "Insulin": 1,
            "BMI": 1,
        }

    def test_sentinel_columns_are_exactly_the_documented_five(self) -> None:
        assert ZERO_SENTINEL_COLUMNS == (
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
        )

    def test_zero_in_a_non_sentinel_column_is_left_alone(self) -> None:
        """``Pregnancies`` of zero is ordinary data, not a placeholder."""
        rows = [list(row) for row in CLEAN_ROWS]
        rows[0][FEATURE_COLUMNS.index("Pregnancies")] = 0
        report = validate_dataframe(_rows_to_frame(rows, REQUIRED_COLUMNS))
        assert IssueCode.ZERO_SENTINEL_VALUE not in _codes(report)
        assert report.is_valid and not report.warnings

    def test_entirely_zero_sentinel_column_is_an_error(self) -> None:
        frame = _rows_to_frame(
            [
                [1, 0, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, 0, 65, 15, 85, 24.0, 0.5, 40, 0],
            ],
            REQUIRED_COLUMNS,
        )
        report = validate_dataframe(frame)
        assert IssueCode.COLUMN_WITHOUT_USABLE_MEASUREMENTS in _codes(report, Severity.ERROR)
        assert not report.is_valid

    def test_all_null_sentinel_column_reports_nulls_not_sentinels(self) -> None:
        frame = _rows_to_frame(
            [
                [1, None, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, None, 65, 15, 85, 24.0, 0.5, 40, 0],
            ],
            REQUIRED_COLUMNS,
        )
        report = validate_dataframe(frame)
        assert IssueCode.ZERO_SENTINEL_VALUE not in _codes(report)
        assert IssueCode.COLUMN_WITHOUT_USABLE_VALUES in _codes(report)

    def test_non_numeric_sentinel_column_is_not_counted(self) -> None:
        frame = _rows_to_frame(
            [["1", "0", "70", "20", "90", "25.0", "0.4", "30", "1"]],
            REQUIRED_COLUMNS,
        )
        report = validate_dataframe(frame)
        assert "Glucose" not in report.zero_counts
        assert IssueCode.ZERO_SENTINEL_VALUE not in _codes(report)


class TestInvalidValues:
    """Values that cannot be real observations."""

    def test_negative_value_is_an_error(self) -> None:
        frame = _rows_to_frame(
            [
                [1, -50, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, 110, 65, 15, 85, 24.0, 0.5, 40, 0],
            ],
            REQUIRED_COLUMNS,
        )
        report = validate_dataframe(frame)
        assert IssueCode.NEGATIVE_VALUE in _codes(report)
        assert not report.is_valid

    def test_negative_value_is_named_and_counted(self) -> None:
        frame = _rows_to_frame(
            [
                [1, -50, 70, 20, 90, 25.0, 0.4, 30, 1],
                [2, -60, 65, 15, 85, 24.0, 0.5, 40, 0],
            ],
            REQUIRED_COLUMNS,
        )
        issue = _issue_for(validate_dataframe(frame), IssueCode.NEGATIVE_VALUE)
        assert (issue.column, issue.count) == ("Glucose", 2)

    def test_negative_bmi_is_reported(self) -> None:
        rows = [list(row) for row in CLEAN_ROWS]
        rows[0][FEATURE_COLUMNS.index("BMI")] = -1.0
        report = validate_dataframe(_rows_to_frame(rows, REQUIRED_COLUMNS))
        assert _issue_for(report, IssueCode.NEGATIVE_VALUE).column == "BMI"

    def test_zero_age_is_an_error(self) -> None:
        rows = [list(row) for row in CLEAN_ROWS]
        rows[0][FEATURE_COLUMNS.index("Age")] = 0
        report = validate_dataframe(_rows_to_frame(rows, REQUIRED_COLUMNS))
        assert IssueCode.NON_POSITIVE_VALUE in _codes(report)
        assert not report.is_valid

    def test_negative_age_is_not_double_reported(self) -> None:
        """A negative Age is one finding, not two overlapping ones."""
        rows = [list(row) for row in CLEAN_ROWS]
        rows[0][FEATURE_COLUMNS.index("Age")] = -3
        report = validate_dataframe(_rows_to_frame(rows, REQUIRED_COLUMNS))
        assert IssueCode.NON_POSITIVE_VALUE not in _codes(report)
        assert _issue_for(report, IssueCode.NEGATIVE_VALUE).column == "Age"

    def test_zero_in_a_non_sentinel_column_is_not_an_error(self) -> None:
        rows = [list(row) for row in CLEAN_ROWS]
        rows[0][FEATURE_COLUMNS.index("DiabetesPedigreeFunction")] = 0.0
        assert validate_dataframe(_rows_to_frame(rows, REQUIRED_COLUMNS)).is_valid


class TestDuplicates:
    """Duplicate-row reporting."""

    def test_duplicate_rows_are_a_warning(self) -> None:
        rows = [list(row) for row in CLEAN_ROWS]
        rows.append(list(rows[0]))
        report = validate_dataframe(_rows_to_frame(rows, REQUIRED_COLUMNS))
        assert IssueCode.DUPLICATE_ROW in _codes(report, Severity.WARNING)
        assert report.is_valid, "duplicates are reported, not silently dropped"

    def test_duplicate_count_is_accurate(self) -> None:
        rows = [list(row) for row in CLEAN_ROWS]
        rows.append(list(rows[0]))
        rows.append(list(rows[1]))
        report = validate_dataframe(_rows_to_frame(rows, REQUIRED_COLUMNS))
        assert report.duplicate_count == 2
        assert _issue_for(report, IssueCode.DUPLICATE_ROW).count == 2

    def test_duplicate_rows_are_retained_in_the_frame(self) -> None:
        rows = [list(row) for row in CLEAN_ROWS]
        rows.append(list(rows[0]))
        frame = _rows_to_frame(rows, REQUIRED_COLUMNS)
        validate_dataframe(frame)
        assert len(frame.index) == len(CLEAN_ROWS) + 1


class TestReportApi:
    """The report surface itself."""

    def test_errors_and_warnings_partition_the_issues(self) -> None:
        rows = [list(row) for row in CLEAN_ROWS]
        rows.append(list(rows[0]))
        rows[0][FEATURE_COLUMNS.index("Glucose")] = 0
        report = validate_dataframe(_rows_to_frame(rows, REQUIRED_COLUMNS))
        assert len(report.errors) + len(report.warnings) == len(report.issues)

    def test_issue_counts_summarise_findings(self) -> None:
        rows = [list(row) for row in CLEAN_ROWS]
        rows[0][FEATURE_COLUMNS.index("Glucose")] = 0
        report = validate_dataframe(_rows_to_frame(rows, REQUIRED_COLUMNS))
        assert report.issue_counts()[IssueCode.ZERO_SENTINEL_VALUE.value] == 1

    def test_issue_counts_are_empty_for_a_clean_dataset(self, clean_frame: pd.DataFrame) -> None:
        assert validate_dataframe(clean_frame).issue_counts() == {}

    def test_column_lookup_returns_none_for_unknown(self, clean_frame: pd.DataFrame) -> None:
        assert validate_dataframe(clean_frame).column("NotAColumn") is None

    def test_issue_renders_readably(self) -> None:
        issue = ValidationIssue(
            code=IssueCode.DUPLICATE_ROW,
            severity=Severity.WARNING,
            message="Two rows repeat.",
            count=2,
        )
        rendered = str(issue)
        assert "WARNING" in rendered
        assert IssueCode.DUPLICATE_ROW.value in rendered
        assert "n=2" in rendered

    def test_issue_without_column_omits_the_location(self) -> None:
        issue = ValidationIssue(
            code=IssueCode.EMPTY_DATASET, severity=Severity.ERROR, message="Empty."
        )
        assert "[]" not in str(issue)

    def test_to_dict_is_json_serialisable(self) -> None:
        rows = [list(row) for row in CLEAN_ROWS]
        rows[0][FEATURE_COLUMNS.index("Glucose")] = 0
        rows.append(list(rows[1]))
        payload = validate_dataframe(_rows_to_frame(rows, REQUIRED_COLUMNS)).to_dict()
        assert json.loads(json.dumps(payload))["row_count"] == len(CLEAN_ROWS) + 1

    def test_to_dict_renders_integral_target_keys_without_decimal(
        self, clean_frame: pd.DataFrame
    ) -> None:
        payload = validate_dataframe(clean_frame).to_dict()
        assert set(payload["target_distribution"]) == {"0", "1"}

    def test_to_dict_renders_fractional_target_keys(self) -> None:
        frame = _rows_to_frame(
            [[1, 100, 70, 20, 90, 25.0, 0.4, 30, 1.5]],
            REQUIRED_COLUMNS,
        )
        payload = validate_dataframe(frame).to_dict()
        assert "1.5" in payload["target_distribution"]

    def test_to_dict_records_validity_and_issues(self, clean_frame: pd.DataFrame) -> None:
        payload = validate_dataframe(clean_frame).to_dict()
        assert payload["is_valid"] is True
        assert payload["issues"] == []

    def test_to_dict_records_column_statistics(self, clean_frame: pd.DataFrame) -> None:
        payload = validate_dataframe(clean_frame).to_dict()
        assert payload["columns"][0]["name"] == REQUIRED_COLUMNS[0]
        assert payload["columns"][0]["is_numeric"] is True

    def test_render_includes_the_measured_shape(self, clean_frame: pd.DataFrame) -> None:
        rendered = validate_dataframe(clean_frame).render()
        assert f"rows:        {len(CLEAN_ROWS)}" in rendered
        assert TARGET_COLUMN in rendered

    def test_render_includes_findings(self) -> None:
        rows = [list(row) for row in CLEAN_ROWS]
        rows[0][FEATURE_COLUMNS.index("Insulin")] = 0
        rendered = validate_dataframe(_rows_to_frame(rows, REQUIRED_COLUMNS)).render()
        assert IssueCode.ZERO_SENTINEL_VALUE.value in rendered

    def test_render_falls_back_for_an_in_memory_frame(self, clean_frame: pd.DataFrame) -> None:
        assert "in-memory DataFrame" in validate_dataframe(clean_frame).render()

    def test_render_omits_the_target_line_when_there_is_no_target(
        self, clean_frame: pd.DataFrame
    ) -> None:
        report = validate_dataframe(clean_frame.drop(columns=[TARGET_COLUMN]))
        assert report.target_distribution == {}
        assert "target:" not in report.render()

    def test_render_omits_columns_for_an_empty_dataset(self) -> None:
        rendered = validate_dataframe(pd.DataFrame()).render()
        assert "rows:        0" in rendered
        assert "- Glucose:" not in rendered, "no per-column lines for an empty dataset"
        assert "zero-valued sentinel columns" not in rendered

    def test_report_has_no_defaults_beyond_the_documented_ones(self) -> None:
        """A report must be built by validation, not assembled by hand."""
        report = DataQualityReport(source=None, row_count=0, column_count=0, column_names=())
        assert report.is_valid and not report.has_warnings
        assert report.issue_counts() == {}
        assert report.to_dict()["target_distribution"] == {}


class TestRaiseForErrors:
    """Opt-in failure."""

    def test_valid_report_is_returned_unchanged(self, clean_frame: pd.DataFrame) -> None:
        report = validate_dataframe(clean_frame)
        assert raise_for_errors(report) is report

    def test_error_report_raises(self, clean_frame: pd.DataFrame) -> None:
        report = validate_dataframe(clean_frame.drop(columns=["Age"]))
        with pytest.raises(DatasetValidationError):
            raise_for_errors(report)

    def test_exception_carries_the_full_report(self, clean_frame: pd.DataFrame) -> None:
        """One exception, but every finding still reachable."""
        report = validate_dataframe(clean_frame.drop(columns=["Age", "Insulin"]))
        with pytest.raises(DatasetValidationError) as excinfo:
            raise_for_errors(report)
        assert len(excinfo.value.report.errors) == 2

    def test_warnings_alone_do_not_raise(self) -> None:
        rows = [list(row) for row in CLEAN_ROWS]
        rows[0][FEATURE_COLUMNS.index("Glucose")] = 0
        report = validate_dataframe(_rows_to_frame(rows, REQUIRED_COLUMNS))
        assert raise_for_errors(report) is report


class TestSchemaHelpers:
    """Convenience accessors over the schema contract."""

    def test_documented_predictors_match_the_schema(self) -> None:
        assert tuple(documented_predictors()) == FEATURE_COLUMNS

    def test_documented_target(self) -> None:
        assert documented_target() == TARGET_COLUMN

    def test_missing_required_columns_lists_absent_only(self, clean_frame: pd.DataFrame) -> None:
        assert missing_required_columns(clean_frame.drop(columns=["Age", "BMI"])) == ["BMI", "Age"]

    def test_missing_required_columns_is_empty_when_complete(
        self, clean_frame: pd.DataFrame
    ) -> None:
        assert missing_required_columns(clean_frame) == []

    def test_expected_filename(self) -> None:
        assert DATASET_FILENAME == "diabetes.csv"

    def test_expected_header_order(self) -> None:
        assert HEADER.split(",") == list(REQUIRED_COLUMNS)
