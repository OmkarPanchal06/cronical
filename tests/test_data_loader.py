"""Tests for :mod:`cronical.data.loader`.

Covers the contract the rest of the project relies on: a caller can point the
loader at a local CSV and get a :class:`pandas.DataFrame`, or a specific typed
exception explaining precisely what was wrong. Nothing here touches the network
or the real ``data/raw/`` directory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cronical.config import ENV_PREFIX, reload_settings
from cronical.data.errors import (
    CronicalDataError,
    DatasetFormatError,
    DatasetNotFoundError,
    DatasetPathError,
    DatasetReadError,
)
from cronical.data.loader import (
    default_dataset_path,
    load_and_validate,
    load_dataset,
    resolve_dataset_path,
)
from cronical.data.schema import DATASET_FILENAME, REQUIRED_COLUMNS

from .conftest import CLEAN_ROWS, HEADER, CsvWriter


def _csv_body(rows: tuple[tuple[int | float, ...], ...]) -> str:
    """Render rows as CSV text below the standard header."""
    return "\n".join(",".join(str(value) for value in row) for row in rows)


class TestLoadDatasetHappyPath:
    """A well-formed file loads as a DataFrame."""

    def test_returns_dataframe(self, clean_csv: Path) -> None:
        frame = load_dataset(clean_csv)
        assert isinstance(frame, pd.DataFrame)

    def test_shape_matches_the_file(self, clean_csv: Path) -> None:
        frame = load_dataset(clean_csv)
        assert frame.shape == (len(CLEAN_ROWS), len(REQUIRED_COLUMNS))

    def test_columns_match_the_schema_exactly(self, clean_csv: Path) -> None:
        frame = load_dataset(clean_csv)
        assert tuple(frame.columns) == REQUIRED_COLUMNS

    def test_values_are_preserved(self, clean_csv: Path) -> None:
        frame = load_dataset(clean_csv)
        assert int(frame.loc[0, "Glucose"]) == 148
        assert float(frame.loc[0, "BMI"]) == 33.6
        assert int(frame.loc[0, "Outcome"]) == 1

    def test_accepts_a_string_path(self, clean_csv: Path) -> None:
        assert len(load_dataset(str(clean_csv))) == len(CLEAN_ROWS)

    def test_extension_check_is_case_insensitive(self, write_csv: CsvWriter) -> None:
        path = write_csv(f"{HEADER}\n{_csv_body(CLEAN_ROWS)}\n", name="diabetes.CSV")
        assert len(load_dataset(path)) == len(CLEAN_ROWS)

    def test_does_not_modify_the_file(self, clean_csv: Path) -> None:
        before = clean_csv.read_bytes()
        load_dataset(clean_csv)
        assert clean_csv.read_bytes() == before, "the loader must not write to raw data"

    def test_does_not_coerce_dtypes(self, clean_csv: Path) -> None:
        """Values are read as-is; interpretation belongs to validation."""
        frame = load_dataset(clean_csv)
        assert pd.api.types.is_numeric_dtype(frame["BMI"])


class TestLoadDatasetErrors:
    """Every failure mode raises a specific, catchable exception."""

    def test_missing_file_raises_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError) as excinfo:
            load_dataset(tmp_path / "absent.csv")
        assert excinfo.value.path == tmp_path / "absent.csv"

    def test_not_found_message_explains_placement(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="data/raw"):
            load_dataset(tmp_path / "absent.csv")

    def test_not_found_names_the_expected_filename(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match=DATASET_FILENAME):
            load_dataset(tmp_path / "absent.csv")

    @pytest.mark.parametrize(
        "name", ["diabetes.txt", "diabetes.parquet", "diabetes", "diabetes.csv.gz"]
    )
    def test_wrong_extension_raises_format_error(self, write_csv: CsvWriter, name: str) -> None:
        path = write_csv("anything", name=name)
        with pytest.raises(DatasetFormatError) as excinfo:
            load_dataset(path)
        assert excinfo.value.expected_suffix == ".csv"

    def test_format_error_reports_the_actual_suffix(self, write_csv: CsvWriter) -> None:
        path = write_csv("anything", name="diabetes.parquet")
        with pytest.raises(DatasetFormatError, match=r"parquet"):
            load_dataset(path)

    def test_directory_raises_path_error(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetPathError, match="directory"):
            load_dataset(tmp_path)

    def test_all_errors_share_a_base_class(self, tmp_path: Path) -> None:
        with pytest.raises(CronicalDataError):
            load_dataset(tmp_path / "absent.csv")

    def test_empty_file_raises_read_error(self, write_csv: CsvWriter) -> None:
        path = write_csv("")
        with pytest.raises(DatasetReadError, match="empty"):
            load_dataset(path)

    def test_header_only_file_loads_with_zero_rows(self, write_csv: CsvWriter) -> None:
        """A file with a header but no rows is readable; emptiness is a content concern."""
        path = write_csv(f"{HEADER}\n")
        frame = load_dataset(path)
        assert len(frame.index) == 0
        assert tuple(frame.columns) == REQUIRED_COLUMNS

    def test_unclosed_quote_raises_read_error(self, write_csv: CsvWriter) -> None:
        path = write_csv(f'{HEADER}\n1,"2\n')
        with pytest.raises(DatasetReadError, match="malformed CSV"):
            load_dataset(path)

    def test_rows_longer_than_the_header_raise_read_error(self, write_csv: CsvWriter) -> None:
        """A row longer than the header would otherwise be absorbed into an index."""
        path = write_csv("Pregnancies,Glucose\n1,2,3,4\n5,6,7,8\n")
        with pytest.raises(DatasetReadError, match="not a rectangular table"):
            load_dataset(path)

    def test_rows_shorter_than_the_header_are_padded_and_reported_later(
        self, write_csv: CsvWriter
    ) -> None:
        """A short row becomes an explicit null, which validation surfaces."""
        path = write_csv("Pregnancies,Glucose,BloodPressure\n1,2\n5,6,7\n")
        frame = load_dataset(path)
        assert frame.shape == (2, 3)
        assert pd.isna(frame.loc[0, "BloodPressure"])

    def test_non_utf8_file_raises_read_error(self, tmp_path: Path) -> None:
        path = tmp_path / "diabetes.csv"
        path.write_bytes(b"Pregnancies,Glucose\n1,\xff\xfe\xfa\n")
        with pytest.raises(DatasetReadError):
            load_dataset(path)

    def test_unreadable_file_raises_read_error(
        self, clean_csv: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*args: object, **kwargs: object) -> None:
            raise OSError("permission denied")

        monkeypatch.setattr(pd, "read_csv", refuse)
        with pytest.raises(DatasetReadError, match="could not be opened"):
            load_dataset(clean_csv)


class TestPathResolution:
    """Paths are derived from settings, never hard-coded."""

    def test_default_path_points_into_data_raw(self) -> None:
        resolved = default_dataset_path()
        assert resolved.parent.name == "raw"
        assert resolved.parent.parent.name == "data"
        assert resolved.name == DATASET_FILENAME

    def test_default_path_follows_settings(
        self, isolated_project_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(f"{ENV_PREFIX}PROJECT_ROOT", str(isolated_project_root))
        reload_settings()
        assert default_dataset_path() == (isolated_project_root / "data" / "raw" / DATASET_FILENAME)

    def test_default_path_is_absolute(self) -> None:
        assert default_dataset_path().is_absolute()

    def test_resolve_none_uses_the_default(self) -> None:
        assert resolve_dataset_path(None) == default_dataset_path()

    def test_resolve_relative_uses_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert resolve_dataset_path("sub/diabetes.csv") == tmp_path / "sub" / "diabetes.csv"

    def test_resolve_keeps_absolute_paths(self, tmp_path: Path) -> None:
        assert resolve_dataset_path(tmp_path / "x.csv") == tmp_path / "x.csv"

    def test_resolve_accepts_a_string(self, tmp_path: Path) -> None:
        assert resolve_dataset_path(str(tmp_path / "x.csv")) == tmp_path / "x.csv"

    def test_resolve_expands_the_home_directory(self, tmp_path: Path) -> None:
        assert resolve_dataset_path("~/diabetes.csv").is_absolute()

    def test_default_dataset_is_absent_before_it_is_supplied(
        self, isolated_project_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fresh checkout has no dataset, and says so rather than inventing one."""
        monkeypatch.setenv(f"{ENV_PREFIX}PROJECT_ROOT", str(isolated_project_root))
        reload_settings()
        assert not default_dataset_path().exists()
        with pytest.raises(DatasetNotFoundError):
            load_dataset()


class TestLoadAndValidate:
    """The convenience entry point."""

    def test_returns_frame_and_report(self, clean_csv: Path) -> None:
        frame, report = load_and_validate(clean_csv)
        assert len(frame.index) == len(CLEAN_ROWS)
        assert report.row_count == len(CLEAN_ROWS)

    def test_report_records_the_source(self, clean_csv: Path) -> None:
        _, report = load_and_validate(clean_csv)
        assert report.source == str(clean_csv)

    def test_clean_file_passes_validation(self, clean_csv: Path) -> None:
        _, report = load_and_validate(clean_csv)
        assert report.is_valid
        assert not report.warnings

    def test_propagates_loader_errors(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError):
            load_and_validate(tmp_path / "absent.csv")

    def test_propagates_format_errors(self, write_csv: CsvWriter) -> None:
        with pytest.raises(DatasetFormatError):
            load_and_validate(write_csv("x", name="diabetes.tsv"))
