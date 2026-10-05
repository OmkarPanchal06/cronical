"""Shared pytest fixtures for the Cronical test suite.

Four guarantees are enforced here for every test in the repository:

* No test writes to the real project tree. Anything that touches the filesystem
  is redirected to ``tmp_path``.
* The global :func:`get_settings` cache and the ``cronical`` logging handlers are
  reset before and after each test, so ordering never changes the result.
* No test loads the real dataset. Tests build tiny CSV files and DataFrames from
  values written out in full below, so every expected count is derived from data
  the test itself controls.
* The synthetic values in this module are **not** observations from the Pima
  Indians Diabetes dataset. They are invented placeholders chosen to exercise
  specific validation rules. No statistic in this repository is derived from
  them, and nothing here may be cited as a property of the real dataset.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from cronical.config import Settings, reload_settings
from cronical.data.schema import REQUIRED_COLUMNS
from cronical.utils.logging import reset_logging

if TYPE_CHECKING:
    import pandas as pd

#: The exact header the loader expects, written out rather than derived, so a
#: schema change breaks these tests loudly instead of silently following along.
HEADER = (
    "Pregnancies,Glucose,BloodPressure,SkinThickness,"
    "Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome"
)

#: Synthetic rows chosen so that a dataset built from them produces **no**
#: errors and **no** warnings. Every sentinel column holds a non-zero value,
#: ``Age`` is positive, nothing is null, and no two rows are identical.
CLEAN_ROWS: tuple[tuple[int | float, ...], ...] = (
    (6, 148, 72, 35, 100, 33.6, 0.627, 50, 1),
    (1, 85, 66, 23, 94, 26.6, 0.351, 32, 0),
    (8, 183, 64, 24, 110, 23.3, 0.672, 41, 1),
    (1, 90, 62, 20, 84, 25.2, 0.482, 28, 0),
)

#: Synthetic rows for preprocessing tests. As with :data:`CLEAN_ROWS`, these are
#: invented placeholders, **not** observations from the Pima Indians Diabetes
#: dataset, and no property of that dataset is inferred from them.
#:
#: Chosen to exercise every preprocessing path at once:
#:
#: * rows 0-11 are ordinary, with no sentinel zeros and no gaps
#: * row 12 is zero across every sentinel column
#: * row 13 has a zero ``Glucose`` only
#: * rows 14-15 carry genuine missing values, plus a zero ``Insulin``
#: * rows 16-17 hold extreme values and are kept for the leakage tests, so that a
#:   median computed over all rows differs measurably from the training median
PREPROCESSING_ROWS: tuple[tuple[float, ...], ...] = (
    (6, 148, 72, 35, 100, 33.6, 0.627, 50, 1),
    (1, 85, 66, 23, 94, 26.6, 0.351, 32, 0),
    (8, 183, 64, 24, 110, 23.3, 0.672, 41, 1),
    (1, 90, 62, 20, 84, 25.2, 0.482, 28, 0),
    (4, 110, 66, 25, 120, 24.0, 0.400, 33, 1),
    (2, 95, 60, 22, 100, 23.5, 0.510, 35, 0),
    (3, 130, 70, 28, 140, 27.0, 0.610, 45, 1),
    (5, 105, 68, 26, 130, 26.5, 0.560, 39, 0),
    (7, 120, 64, 27, 125, 25.0, 0.520, 36, 1),
    (0, 100, 66, 24, 110, 25.8, 0.430, 31, 0),
    (2, 140, 70, 30, 150, 29.0, 0.700, 48, 1),
    (9, 88, 62, 21, 96, 24.5, 0.380, 30, 0),
    (0, 0, 0, 0, 0, 0.0, 0.250, 22, 0),
    (11, 0, 82, 30, 160, 31.5, 0.620, 55, 1),
    (5, float("nan"), 68, 24, 0, 27.5, float("nan"), 44, 1),
    (3, 95, float("nan"), 22, 100, 24.0, 0.470, 34, 0),
    (0, 200, 90, 40, 300, 45.0, 0.900, 60, 1),
    (0, 210, 92, 42, 320, 46.0, 0.950, 62, 0),
)

CsvWriter = Callable[..., Path]


@pytest.fixture(autouse=True)
def _isolate_global_state() -> Iterator[None]:
    """Reset the settings cache and logging handlers around every test."""
    reload_settings()
    reset_logging()
    yield
    reload_settings()
    reset_logging()


@pytest.fixture
def isolated_project_root(tmp_path: Path) -> Path:
    """Return a throwaway project root that mirrors the real directory layout."""
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def settings(isolated_project_root: Path) -> Settings:
    """Return :class:`Settings` rooted at :func:`isolated_project_root`."""
    return Settings.for_root(isolated_project_root)


@pytest.fixture
def write_csv(tmp_path: Path) -> CsvWriter:
    """Return a helper that writes text into ``tmp_path`` and returns the path.

    Defaults to the filename the loader expects, so a test only overrides the
    name when the filename itself is what it is testing.
    """

    def _write(
        content: str,
        name: str = "diabetes.csv",
        *,
        encoding: str = "utf-8",
    ) -> Path:
        path = tmp_path / name
        path.write_text(content, encoding=encoding)
        return path

    return _write


@pytest.fixture
def clean_csv(write_csv: CsvWriter) -> Path:
    """Return a CSV that passes every validation check with no findings."""
    body = "\n".join(",".join(str(value) for value in row) for row in CLEAN_ROWS)
    return write_csv(f"{HEADER}\n{body}\n")


@pytest.fixture
def clean_frame() -> pd.DataFrame:
    """Return a DataFrame equivalent to :func:`clean_csv`."""
    import pandas as pd

    return pd.DataFrame(CLEAN_ROWS, columns=list(REQUIRED_COLUMNS))


@pytest.fixture
def dataset_frame() -> pd.DataFrame:
    """Return a full dataset with the schema's columns and float feature dtypes.

    Built from :data:`PREPROCESSING_ROWS`, so it passes validation with
    zero-sentinel and missing-value warnings but no errors.
    """
    import pandas as pd

    from cronical.data.schema import FEATURE_COLUMNS, TARGET_COLUMN

    frame = pd.DataFrame(
        [list(row) for row in PREPROCESSING_ROWS],
        columns=[*FEATURE_COLUMNS, TARGET_COLUMN],
    )
    return frame.astype(dict.fromkeys(FEATURE_COLUMNS, "float64"))


@pytest.fixture
def features_frame(dataset_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the feature columns of :func:`dataset_frame`, without the target."""
    from cronical.data.schema import FEATURE_COLUMNS

    return cast("pd.DataFrame", dataset_frame.loc[:, list(FEATURE_COLUMNS)])


@pytest.fixture
def patient_record() -> dict[str, float]:
    """Return one synthetic patient record using the schema's feature names."""
    return {
        "Pregnancies": 2,
        "Glucose": 0,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 120,
        "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 38,
    }
