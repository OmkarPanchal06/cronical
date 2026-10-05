"""Loading the dataset CSV from a configurable local path.

This module deals only with *getting bytes onto a table*. It does not inspect
clinical meaning and does not clean anything; content checks live in
:mod:`cronical.data.validation`.

No automatic download
---------------------
There is no network access anywhere in this package, by design. The dataset is
an externally supplied artifact that a developer places in ``data/raw/`` as
described in ``data/raw/README.md``. Reproducibility, licensing and provenance
are the operator's responsibility, not something a build step should quietly
handle.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from cronical.config import get_settings
from cronical.data.errors import (
    DatasetFormatError,
    DatasetNotFoundError,
    DatasetPathError,
    DatasetReadError,
)
from cronical.data.schema import DATASET_FILENAME
from cronical.data.validation import validate_dataframe
from cronical.utils.logging import get_logger

if TYPE_CHECKING:
    from cronical.data.report import DataQualityReport

__all__ = [
    "default_dataset_path",
    "load_and_validate",
    "load_dataset",
    "resolve_dataset_path",
]

_LOG = get_logger(__name__)

#: The only delimited format accepted at this stage.
SUPPORTED_SUFFIX = ".csv"


def default_dataset_path() -> Path:
    """Return the default dataset location, derived from the project root.

    Returns:
        ``<project_root>/data/raw/diabetes.csv``. Resolved from settings rather
        than the working directory, so the same file is found regardless of where
        a command is invoked from.
    """
    return get_settings().paths.raw_data_dir / DATASET_FILENAME


def resolve_dataset_path(path: Path | str | None = None) -> Path:
    """Turn a caller-supplied location into an absolute path.

    Args:
        path: A path to the dataset. ``None`` selects
            :func:`default_dataset_path`. A relative path is resolved against the
            current working directory, which is the conventional behaviour for a
            command-line argument.

    Returns:
        An expanded, absolute path. The file is not touched here.
    """
    if path is None:
        return default_dataset_path()
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def _assert_readable(path: Path) -> None:
    """Check the file exists and is a regular file.

    Raises:
        DatasetNotFoundError: If nothing exists at ``path``.
        DatasetPathError: If something exists there but is not a regular file.
    """
    if not path.exists():
        raise DatasetNotFoundError(path)
    if not path.is_file():
        raise DatasetPathError(f"Expected a file at {path} but found a directory.")


def _assert_supported_suffix(path: Path) -> None:
    """Check the extension is one this loader understands.

    Raises:
        DatasetFormatError: If the extension is not ``.csv``.
    """
    if path.suffix.lower() != SUPPORTED_SUFFIX:
        raise DatasetFormatError(path, expected_suffix=SUPPORTED_SUFFIX)


def _assert_rectangular(resolved: Path, frame: pd.DataFrame) -> None:
    """Reject a table pandas could only build by inventing an index.

    When a data row holds more values than the header declares, pandas silently
    folds the surplus into a multi-level index and shifts every remaining column.
    The resulting frame looks plausible but describes different data, so this is
    treated as a read failure rather than returned.

    Rows holding *fewer* values than the header are left alone: pandas pads them
    with nulls, which validation reports explicitly as missing values.

    Args:
        resolved: The file that was read, for the error message.
        frame: The frame pandas produced.

    Raises:
        DatasetReadError: If the frame does not have a default row index.
    """
    index = frame.index
    if isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1:
        return
    raise DatasetReadError(
        resolved,
        reason=(
            "some rows contain more values than the header declares, so the file "
            "is not a rectangular table; every row must have exactly one value "
            "per column"
        ),
    )


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Read the dataset CSV into a :class:`pandas.DataFrame`.

    The frame is returned exactly as parsed: no column is renamed, coerced or
    dropped, so validation and preprocessing see the original file.

    Args:
        path: Location of the CSV. Defaults to
            :func:`default_dataset_path`.

    Returns:
        The parsed dataset.

    Raises:
        DatasetNotFoundError: If no file exists at the resolved path.
        DatasetPathError: If the path points at something other than a file.
        DatasetFormatError: If the extension is not ``.csv``.
        DatasetReadError: If the file exists but cannot be parsed, including the
            case of a zero-byte file.
    """
    resolved = resolve_dataset_path(path)
    _assert_readable(resolved)
    _assert_supported_suffix(resolved)

    try:
        frame = pd.read_csv(resolved)
    except pd.errors.EmptyDataError as exc:
        raise DatasetReadError(resolved, reason="the file is empty") from exc
    except pd.errors.ParserError as exc:
        raise DatasetReadError(resolved, reason=f"malformed CSV: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DatasetReadError(resolved, reason=f"the file is not UTF-8 text: {exc}") from exc
    except OSError as exc:
        raise DatasetReadError(resolved, reason=f"the file could not be opened: {exc}") from exc

    _assert_rectangular(resolved, frame)

    _LOG.info(
        "dataset loaded",
        extra={"path": str(resolved), "rows": len(frame.index), "columns": len(frame.columns)},
    )
    return frame


def load_and_validate(path: Path | str | None = None) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load the dataset and validate it in one call.

    Args:
        path: Location of the CSV. Defaults to :func:`default_dataset_path`.

    Returns:
        The parsed frame together with its
        :class:`~cronical.data.report.DataQualityReport`. Findings are returned
        rather than raised; use
        :func:`cronical.data.validation.raise_for_errors` to fail fast.

    Raises:
        DatasetNotFoundError: If no file exists at the resolved path.
        DatasetPathError: If the path points at something other than a file.
        DatasetFormatError: If the extension is not ``.csv``.
        DatasetReadError: If the file exists but cannot be parsed.
    """
    resolved = resolve_dataset_path(path)
    frame = load_dataset(resolved)
    return frame, validate_dataframe(frame, source=str(resolved))
