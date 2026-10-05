"""The explicit data contract for the Pima Indians Diabetes dataset.

This module is the single source of truth for the expected schema. The loader,
the validators, the tests and the documentation all read these definitions, so a
column rename is a one-line change rather than a search across the repository.

Provenance
----------
The expected schema matches the "Pima Indians Diabetes" dataset published by the
National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) and
distributed through the UCI Machine Learning Repository. This project treats the
CSV as an externally supplied artifact: nothing here downloads it, and nothing
here verifies its provenance. See ``data/raw/README.md``.

Scope and limits
----------------
* The cohort is a specific group of adult patients and is **not** a random or
  representative sample of any wider population. It must not be read as
  describing the general population or current clinical practice.
* ``units`` below are transcribed from the published dataset description. The CSV
  file itself carries no units, so they are documentation, not something the code
  can infer or enforce.
* This module describes *data*, not medicine. It contains no diagnostic
  thresholds, no reference ranges and no clinical decision rules.

Clinically invalid zero values
------------------------------
Several columns in this dataset use ``0`` to mean *no measurement recorded*
rather than a genuine zero. A recorded glucose concentration, blood pressure,
skin thickness, insulin level or BMI of exactly zero is not a physiological
measurement. Those columns are flagged by
:attr:`ColumnSpec.zero_is_missing_sentinel` so that validation can report them,
and so that a later preprocessing stage can decide explicitly how to treat them.

Zero values in columns *not* flagged here are ordinary data: ``Pregnancies`` of
zero is a valid observation, and this module treats it as such.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

__all__ = [
    "COLUMN_SPECS",
    "COLUMN_SPECS_BY_NAME",
    "DATASET_FILENAME",
    "FEATURE_COLUMNS",
    "FEATURE_SPECS",
    "REQUIRED_COLUMNS",
    "TARGET_COLUMN",
    "TARGET_COLUMN_ALIASES",
    "TARGET_SPEC",
    "VALID_TARGET_VALUES",
    "ZERO_SENTINEL_COLUMNS",
    "ColumnSpec",
]

#: Filename expected inside ``data/raw/``.
DATASET_FILENAME: Final[str] = "diabetes.csv"

#: Name of the binary outcome column.
TARGET_COLUMN: Final[str] = "Outcome"

#: The only values the outcome column is permitted to take.
VALID_TARGET_VALUES: Final[frozenset[int]] = frozenset({0, 1})

#: Lower-cased spellings that indicate a differently named outcome column. Used
#: only to produce a more helpful error message; never to silently accept them.
TARGET_COLUMN_ALIASES: Final[frozenset[str]] = frozenset(
    {
        "outcome",
        "class",
        "classvariable",
        "class_variable",
        "target",
        "diabetes",
        "diabetesoutcome",
        "diabetes_outcome",
        "label",
        "y",
    }
)


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """The contract for one column of the dataset.

    Attributes:
        name: Exact column name as it must appear in the CSV header.
        description: What the column records, per the published dataset
            documentation. Descriptive only; not a clinical definition.
        dtype: Expected pandas dtype family. ``"int"`` columns must load without
            a fractional part; ``"float"`` columns accept both.
        unit: Unit as published with the dataset, or ``None`` when the published
            description does not state one. Never inferred from the data.
        zero_is_missing_sentinel: ``True`` when ``0`` is used by the dataset to
            mean "not recorded" and therefore cannot be a real observation for
            this column.
        must_be_positive: ``True`` when a value of zero or below is
            impossible for the column even as a measurement placeholder.
    """

    name: str
    description: str
    dtype: Literal["int", "float"]
    unit: str | None = None
    zero_is_missing_sentinel: bool = False
    must_be_positive: bool = False


#: Predictor columns, in published order.
FEATURE_SPECS: Final[tuple[ColumnSpec, ...]] = (
    ColumnSpec(
        name="Pregnancies",
        description="Number of times the patient has been pregnant.",
        dtype="int",
        unit="count",
    ),
    ColumnSpec(
        name="Glucose",
        description="Plasma glucose concentration.",
        dtype="int",
        unit="mg/dL",
        zero_is_missing_sentinel=True,
    ),
    ColumnSpec(
        name="BloodPressure",
        description="Diastolic blood pressure.",
        dtype="int",
        unit="mm Hg",
        zero_is_missing_sentinel=True,
    ),
    ColumnSpec(
        name="SkinThickness",
        description="Triceps skin thickness.",
        dtype="int",
        unit="mm",
        zero_is_missing_sentinel=True,
    ),
    ColumnSpec(
        name="Insulin",
        description="Two-hour serum insulin level.",
        dtype="int",
        unit="microU/mL",
        zero_is_missing_sentinel=True,
    ),
    ColumnSpec(
        name="BMI",
        description="Body mass index, defined as weight divided by height squared.",
        dtype="float",
        unit="kg/m^2",
        zero_is_missing_sentinel=True,
    ),
    ColumnSpec(
        name="DiabetesPedigreeFunction",
        description="Score summarising diabetes history in close relatives.",
        dtype="float",
        unit=None,
    ),
    ColumnSpec(
        name="Age",
        description="Age of the patient in years.",
        dtype="int",
        unit="years",
        must_be_positive=True,
    ),
)

#: The outcome column, described separately because it is not a predictor.
TARGET_SPEC: Final[ColumnSpec] = ColumnSpec(
    name=TARGET_COLUMN,
    description="Binary class label recorded in the dataset.",
    dtype="int",
    unit=None,
)

#: Every column the loader expects to find, in published order.
COLUMN_SPECS: Final[tuple[ColumnSpec, ...]] = (*FEATURE_SPECS, TARGET_SPEC)

#: Lookup of every expected column by name.
COLUMN_SPECS_BY_NAME: Final[Mapping[str, ColumnSpec]] = MappingProxyType(
    {spec.name: spec for spec in COLUMN_SPECS}
)

#: Predictor column names, in published order.
FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(spec.name for spec in FEATURE_SPECS)

#: Every required column name, in published order.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = tuple(spec.name for spec in COLUMN_SPECS)

#: Columns whose ``0`` values represent an unrecorded measurement.
ZERO_SENTINEL_COLUMNS: Final[tuple[str, ...]] = tuple(
    spec.name for spec in COLUMN_SPECS if spec.zero_is_missing_sentinel
)
