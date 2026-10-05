# `data/raw/` — raw dataset staging area

This directory is where the dataset CSV belongs. **Nothing is committed here,
and nothing is downloaded automatically.** The dataset is an externally supplied
artifact that you place yourself.

---

## 1. Where the dataset comes from

The expected dataset is the **Pima Indians Diabetes** dataset:

| | |
| --- | --- |
| Publisher | National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) |
| Distributed by | UCI Machine Learning Repository |
| UCI dataset id | 329 |
| Reference | Smith, J. W. *An Expert System for Self-Integrated Therapy Monitoring and Management for Insulin-Dependent Diabetes Mellitus and Non-Insulin-Dependent Diabetes Mellitus* |
| Age | Collected in the 1980s |

Obtain it yourself from the UCI Machine Learning Repository, read the terms
attached to it, and record where you got it. This project deliberately ships no
copy, performs no download, and cannot verify provenance for you.

### Important limitations of this dataset

These are properties of the **dataset**, not medical advice, and they constrain
how far any result built on it can be pushed:

- **Not representative.** The cohort is a specific group of patients at a
  specific time and place. It is *not* a random or representative sample of any
  wider population, and performance measured on it must not be generalised to the
  general population.
- **Not a description of current clinical practice.** Data collection
  predates current diagnostic criteria, guidelines, laboratory practice and
  treatment pathways by decades.
- **A benchmark, not a clinical cohort.** It exists to support comparative
  method evaluation. It was not assembled to answer a clinical question.
- **Small and narrow.** Restricted to adults of one specific ancestry. Anything
  built on it carries that restriction.
- **Ageing and incomplete.** Recorded decades ago, so it does not reflect
  contemporary populations, diagnostics or therapies.

This project exists to demonstrate engineering and explainability technique.
It does not assess this dataset's fitness for any clinical purpose, and it makes
no diagnostic claim.

---

## 2. Expected filename and location

```
data/raw/diabetes.csv
```

`cronical.data.loader.default_dataset_path()` resolves this location from the
configured project root, so no path is hard-coded. Any other filename works, but
must be passed explicitly:

```python
from cronical.data.loader import load_and_validate

frame, report = load_and_validate("data/raw/some_other_name.csv")
```

---

## 3. Expected schema

The header must be exactly these nine columns, in this order:

```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
```

| # | Column | Expected type | Unit (per source docs) | Zero is a "not recorded" placeholder? |
| --- | --- | --- | --- | --- |
| 1 | `Pregnancies` | integer | count | No — zero is a real observation |
| 2 | `Glucose` | integer | mg/dL | **Yes** |
| 3 | `BloodPressure` | integer | mm Hg | **Yes** |
| 4 | `SkinThickness` | integer | mm | **Yes** |
| 5 | `Insulin` | integer | µU/mL | **Yes** |
| 6 | `BMI` | decimal | kg/m² | **Yes** |
| 7 | `DiabetesPedigreeFunction` | decimal | — | No |
| 8 | `Age` | integer | years | No — and zero is invalid regardless |
| 9 | `Outcome` | integer | — | No — binary label, must be `0` or `1` |

Notes on the table above:

- Units are transcribed from the published dataset description. **The CSV file
  itself contains no units**, so they are documentation rather than something the
  code can infer or verify. Confirm them before interpreting any output.
- `Outcome` is the binary class label recorded in the dataset. The dataset
  documentation does not define the test or criteria behind it, so this project
  does not restate or reinterpret what the label means.
- The authoritative version of this contract is
  `src/cronical/data/schema.py`. If the two ever disagree, that file wins.

---

## 4. Raw data must not be modified

This directory is **immutable input**.

- Do not edit, reformat, re-order columns, or "fix" values in `diabetes.csv`.
- Do not impute, drop or recode anything here.
- `.gitignore` excludes dataset contents so nothing is committed by accident.
- Every transformation belongs in a later, recorded preprocessing step that
  reads from here and writes to `data/processed/`.

The reason is reproducibility: if the raw file can change quietly, no result
downstream can be trusted or reproduced.

---

## 5. Zero values need special handling

This dataset uses **`0` to mean "this measurement was not recorded"** in five
columns:

```
Glucose · BloodPressure · SkinThickness · Insulin · BMI
```

A recorded value of exactly zero in any of these is not a physiological
measurement — nobody's blood pressure is zero. Treating these cells as real zeros
would silently corrupt anything computed from them.

### What this project does, and does not do, with them

| Stage | Behaviour |
| --- | --- |
| **Raw file** | Untouched. The placeholder stays as `0`. |
| **Loader** | Untouched. Values are read exactly as written. |
| **Validation** | **Reported.** Zero counts and a per-column warning are emitted. The data is not changed. |
| **Preprocessing** | **Not implemented yet.** The decision — impute, drop, or model the missingness — must be made explicitly and recorded when that stage is built. |

### Severity, and why

Zero-sentinel values are reported as **warnings**, not errors. A dataset
containing them is still loadable and still analysable; treating this known,
documented property of the dataset as a hard failure would make the loader
useless on the very dataset it is written for.

They are escalated to an **error** in one case only: if an entire sentinel column
is zero, that column carries no usable measurement at all.

### Zero is not always a placeholder

`Pregnancies` of `0` is a genuine observation and is treated as ordinary data.
`Age` of `0` is reported as invalid regardless.

---

## 6. Placing the file: checklist

1. Download the CSV from the UCI Machine Learning Repository.
2. Save it as `data/raw/diabetes.csv` with the header shown in section 3.
3. Confirm nothing else was changed: same column names, same order, same rows.
4. Validate it:

```python
from cronical.data.loader import load_and_validate

frame, report = load_and_validate()
print(report.render())
```

A healthy result has `is_valid == True`. Warnings about zero-sentinel values are
**expected** for this dataset and do not indicate a problem with your copy.

If the dataset is absent, the loader raises `DatasetNotFoundError` with the
expected path — it never fabricates or substitutes data.
