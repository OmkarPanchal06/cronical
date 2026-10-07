# Cronical

> **AI-assisted decision support. Not a diagnostic tool.**
> Cronical is an educational research prototype. It does not diagnose diabetes,
> it does not treat disease, and it must never be used to make a clinical
> decision on its own. Every output requires independent clinician verification.
> See [Safety disclaimer](#7-safety-disclaimer).

---

## Table of contents

1. [Project purpose](#1-project-purpose)
2. [System architecture](#2-system-architecture)
3. [Planned ML pipeline](#3-planned-ml-pipeline)
4. [Planned SHAP/LIME explainability](#4-planned-shaplime-explainability)
5. [Planned Streamlit doctor dashboard](#5-planned-streamlit-doctor-dashboard)
6. [Planned FastAPI backend](#6-planned-fastapi-backend)
7. [Safety disclaimer](#7-safety-disclaimer)
8. [Local development setup](#8-local-development-setup)

---

## 1. Project purpose

### The problem

Diabetes risk is estimated in practice from a combination of laboratory values,
anthropometrics, history and clinical judgement. Machine-learning models can
pattern-match that data quickly, but they are typically deployed as black boxes:
a number comes out, and the clinician has no way to see *why*. In a clinical
setting that is not merely inconvenient — it is a blocker to adoption.

### What this project is

Cronical is a **production-quality portfolio prototype** for an explainable
diabetes risk decision-support system. It demonstrates three things working
together:

1. A **reproducible ML pipeline** — dataset provenance, leakage-safe splits,
   a baseline before a tuned model, and honestly measured metrics.
2. **Genuine explanations** — SHAP and LIME local and global attributions, shown
   to the clinician rather than hidden behind a score.
3. **Production-shaped delivery** — a FastAPI service and a Streamlit dashboard,
   with tests, linting, type checking and CI.

### What this project is not

| This is **not** | Because |
| --- | --- |
| A diagnostic system | It produces model output, not a diagnosis. Diagnosis requires a clinician and confirmatory testing. |
| A medical device | It is not validated, certified or intended for clinical use. |
| A treatment recommender | No medication, dosing or therapy advice is generated anywhere in this repository. |
| A source of clinical thresholds | Risk bands are *model-relative output ranges*, not clinical cut-offs. No clinical cut-off is hard-coded. |
| A finished ML system | **No model has been trained yet.** There are deliberately no metrics in this repository, because there is nothing measured to report. |

### Current status

**Stage 1 — architecture and engineering foundation: complete.**

- ✅ Directory layout, packaging and tooling configuration
- ✅ Environment-driven configuration module with a derived filesystem layout
- ✅ Structured logging utility
- ✅ CI pipeline (format, lint, types, tests, build)

**Stage 2 — dataset ingestion and validation: complete.**

- ✅ Explicit schema contract (`cronical.data.schema`) with the zero-sentinel rule
- ✅ Typed exception hierarchy for every failure mode
- ✅ CSV loader with configurable, project-root-derived paths — no network access
- ✅ Read-only validation producing a typed data-quality report

**Stage 3 — preprocessing pipeline: complete.**

- ✅ Zero-sentinel handling: `0 → NaN` in the five documented columns
- ✅ Median imputation, fitted on the training partition only
- ✅ Model-appropriate scaling — standardise for linear, omit for trees
- ✅ Stratified, deterministic, configurable train/test split
- ✅ Single fitted pipeline reused for training, testing and one-patient inference
- ✅ Optional joblib serialisation of the fitted preprocessor
- ✅ Leakage prevention asserted by test, not just by convention

**Stage 4 — baseline training and evaluation framework: complete.**

- ✅ Typed model configuration and a baseline registry with explicit hyperparameters
- ✅ Logistic Regression, Random Forest and XGBoost composed with the preprocessing pipeline
- ✅ Stratified, deterministic split; preprocessing fitted on training rows only
- ✅ Full metric set computed from probability scores, with undefined metrics reported as gaps
- ✅ Model comparison, evaluation report generation, and figures from real predictions
- ✅ joblib persistence with metadata stored separately from the pipeline

**No experiment has been run on the real dataset** — `data/raw/diabetes.csv` is
absent and is never downloaded automatically. See
[`reports/model_evaluation.md`](reports/model_evaluation.md), which states this
explicitly rather than containing invented figures.

**Not started.**

- ⬜ SHAP / LIME explainers
- ⬜ FastAPI service
- ⬜ Streamlit dashboard

`CRONICAL_MODEL_VERSION` defaults to `untrained`, and the API and dashboard are
required to report that no model is available rather than return a placeholder
prediction.

### Dataset

The project targets the **Pima Indians Diabetes** dataset, published by NIDDK and
distributed through the UCI Machine Learning Repository (dataset id 329). The CSV
is **not** included and is **never downloaded automatically** — you place it at
`data/raw/diabetes.csv` yourself. See [`data/raw/README.md`](data/raw/README.md)
for provenance, the full schema, and the rules governing raw data.

> The dataset is a specific cohort collected in the 1980s. It is **not** a
> representative sample of any wider population and does **not** describe current
> clinical practice. Results measured on it cannot be generalised beyond it.

---

## 2. System architecture

### Design principles

- **Modularity** — each concern lives in exactly one package with a single
  direction of dependency.
- **Strict layering** — `data` → `models` → `explainability` → `clinical` →
  `api` / `app`. Dependencies never point backwards.
- **No hard-coded paths** — every path derives from `Settings.project_root`.
- **Type hints and docstrings everywhere** — enforced by `mypy --strict` and
  `ruff`, not by good intentions.
- **Honesty by construction** — the code cannot report a metric or a
  recommendation that has not actually been computed.
- **Safety as a boundary** — the disclaimer is a package constant, not a
  paragraph someone can forget to paste.

### Repository layout

```text
cronical/
├── data/                     # Datasets. Contents are git-ignored.
│   ├── raw/                  # As-supplied, never modified. README.md + diabetes.csv.
│   └── processed/            # Deterministic outputs of the preprocessing pipeline.
├── notebooks/                # Exploration and training. Reproducible, not load-bearing.
├── src/cronical/             # The installable library.
│   ├── __init__.py           # Version, disclaimer constants, package contract.
│   ├── config.py             # Settings + derived Paths. The only source of paths.
│   ├── data/                 # Ingestion, validation, preprocessing.
│   │   ├── schema.py         # The column contract, incl. the zero-sentinel rule.
│   │   ├── errors.py         # Typed exception hierarchy.
│   │   ├── loader.py         # CSV loading. No network access.
│   │   ├── validation.py     # Read-only checks.
│   │   ├── report.py         # Typed data-quality report structures.
│   │   └── preprocessing.py  # Leakage-safe feature pipeline, train/test split.
│   ├── models/               # Baselines, training, evaluation, figures, reporting.
│   ├── explainability/       # SHAP and LIME. Reads artifacts; never trains.
│   ├── clinical/             # Clinician-facing rendering. The safety boundary.
│   └── utils/                # Cross-cutting helpers (logging).
├── models/                   # Persisted, versioned model artifacts (git-ignored).
├── app/                      # Streamlit dashboard entry points.
├── api/                      # FastAPI backend.
├── tests/                    # pytest suite.
├── reports/
│   └── figures/              # Generated plots (git-ignored).
├── .github/workflows/        # CI.
├── pyproject.toml            # Canonical dependency and tooling metadata.
├── requirements.txt          # Flat convenience list mirroring pyproject.
├── .env.example              # Documented environment variables.
└── .gitignore
```

### Package responsibilities and boundaries

| Package | Owns | Must never |
| --- | --- | --- |
| `cronical.config` | Settings, environment parsing, filesystem layout | Contain business logic |
| `cronical.data` | Dataset loading, schema contracts, preprocessing | Import `models` |
| `cronical.models` | Pipelines, estimators, training, evaluation | Import `api`, `app`, or `clinical` |
| `cronical.explainability` | SHAP / LIME attribution computation | Train or mutate a model |
| `cronical.clinical` | Risk-band rendering, disclaimer enforcement | Emit treatment advice or hard-code clinical cut-offs |
| `cronical.utils` | Logging and other cross-cutting helpers | Contain project-specific logic |
| `api` | HTTP transport, request validation, response schemas | Contain modelling logic |
| `app` | Presentation and session state | Contain modelling logic |

### Request flow

```text
 Clinician
     │
     ▼
┌──────────────────────┐   renders disclaimer + attributions
│  Streamlit (app/)    │──── needs explanations, not raw model calls
└──────────┬───────────┘
           │ HTTP (JSON)
           ▼
┌──────────────────────┐   validates schema, no business rules
│  FastAPI (api/)      │
└──────────┬───────────┘
           │ in-process calls
           ▼
┌──────────────────────┐   renders risk band + mandatory disclaimer
│  clinical/           │◄── the only layer allowed to speak to a clinician
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐   attributions from the fitted artifact
│  explainability/     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐   loads the persisted pipeline
│  models/             │
└──────────┬───────────┘
           │ offline only
           ▼
┌──────────────────────┐
│  data/               │
└──────────────────────┘
```

### Configuration

`src/cronical/config.py` exposes a `Settings` object (pydantic-settings) with a
nested `LogSettings` and a derived `Paths` object. Resolution order, highest
priority first:

1. Keyword arguments to `Settings(...)` — used by tests.
2. Environment variables prefixed with `CRONICAL_`.
3. The `.env` file at the project root.
4. Defaults declared in code.

Nested settings use a double underscore:

```bash
CRONICAL_LOG__LEVEL=DEBUG     # -> Settings.log.level
CRONICAL_LOG__FORMAT=json     # -> Settings.log.format
CRONICAL_PROJECT_ROOT=/srv    # -> Settings.project_root, and therefore every path
```

---

## 3. Planned ML pipeline

**Status: partially implemented.** Stages 3.1–3.9 (raw → validation →
preprocessing → leakage-safe split → baseline training → evaluation) are
complete. Interpretation, serving and the user interfaces are not, and are
documented below as the design contract the implementation will be held to.

### 3.1 The three stages

The data flow is deliberately split so that each stage has one job and cannot
quietly do another's.

```text
   ┌─────────────────────────┐
   │  RAW  data/raw/         │  Externally supplied CSV. Immutable.
   │  diabetes.csv           │  Never edited, never committed.
   └───────────┬─────────────┘
               │  loader.load_dataset()
               │  · path from settings, never hard-coded
               │  · exists? .csv? parses? rectangular?
               │  · values read exactly as written
               ▼
   ┌─────────────────────────┐
   │  VALIDATION             │  validation.validate_dataframe()
   │  read-only              │  · schema, dtypes, target, nulls,
   │  returns a report       │    duplicates, invalid values
   │  modifies nothing       │  · zero sentinels REPORTED, not fixed
   └───────────┬─────────────┘
               │  DataQualityReport
               ▼
   ┌─────────────────────────┐
   │  SENTINEL ZERO HANDLING │  SentinelZeroHandler
   │  0 → NaN, five columns  │  · Glucose, BloodPressure,
   │  only                   │    SkinThickness, Insulin, BMI
   │  raw file untouched     │  · Pregnancies 0 is a real value
   └───────────┬─────────────┘
               ▼
   ┌─────────────────────────┐
   │  MEDIAN IMPUTATION      │  SimpleImputer(strategy="median")
   │  fitted on TRAIN only   │  · medians learned from training rows
   │  keeps feature count    │  · statistics inside the pipeline
   └───────────┬─────────────┘
               ▼
   ┌─────────────────────────┐
   │  MODEL-SPECIFIC SCALING │  StandardScaler, or omitted
   │  linear → standardise   │  · tree ensembles: no scaling
   │  tree   → no scaling    │  · one flag, everything else shared
   └───────────┬─────────────┘
               │  fitted Pipeline (reused at inference)
               ▼
   ┌─────────────────────────┐
   │  MODEL TRAINING         │  cronical.models.train
   │  one fitted Pipeline    │  · LogisticRegression, standardised
   │  carries both stages    │  · RandomForest + XGBoost, unscaled
   │                         │  · hyperparameters from models.config
   └───────────┬─────────────┘
               │  held-out predictions only
               ▼
   ┌─────────────────────────┐
   │  EVALUATION             │  cronical.models.evaluate
   │  metrics from scores    │  · ROC-AUC / PR-AUC from probabilities
   │  never from 0/1 labels  │  · undefined metrics reported as gaps
   │                         │  · 0.5 threshold: technical, not clinical
   └─────────────────────────┘
```

**Nothing crosses a stage boundary silently.** Raw is never written; validation
never repairs; the loader never coerces; preprocessing never edits the raw file
and never sees the target.

### 3.2 Dataset, source and provenance

| | |
| --- | --- |
| Dataset | Pima Indians Diabetes |
| Publisher | National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) |
| Distributed by | UCI Machine Learning Repository, dataset id 329 |
| Collection era | 1980s |
| Local filename | `data/raw/diabetes.csv` |
| In this repository? | **No.** Supplied by you; never downloaded automatically. |

Full details: [`data/raw/README.md`](data/raw/README.md).

**Still to do:** record a provenance manifest with the source URL, licence,
retrieval date and SHA-256 checksum, so a result can be traced to an exact input
file. Until that exists, reproducibility rests on the operator's own record.

**Known limitations of this dataset** — properties of the data, not medical
claims:

- A specific cohort from a specific place and time, **not** a representative
  sample of any wider population.
- Does **not** describe current clinical practice, diagnostic criteria or
  treatment pathways.
- A research benchmark for comparing methods, not a clinical cohort assembled to
  answer a clinical question.
- Collected decades ago, so it cannot reflect contemporary practice.

Any result measured on this dataset inherits all four restrictions.

### 3.3 Data dictionary

The authoritative contract is `src/cronical/data/schema.py`. This table
documents it; if the two ever disagree, the code wins.

| # | Column | Type | Unit (per source docs) | What it records | Zero handling |
| --- | --- | --- | --- | --- | --- |
| 1 | `Pregnancies` | int | count | Number of pregnancies | Zero is a **real** observation |
| 2 | `Glucose` | int | mg/dL | Plasma glucose concentration | **Zero = not recorded** |
| 3 | `BloodPressure` | int | mm Hg | Diastolic blood pressure | **Zero = not recorded** |
| 4 | `SkinThickness` | int | mm | Triceps skin thickness | **Zero = not recorded** |
| 5 | `Insulin` | int | µU/mL | Two-hour serum insulin | **Zero = not recorded** |
| 6 | `BMI` | float | kg/m² | Weight ÷ height² | **Zero = not recorded** |
| 7 | `DiabetesPedigreeFunction` | float | — | Score summarising family history | Zero is ordinary data |
| 8 | `Age` | int | years | Age in years | Zero is **invalid** |
| 9 | `Outcome` | int | — | Binary class label, `0` or `1` | Not applicable — the target |

Caveats that apply to this table:

- **Units are documentation, not data.** The CSV contains no unit metadata. They
  are transcribed from the published dataset description and must be confirmed
  before any output is interpreted.
- The dataset documentation does **not** define the test or criteria behind
  `Outcome`, so this project does not restate or reinterpret what the label means.
- This table describes *data*. It contains no diagnostic thresholds, no reference
  ranges and no clinical decision rules, by design.

#### The zero-sentinel rule

`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` use `0` to mean
**"this measurement was not recorded"**. A recorded value of exactly zero in any
of them is not a physiological measurement, so accepting it as a real number
would silently corrupt everything computed from it.

| Stage | Treatment of zero sentinels |
| --- | --- |
| Raw file | Left exactly as supplied |
| Loader | Left exactly as written |
| Validation | **Counted and reported** as a warning per column; the data is untouched |
| Preprocessing | ⬜ Decision (impute / drop / model) to be made explicitly later |

They are **warnings**, not errors: this is a documented property of the dataset,
so failing hard would make the loader unusable on its own target file. They are
escalated to an **error** only when an entire sentinel column is zero, meaning
that column carries no usable measurement at all.

### 3.4 What validation checks

`validation.validate_dataframe(frame)` reads the frame and returns a
`DataQualityReport`. It never raises for a data problem, so one pass surfaces
every issue.

| Area | Checks |
| --- | --- |
| Structure | Dataset non-empty; all required columns present; outcome column named as expected; unexpected columns flagged |
| Types | Every predictor numeric; outcome numeric |
| Target | Outcome contains only `0` and `1`; distribution recorded |
| Content | No negative values; no zero `Age` |
| Reported | Duplicate rows; null cells; wholly-null columns; zero sentinels; wholly-zero sentinel columns |

Severity is explicit. **Errors** mean the dataset cannot be used (missing column,
non-numeric predictor, non-binary target, wholly-null column, impossible value).
**Warnings** mean a human should know (duplicates, some nulls, zero sentinels,
unknown columns).

Use `raise_for_errors(report)` when you would rather fail fast; the raised
`DatasetValidationError` still carries the full report.

The report contains row and column counts, column names, dtypes, per-column
missing counts, zero counts for sentinel columns, duplicate count, target class
distribution, and every error and warning with stable issue codes. Every figure is
computed from the dataset actually inspected — nothing is defaulted or carried
over.

### 3.5 Preprocessing — implemented

`cronical.data.preprocessing` builds an ordinary scikit-learn `Pipeline`, so the
*same fitted object* serves training, cross-validation, testing and
single-patient inference. Steps, in order:

| # | Step | Class | Learns |
| --- | --- | --- | --- |
| 1 | Contract | `FeatureContract` | nothing; validates and fixes column order |
| 2 | Sentinel zeros | `SentinelZeroHandler` | nothing; `0 → NaN` in five columns |
| 3 | Imputation | `SimpleImputer(strategy="median")` | per-feature medians |
| 4 | Scaling *(optional)* | `StandardScaler` | per-feature mean and scale |

#### Why zero sentinels become missing

`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` use `0` to mean
**"this measurement was not recorded"**. Nobody's blood pressure is zero, so a
recorded zero is a placeholder for absent data.

Left as a number, the pipeline would learn that "zero glucose" is a very low
value — quietly treating a missing measurement as an extreme observation, and
pulling the fitted median and scale towards it. Converting to `NaN` first and
imputing afterwards means every derived statistic comes from measurements that
were actually taken.

`Pregnancies` is deliberately excluded: a pregnancy count of zero is a real
observation, not a placeholder. `Outcome` never enters the pipeline at all.

#### Why median imputation

Missing measurements are unlikely to be missing at random — they may be absent
precisely because they were hard to obtain. The median sits in the middle of the
observed distribution and, unlike the mean, is not dragged around by a few
unusually large readings. It also does not manufacture a value outside the range
that was recorded.

Preferred over iterative or model-based imputation because it introduces no
second fitted model that would need its own validation.

> This is a **conservative default, not a claim that it is optimal.** Whether
> these rows should instead be dropped, or whether the missingness should be
> modelled explicitly as a feature, remains an open modelling decision. It is
> not settled by this code.

#### Why scaling differs by model family

| Model family | Scaling | Reason |
| --- | --- | --- |
| Logistic regression, other linear | `StandardScaler` | A coefficient is fitted per feature, so features measured on wildly different scales would dominate purely because of their units. |
| Random forest, XGBoost, other trees | none | Trees split on thresholds; only the *ordering* of values matters, so rescaling changes nothing while adding fitted parameters and a serialisation dependency. |

The choice is one explicit flag (`ScalingMode`). Both paths share every other
step, so the difference is genuinely a single step and the two cannot drift apart.

#### Public API

| Function | Purpose |
| --- | --- |
| `build_preprocessor()` | Assemble an unfitted pipeline |
| `build_training_pipeline(estimator)` | Preprocessing plus an estimator |
| `fit_preprocessor(X_train)` | Fit on the **training partition only** |
| `transform_features(pipeline, X)` | Transform a batch |
| `transform_patient(pipeline, record)` | Transform one record at inference |
| `split_dataset(frame)` | Stratified train/test split |
| `describe_preprocessor(pipeline)` | Audit trail of what was fitted |
| `save_preprocessor()` / `load_preprocessor()` | Persist and reuse the fitted pipeline |
| `dump_preprocessor()` | Write a JSON description for a report |

#### Reuse at inference

`transform_patient(pipeline, record)` runs one record through the *already fitted*
pipeline. Nothing is refitted, so inference sees exactly the transformations the
model was trained on:

```python
from cronical.data.loader import load_dataset
from cronical.data.preprocessing import fit_preprocessor, split_dataset, transform_patient

split = split_dataset(load_dataset())
pipeline = fit_preprocessor(split.X_train)  # fitted once, on training data

row = transform_patient(
    pipeline,
    {
        "Pregnancies": 2,
        "Glucose": 0,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 120,
        "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 38,
    },
)
```

That `Glucose` of `0` is treated as an unrecorded measurement and filled with the
training median — never passed through as a real value.

### 3.6 Splitting and leakage control — implemented

`split_dataset(frame)` returns a `DatasetSplit` holding `X_train`, `X_test`,
`y_train`, `y_test`. It validates the dataset first, so a missing column or a
non-binary outcome stops the split before anything is produced.

| Property | Behaviour |
| --- | --- |
| Deterministic | Seeded; `CRONICAL_RANDOM_SEED` by default, overridable per call |
| Stratified | Outcome balance preserved in both partitions, never optional |
| Configurable | `test_size` from settings or per call |
| Auditable | Row indices preserved, so any transformed row traces back to its source |
| Guarded | A class too rare to appear in both partitions raises `SplitError` |

#### Why statistics must be fitted on training data only

Any statistic learned from data — a median, a mean, a scale — carries information
about **every row it saw**. If the test partition contributes to those statistics,
the model is indirectly exposed to the answers it will be scored against, and the
reported metrics are optimistic for reasons that have nothing to do with
predictive skill. The effect is small but real, and it is worst for exactly the
high-dimensional, small-sample case this dataset represents.

Three mechanisms enforce the separation:

1. `split_dataset()` partitions before any fitting happens.
2. `fit_preprocessor()` is handed the training partition only, and holds no
   reference to the test data.
3. All transforms live **inside** the scikit-learn pipeline, so cross-validation
   refits them per fold and cannot leak the validation fold into the fit.

The tests assert this directly, against a dataset whose training median and
whole-dataset median differ measurably:

```text
training median of Glucose   105.0
whole-dataset median         110.0
fitted statistic             105.0   ← training only
```

The same check is applied to the scaler's fitted mean. A regression that fitted on
everything would show `110.0` and fail the test.

One further guarantee: `fit_preprocessor` accepts `y_train` for symmetry with
scikit-learn, but **the target is never used to fit anything**. A test asserts
that passing it changes no statistic.

#### Not yet handled

- Temporal splitting. This dataset is not time-ordered, so a random split is
  appropriate here; if temporal data were used, a random split would leak the
  future into the past.
- No grouping by patient. The dataset has one row per record, so there is nothing
  to group by — but a dataset with repeated visits would need it.

### 3.7 Model progression — baselines implemented

Three baselines are implemented and compared. **No hyperparameter tuning happens
at this stage**: the goal is a reliable reference point, not the best model this
data could support.

| Model | Preprocessing | Why it is in the comparison |
| --- | --- | --- |
| `LogisticRegression` | standardised | Interpretable floor. One readable coefficient per feature, and the fallback whenever an explainer is unavailable. |
| `RandomForestClassifier` | unscaled | Bagged trees. Invariant to monotone feature rescaling, robust to interactions and to outliers. |
| `XGBClassifier` | unscaled | Gradient boosting. Usually the strongest tabular baseline, and the easiest to overfit if left unbounded — hence the modest defaults. |

All hyperparameters, seeds and preprocessing paths live in
`cronical.models.config`, so a run is reproducible from its configuration alone.

### 3.8 Why three model families, and why accuracy alone is not enough

**Why compare families.** They fail differently, which is what makes the comparison
informative. A linear model is transparent and may be preferred *for that reason*
at a lower score. A forest is stable and resistant to feature scaling. A boosted
model may win on accuracy while being harder to interpret and easier to overfit.
A leaderboard between three untuned baselines on one split is the start of a
discussion, not the end of one.

**Why accuracy is insufficient.** On a dataset where one class is much more common
than the other, a model that predicts the majority class every time can post a
high accuracy while detecting almost nothing of interest. Accuracy does not
distinguish the two kinds of mistake: a false positive and a false negative are
counted identically, even though they carry very different weight in any
downstream decision.

That is why the metric set here is reported in full — precision, recall,
specificity, F1, **and** ROC-AUC and PR-AUC — and why no model is selected on a
single number.

**Ranking is not selection.** `compare_models()` orders by ROC-AUC for technical
comparison, and its documentation says so explicitly. ROC-AUC summarises behaviour
across *every* threshold, which makes it useful for comparing ranking ability and
useless for choosing an operating point. It also ignores class balance, which is
why PR-AUC is reported alongside it.

Nothing here is tuned, validated on held-out data for selection, or assessed for
calibration. Selecting a model, and selecting a threshold, are both later-stage
decisions that need evidence this stage does not produce.

### 3.9 Evaluation — implemented

`cronical.models.evaluate` computes, for each model on the held-out split:

| Group | Metrics |
| --- | --- |
| Counts | true negatives, false positives, false negatives, true positives |
| Ratios | accuracy, precision, recall, F1, specificity, sensitivity |
| Ranking | ROC-AUC, PR-AUC (average precision) |

Three rules govern this:

1. **Rank metrics come from probability scores**, never from thresholded labels. An
   AUC computed from 0/1 predictions discards the ordering information that makes
   it informative, and would report a different quantity.
2. **An undefined metric is a gap, not a zero.** Precision with no positive
   predictions, or ROC-AUC with a class absent, are reported as `None` and named in
   `Metrics.undefined` — so an absent number can never be read as a poor one.
3. **No metric is invented.** Every figure comes from arrays a fitted pipeline
   produced. No metric is written unless it was computed.

`compare_models()` orders results by ROC-AUC for technical comparison and says so
in its own documentation: that ordering is **not** a model selection.

Figures — confusion matrix, ROC curve, precision-recall curve — are drawn in
`cronical.models.figures` from real predictions and written to
`reports/figures/`. No placeholder graphic is ever generated.

#### What has not been done yet

- **No confidence intervals.** A single split gives one number per metric.
- **No repeated or cross-validated runs.**
- **No calibration assessment.** No Brier score or reliability curve yet.
- **No subgroup breakdown.** An aggregate figure can hide a group where a model
  performs far worse.

#### Report status

[`reports/model_evaluation.md`](reports/model_evaluation.md) currently contains
**no results**, because `data/raw/diabetes.csv` is absent and nothing is
downloaded automatically. It states that explicitly. Writing plausible figures
before any model has been fitted would be fabricating results.

### 3.10 Reproducibility — implemented

Every run records an `ExperimentMetadata` block:

| Field | Purpose |
| --- | --- |
| `dataset_path`, `dataset_sha256` | Ties the result to an exact input file |
| `feature_names`, `target` | What was modelled |
| `random_state`, `test_size` | Reproduces the split and the estimator |
| `decision_threshold` | The cut-off actually used when scoring |
| `preprocessing` | What the fitted transforms actually learned |
| `model_configuration` | Every hyperparameter, as set |
| `created_at`, `project_version` | When, and with which version of the code |

Artifacts are persisted under `models/` with joblib, and the metadata is written to
a **separate** `.json` beside the pipeline. Keeping them apart means provenance can
be read and diffed without unpickling anything.

`describe_preprocessor()` returns the fitted medians and scaling parameters, so a
run can be audited without refitting anything. Nothing is recomputed from data at
report time, and no metric is written into the metadata — metrics belong in a
report, not in an artifact.

Seeds are never changed implicitly. `train_all_baselines()` passes one seed to
every model so all three are scored on exactly the same partition; a difference
between two models is then a difference between the models, not between the data
each happened to see.

---

## 4. Planned SHAP/LIME explainability

**Status: not implemented.**

The goal is a clinician who can see *why* a number appeared, and an auditor who
can see *how much of the explanation is trustworthy*.

### 4.1 SHAP

- **TreeExplainer** for tree ensembles — exact and fast; no sampling noise.
- **KernelExplainer** as a model-agnostic fallback, with a documented
  approximation budget because it samples.
- **Global** views: mean absolute attribution per feature, beeswarm plot, and
  the importance-interaction matrix for correlated features.
- **Local** views: waterfall and force plots for a single prediction.

### 4.2 LIME

- Tabular explainer as an independent, model-agnostic cross-check on SHAP.
- Random seed pinned, so an explanation is reproducible rather than a different
  answer on every page load.
- Because LIME is stochastic and perturbs the data, its output must be labelled
  as a local approximation.

### 4.3 Comparing the two

Where SHAP and LIME disagree materially, the interface must **surface the
disagreement** rather than silently prefer one. Disagreement is information about
model fragility, and hiding it would defeat the purpose of building two
explainers.

### 4.4 Reporting rules

- Attributions are expressed in the model's output space (log-odds or
  probability), with the base value shown so the arithmetic reconciles.
- Explanations state that they describe **model behaviour, not physiology**.
  An attribution of `-0.18` to HbA1c means "this feature pushed the model's
  output down by 0.18 log-odds". It does not mean HbA1c is worth `-0.18` to a
  patient, and the UI must not let a reader infer otherwise.
- Correlation caveat: SHAP splits credit between correlated features. Where
  features are known to be correlated, this is stated next to the chart.
- Explanation objects are computed once in `explainability/` and serialised. The
  API and dashboard never recompute attributions per request.

---

## 5. Planned Streamlit doctor dashboard

**Status: not implemented.** Entry point planned at `app/main.py`.

### 5.1 Persistent safety banner

The disclaimer from `cronical.CLINICAL_DISCLAIMER` is rendered at the top of
every page, unconditionally. It is imported from the package constant so it
cannot drift from the one the API serves.

### 5.2 Clinician input panel

- Form inputs for each feature in the validated schema, with units, plausible
  ranges and explicit units on every field (mg/dL vs mmol/L).
- Validation feedback before submission, so a physiologically impossible input is
  caught rather than silently scored.
- Optional batch mode reading a CSV that conforms to the schema.

### 5.3 Output panels

1. **Prediction summary** — model probability, the risk band label, and a
   reminder that this is model output, not a diagnosis.
2. **Local explanation** — SHAP waterfall for the specific prediction, with the
   base value visible so the contributions reconcile to the output.
3. **LIME cross-check** — the model-agnostic view, side by side with SHAP, with
   disagreement highlighted.
4. **Global context** — dataset-level feature importance, so a single
   prediction can be sanity-checked against overall model behaviour.

### 5.4 UX rules

- If no model is loaded, the dashboard says so plainly and explains why. It
  never shows a placeholder prediction.
- Missing inputs produce an explicit "insufficient data to predict" state, not
  an imputed confident-looking score.
- No page may present treatment, medication or dosing suggestions. If such a
  feature is ever proposed, that is a scope change requiring clinical review —
  not a UI task.
- All figures are built through a shared plotting helper so charts are
  consistent and accessible.

---

## 6. Planned FastAPI backend

**Status: not implemented.** Entry point planned at `api/main.py`.

### 6.1 Design

- Pydantic request/response models mirroring the validated feature schema.
- **Thin transport layer**: validation, serialisation, and delegation to
  `clinical` / `explainability`. No modelling logic lives here.
- Dependency injection for `Settings`, so tests and the dashboard can override
  configuration without monkey-patching.
- OpenAPI docs at `/docs`, disabled only in production.

### 6.2 Planned endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness. Always answers, even with no model loaded. |
| `GET` | `/ready` | Readiness. `503` until a real artifact is available. |
| `GET` | `/api/v1/config` | Non-sensitive settings snapshot; `describe()` redacts secrets. |
| `GET` | `/api/v1/schema` | Feature schema, types and units. Drives the dashboard form. |
| `GET` | `/api/v1/model` | Model metadata: version, feature list, training provenance. |
| `POST` | `/api/v1/predict` | Single prediction with risk band **and** local explanation. |

### 6.3 Contract requirements

- Every prediction response embeds the disclaimer.
- Any request that cannot be served returns an explicit error envelope. There is
  no "default" prediction to fall back on.
- Response models are typed and validated on the way out, so an internal change
  cannot silently alter the published contract.
- Explanations are computed by `explainability/` and returned with the
  prediction; the client never calls the model directly.
- Unavailable-model responses must be distinguishable from error responses, so
  a client cannot mistake "no model yet" for "prediction failed".

---

## 7. Safety disclaimer

### The mandatory notice

> **AI-ASSISTED DECISION SUPPORT — NOT A DIAGNOSIS.**
>
> Cronical outputs are produced by a prototype research model and are intended
> only to support, never replace, professional clinical judgement. Predictions
> are unvalidated, may be wrong, and must be independently verified against the
> patient's own clinical assessment, history and examination before any care
> decision is made. Do not use these outputs for screening, diagnosis,
> treatment or dosing decisions.

This text is defined once, in `src/cronical/__init__.py` as
`CLINICAL_DISCLAIMER`, and is the single source for the dashboard banner and the
API response payload.

### Explicit constraints on this repository

1. **No diagnosis.** The system never asserts, confirms or excludes diabetes. A
   model output is a model output.
2. **No treatment guidance.** No medication, dosing, lifestyle or therapy
   recommendation is produced anywhere in this code base.
3. **No hard-coded clinical thresholds.** Any risk band is a *model-relative
   output range* described in the model's own terms. A clinical cut-off is a
   clinical decision requiring evidence and clinical governance; this project
   does not make one.
4. **No fabricated evidence.** No metric, confidence interval or performance
   claim may appear in code, notebooks, reports or this README unless it was
   computed by the pipeline. The absence of reported metrics in this repository
   is deliberate and honest.
5. **No fabricated data.** No dataset is bundled, synthesised or downloaded
   without a recorded licence and provenance.
6. **Known limitations must be stated**, not just known. Section 3.9 defines
   where limitations belong: next to the numbers.
7. **Fail loudly.** When the system cannot answer — no model, insufficient
   inputs, schema mismatch — it must say so instead of returning something that
   looks like an answer.

### Intended users

Clinicians, students and software engineers evaluating explainable decision
support. Not intended for direct patient self-assessment.

### Regulatory position

This project has not been submitted to any regulator, has not been clinically
validated, and is not a medical device. Real clinical deployment would require
regulatory clearance, prospective validation, audit trails, human-factors
testing and formal clinical governance. None of that is in scope here.

---

## 8. Local development setup

### Prerequisites

- Python **3.11+** (3.12 recommended)
- Git
- ~2 GB free disk if installing the full `[all]` extra

### Install

```bash
git clone https://github.com/OmkarPanchal06/cronical.git
cd cronical

python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install only what you need. The core extra is deliberately lightweight, so the
test suite runs on a fresh clone without a multi-minute download:

```bash
pip install -e ".[dev]"        # core + pytest, ruff, mypy  -> recommended to start
pip install -e ".[all]"        # everything, including SHAP, FastAPI and Streamlit
pip install -e ".[data,ml]"    # data science subset
pip install -e ".[api,ui]"     # serving and dashboard subset
```

Alternatively, from `requirements.txt`:

```bash
pip install -r requirements.txt
```

`pyproject.toml` is the canonical source of dependency metadata;
`requirements.txt` is a flat convenience mirror. Keep them in sync.

### Configure

```bash
cp .env.example .env            # Windows PowerShell: Copy-Item .env.example .env
```

`.env` is git-ignored. Defaults are sensible, so an empty `.env` works.

### Verify the install

```bash
python -c "from cronical.config import get_settings; print(get_settings().describe()['paths'])"
pytest -q
```

### Day-to-day commands

| Task | Command |
| --- | --- |
| Run tests | `pytest` |
| Tests with coverage | `pytest --cov=cronical --cov-report=html` then open `htmlcov/index.html` |
| Watch tests | `pytest -W error --lf -x` |
| Format | `ruff format .` |
| Check formatting | `ruff format --check --diff .` |
| Lint | `ruff check .` |
| Lint with auto-fix | `ruff check --fix .` |
| Type check | `mypy src` |
| Run everything | `pre-commit run --all-files` |
| Run the API (planned) | `uvicorn api.main:app --reload` |
| Run the dashboard (planned) | `streamlit run app/main.py` |
| Notebooks | `jupyter lab` |

### Using the library

```python
from cronical.config import get_settings
from cronical.utils.logging import get_logger

settings = get_settings()
log = get_logger(__name__)

log.info("project root resolved", extra={"project_root": settings.project_root})
print(settings.paths.raw_data_dir)  # always pathlib, never a hard-coded string
```

### Using the data layer

Place the CSV at `data/raw/diabetes.csv` first — see
[`data/raw/README.md`](data/raw/README.md). Nothing is downloaded for you.

```python
from cronical.data.loader import load_and_validate

frame, report = load_and_validate()  # data/raw/diabetes.csv by default
print(report.render())  # human-readable summary
report.to_dict()  # JSON-ready, for reports or an API

if not report.is_valid:
    for issue in report.errors:
        print(issue)
```

Load and validate separately if you want to handle them independently:

```python
from cronical.data.errors import DatasetNotFoundError
from cronical.data.loader import load_dataset
from cronical.data.validation import raise_for_errors, validate_dataframe

try:
    frame = load_dataset()
except DatasetNotFoundError as exc:
    print(exc)  # names the expected path
else:
    report = raise_for_errors(validate_dataframe(frame))
    print(f"{report.row_count} rows, {report.duplicate_count} duplicates")
    print(f"{len(report.errors)} errors, {len(report.warnings)} warnings")
```

Neither function modifies anything. The frame is returned exactly as parsed, and
validation only reads — zero-sentinel values are reported, never imputed.

### Using the preprocessing layer

Each snippet below is self-contained and can be run on its own.

```python
from cronical.config import ScalingMode
from cronical.data.loader import load_dataset
from cronical.data.preprocessing import fit_preprocessor, split_dataset, transform_features

frame = load_dataset()  # data/raw/diabetes.csv
split = split_dataset(frame)  # stratified, seeded

# Linear model: standardise.  Tree ensemble: pass ScalingMode.NONE instead.
pipeline = fit_preprocessor(split.X_train, scaling=ScalingMode.STANDARD)

features = transform_features(pipeline, split.X_test)
```

Fit **once**, on the training partition, then reuse that same fitted object for
evaluation and for inference:

```python
from cronical.config import ScalingMode
from cronical.data.loader import load_dataset
from cronical.data.preprocessing import (
    describe_preprocessor,
    fit_preprocessor,
    split_dataset,
    transform_patient,
)

split = split_dataset(load_dataset())
pipeline = fit_preprocessor(split.X_train, scaling=ScalingMode.STANDARD)

row = transform_patient(
    pipeline,
    {
        "Pregnancies": 2,
        "Glucose": 0,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 120,
        "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 38,
    },
)

# That Glucose of 0 was treated as an unrecorded measurement, not a real value.
describe_preprocessor(pipeline)  # audit trail: fitted medians, scaling, column order
```

`describe_preprocessor()` returns exactly what was learned, so a run can be
audited and reproduced. Nothing is recomputed from data at report time.

To persist the fitted pipeline and reuse it in a later process:

```python
from cronical.data.loader import load_dataset
from cronical.data.preprocessing import (
    fit_preprocessor,
    load_preprocessor,
    save_preprocessor,
    split_dataset,
)

split = split_dataset(load_dataset())
pipeline = fit_preprocessor(split.X_train)

save_preprocessor(pipeline)  # writes models/preprocessor.joblib
pipeline = load_preprocessor()  # reuses the same transformations
```

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `CRONICAL_APP_NAME` | `cronical` | Application name in logs. |
| `CRONICAL_ENVIRONMENT` | `local` | `local` \| `dev` \| `ci` \| `prod`. |
| `CRONICAL_DEBUG` | `false` | Verbose internal logging. |
| `CRONICAL_PROJECT_ROOT` | auto-detected | Root that every path derives from. |
| `CRONICAL_RANDOM_SEED` | `42` | Seed for estimators, splits and explainers. |
| `CRONICAL_MODEL_ARTIFACT_NAME` | `diabetes_risk_model.joblib` | Artifact filename. |
| `CRONICAL_MODEL_VERSION` | `untrained` | Artifact identifier. `untrained` = no model. |
| `CRONICAL_LOG__LEVEL` | `INFO` | `DEBUG`–`CRITICAL`. |
| `CRONICAL_LOG__FORMAT` | `text` | `text` or `json` (one object per line). |
| `CRONICAL_LOG__DATE_FORMAT` | ISO 8601 | Timestamp format. |
| `CRONICAL_LOG__TEXT_TEMPLATE` | `%(asctime)s \| %(levelname)-8s \| …` | `logging` format string used when `FORMAT=text`. |
| `CRONICAL_LOG__PROPAGATE` | `false` | Forward records to the interpreter-wide root logger. Leave `false` to avoid duplicate output. |
| `CRONICAL_PREPROCESSING__IMPUTATION_STRATEGY` | `median` | One of `median`, `mean`, `most_frequent`, `constant`. |
| `CRONICAL_PREPROCESSING__DEFAULT_SCALING` | `standard` | `standard` for linear models, `none` for tree ensembles. |
| `CRONICAL_PREPROCESSING__TEST_SIZE` | `0.2` | Fraction held out by `split_dataset()`. |
| `CRONICAL_PREPROCESSING__ARTIFACT_NAME` | `preprocessor.joblib` | Filename for the fitted preprocessor under `models/`. |
| `CRONICAL_TRAINING__DECISION_THRESHOLD` | `0.5` | Probability at which a prediction becomes positive. **A technical default, not a clinical cut-off.** |
| `CRONICAL_TRAINING__N_JOBS` | `-1` | Parallelism passed to estimators that support it. |
| `CRONICAL_API_HOST` | `127.0.0.1` | API bind address. |
| `CRONICAL_API_PORT` | `8000` | API port. |
| `CRONICAL_API_RELOAD` | `true` | Auto-reload for local development. |
| `CRONICAL_STREAMLIT_PORT` | `8501` | Dashboard port. |

### Project conventions

- **Type hints and docstrings are mandatory.** `mypy --strict` and `ruff` (`D`,
  `ANN`) enforce this in CI; they are not optional.
- **`pathlib` only.** `os.path` usage is rejected by the `PTH` ruff rules.
- **Logging over `print`.** Use `get_logger(__name__)`.
- **No relative imports above the parent package** (`ban-relative-imports`).
- **Format before pushing.** `ruff format . && ruff check --fix .`.

### CI

`.github/workflows/ci.yml` runs on Python 3.11, 3.12 and 3.13 and gates on
`ruff format --check`, `ruff check`, `mypy`, `pytest` and a wheel build. CI
installs only the `[dev]` extra, so the quality gate never depends on heavy ML
dependencies.

### Contributing

This is an educational project. Contributions that are in scope: correctness,
tests, documentation, explainability and engineering quality. Contributions that
are **out of scope**: anything that produces a diagnosis, treatment advice, or a
clinical threshold, and anything that inserts a performance number that has not
been measured.

---

## License

MIT. See the `license` field in `pyproject.toml`.

## Acknowledgements

This project exists to demonstrate engineering practice, not to provide medical
guidance. Any real-world use of predictive models in medicine requires
professional clinical judgement, validated evidence and formal governance.
