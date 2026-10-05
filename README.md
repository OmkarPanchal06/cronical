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

This repository currently contains the **architecture and engineering
foundation only**:

- ✅ Directory layout, packaging and tooling configuration
- ✅ Environment-driven configuration module with a derived filesystem layout
- ✅ Structured logging utility
- ✅ Test suite covering the foundation (config, logging, package structure)
- ✅ CI pipeline (format, lint, types, tests, build)
- ⬜ Dataset acquisition and provenance tracking
- ⬜ Preprocessing pipeline
- ⬜ Model training and evaluation
- ⬜ SHAP / LIME explainers
- ⬜ FastAPI service
- ⬜ Streamlit dashboard

`CRONICAL_MODEL_VERSION` defaults to `untrained`, and the API and dashboard are
required to report that no model is available rather than return a placeholder
prediction.

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
│   ├── raw/                  # As-downloaded, never modified. Provenance recorded.
│   └── processed/            # Deterministic outputs of the preprocessing pipeline.
├── notebooks/                # Exploration and training. Reproducible, not load-bearing.
├── src/cronical/             # The installable library.
│   ├── __init__.py           # Version, disclaimer constants, package contract.
│   ├── config.py             # Settings + derived Paths. The only source of paths.
│   ├── data/                 # Ingestion, schema validation, preprocessing.
│   ├── models/               # Feature pipelines, estimators, training, evaluation.
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

**Status: not implemented.** This section is the design contract that the
implementation will be held to.

### 3.1 Dataset and provenance

- Select a public, citable diabetes dataset and record **source URL, licence,
  retrieval date and SHA-256 checksum** in a provenance manifest written to
  `data/raw/`. Results must be traceable to an exact input.
- Establish the **prediction target and prediction horizon** explicitly and
  document them. A model that predicts "is this patient diabetic *now*" and one
  that predicts "will this patient develop diabetes within N years" are
  different products with different ethics, and the distinction must not be
  blurred.
- Treat any dataset that already encodes a diagnostic decision as a **modelling
  shortcut**: predicting it back is circular and its metrics are meaningless.

### 3.2 Schema validation

Define an explicit contract — feature names, dtypes, units, expected ranges and
missing-value conventions — and validate against it on load. Missing features or
unit drift should fail loudly at the boundary, not silently produce a wrong
prediction.

### 3.3 Preprocessing

A single deterministic entry point, fitted on the training split only:

- Missing-value handling, with imputation statistics recorded in the artifact.
- Encoding for categorical features.
- Scaling where the estimator requires it.
- Optional, **explicitly justified** feature transforms — no silent feature
  engineering whose effect cannot be explained later.

### 3.4 Splitting and leakage control

- One deterministic split, created once and reused for every experiment.
- Stratified splitting when the outcome is imbalanced.
- **All transforms fitted inside the pipeline**, so cross-validation cannot leak
  information from the validation fold.
- If temporal data is used, split chronologically. A random split on
  time-ordered clinical data leaks the future into the past.

### 3.5 Model progression

| Stage | Model | Purpose |
| --- | --- | --- |
| Baseline | Logistic regression | Interpretable floor. If a complex model cannot beat it, ship the baseline. |
| Candidate | Gradient-boosted trees | Expected to outperform on tabular data. |
| Ensemble | Stacked / soft-voting | Only if it earns its complexity. |

A logistic-regression baseline is not a formality — with an
`interpretable=True` sklearn model, its coefficients are themselves a form of
explanation, and it is the fallback whenever SHAP is unavailable.

### 3.6 Evaluation

Metrics appropriate for an imbalanced binary classification problem:
ROC-AUC, PR-AUC, sensitivity, specificity, PPV, NPV, and calibration
(Brier score, calibration curve). Chosen on a **held-out split never used for
model selection**.

Reported honestly:

- Point estimates **with confidence intervals**, not bare numbers.
- Performance broken down by relevant subgroups, so a good average does not
  hide a subgroup where the model is worse than chance.
- Limitations documented alongside the numbers.
- **No metric is written to a report unless it was computed by
  `models/evaluation.py`.** Placeholder values are forbidden. Until a model is
  trained, `reports/` states that no model has been fitted.

### 3.7 Reproducibility

Seeded via `CRONICAL_RANDOM_SEED`; artifacts persisted under `models/` with the
resolved config and data checksum embedded, so a given artifact can always be
traced back to the code and data that produced it.

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
6. **Known limitations must be stated**, not just known. Section 3.6 defines
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
