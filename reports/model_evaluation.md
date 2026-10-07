# Model evaluation report

> **STATUS: NOT YET RUN.**
>
> This file contains **no results**. No experiment has been executed against the
> Pima Indians Diabetes dataset, because `data/raw/diabetes.csv` is not present in
> this repository. Nothing is downloaded automatically, so the dataset must be
> supplied by a developer — see [`../data/raw/README.md`](../data/raw/README.md).
>
> Writing plausible-looking metrics into this file before any model has been fitted
> would be fabricating results. That has not been done, and will not be.

---

## What this report will contain once an experiment has run

Every figure below will be computed by `cronical.models.evaluate` from the
predictions of a fitted pipeline. Nothing is carried over from a previous run,
defaulted, or estimated.

| Section | Content |
| --- | --- |
| 1. Dataset | Row and column counts, target, feature list, class distribution, missing-cell and zero-sentinel counts |
| 2. Train/test split | Test fraction, partition sizes, random seed, confirmation of stratification |
| 3. Preprocessing | Imputation strategy, scaling path, sentinel columns, and the fitted statistics |
| 4. Models evaluated | Each baseline with its full hyperparameter set and preprocessing path |
| 5. Metric comparison | Accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC, ordered by ROC-AUC |
| Confusion matrices | TN, FP, FN, TP per model |
| 6. Figures | Confusion matrix, ROC curve and precision-recall curve per model |
| 7. Limitations | What the numbers do and do not establish |

## Metrics that will be reported

* **Count-based**: true negatives, false positives, false negatives, true positives
* **Ratio-based**: accuracy, precision, recall, F1, specificity, sensitivity
* **Ranking**: ROC-AUC and average precision (PR-AUC)

ROC-AUC and PR-AUC are computed from **probability scores**, never from
thresholded class predictions — an AUC computed from 0/1 labels discards the
ordering information that makes it informative.

A metric that cannot be computed for a given input — precision when nothing is
predicted positive, ROC-AUC when a class is absent — is reported as `n/a` and
named explicitly. It is never replaced with a zero.

## The decision threshold

Predictions use a threshold of **0.5**, which is scikit-learn's conventional
default.

**0.5 is a technical default, not a medically validated threshold.** It is not
derived from this dataset, from any diagnostic criterion, or from any clinical
guideline. Threshold selection is out of scope for this stage and must be
justified on validation data later — never chosen by inspecting test performance.

## How to run the experiment

```python
from cronical.data.preprocessing import split_dataset
from cronical.models import evaluate_model, train_all_baselines
from cronical.models.reporting import compare_models, write_report

frame, path = load_training_frame()  # data/raw/diabetes.csv
models = train_all_baselines(frame, random_state=42, test_size=0.25, dataset_path=path)

split = split_dataset(frame, test_size=0.25, random_state=42)
evaluations = [evaluate_model(m, split.X_test, split.y_test) for m in models]

print(compare_models(evaluations).as_table())
write_report(evaluations, dataset_summary={...}, figures={...})
```

## Limits of what these numbers will mean

1. **Untuned baselines.** No hyperparameter search has been run. A tuned model may
   change every figure.
2. **One split, one dataset.** No repeated runs, no confidence intervals, no
   cross-validated estimate. Differences between models can fall entirely within
   the noise a single split produces.
3. **Ranking is not selection.** Ordering by ROC-AUC summarises ranking ability
   across all thresholds. It is not a recommendation and it ignores class balance.
4. **Performance on this dataset does not establish clinical validity.** This is a
   1980s research cohort, not representative of any wider population, and it does
   not describe current clinical practice. It cannot support a diagnostic,
   screening or treatment claim.
5. **No calibration, no subgroup analysis, no external validation.** A model can
   rank well and still produce badly calibrated probabilities, and aggregate
   figures can hide a subgroup where it performs far worse.

> This project is an educational prototype of **AI-assisted decision support**. It
> does not diagnose diabetes. Any output requires independent verification by a
> qualified clinician.