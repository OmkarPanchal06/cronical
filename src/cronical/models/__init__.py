"""Modelling layer: baselines, training and evaluation.

Scope
-----
This package owns the estimator and the measurement of its behaviour. It does not
own preprocessing (that lives in :mod:`cronical.data.preprocessing`, and is
composed into the same pipeline), and it does not own interpretation (that comes
later in :mod:`cronical.explainability`).

Current contents
----------------
``config``
    Typed :class:`~cronical.models.config.ModelSpec` objects and the baseline
    registry. Every hyperparameter, seed and preprocessing path lives here, so a
    run is reproducible from its configuration alone.
``train``
    Dataset loading with validation, pipeline composition, fitting, experiment
    metadata and joblib persistence.
``evaluate``
    Binary-classification metrics computed from probability scores, with
    undefined metrics reported as gaps rather than zeros.
``reporting``
    Comparison of trained models and generation of the evaluation report.
``figures``
    Confusion-matrix, ROC and precision-recall plots drawn from real predictions.
``errors``
    Typed exceptions.

Modelling policy
----------------
* **Baselines only.** No hyperparameter tuning. The point of this stage is a
  reliable reference point, not the best model this data could support.
* **One pipeline.** Preprocessing and the estimator are fitted together, so
  learned statistics come from training rows only and inference cannot drift.
* **Report what was measured.** Every figure comes from a fitted pipeline's real
  predictions. A metric is never invented, carried over, or filled in with a
  placeholder.
* **Say when a metric is unavailable.** A metric that cannot be computed is
  reported as a gap and named, so it is never mistaken for a value of zero.

Safety
------
Nothing in this package produces a diagnosis, a clinical threshold, or a
treatment recommendation. The only threshold involved is the conventional 0.5
classifier cut-off, which is a technical default with no clinical meaning.

Performance measured on this dataset **does not establish clinical validity**.
The dataset is a 1980s research cohort, not representative of any wider
population, and does not describe current clinical practice.
"""

from __future__ import annotations

from cronical.models.config import (
    BASELINE_REGISTRY,
    EstimatorName,
    ModelSpec,
    available_models,
    get_spec,
    model_names,
    require_dependency,
)
from cronical.models.errors import (
    ArtifactError,
    CronicalModelError,
    EvaluationError,
    MissingDependencyError,
    TrainingError,
    UnknownModelError,
)
from cronical.models.evaluate import (
    DECISION_THRESHOLD,
    ConfusionCounts,
    Metrics,
    compute_metrics,
    confusion_counts,
    prediction_summary,
    pr_curve_points,
    roc_curve_points,
)
from cronical.models.reporting import (
    Comparison,
    ModelEvaluation,
    compare_models,
    evaluate_model,
    render_report,
    write_report,
)
from cronical.models.train import (
    ExperimentMetadata,
    TrainedModel,
    artifact_path,
    build_model_pipeline,
    load_metadata,
    load_model,
    load_training_frame,
    save_model,
    train_all_baselines,
    train_model,
)

__all__ = [
    "BASELINE_REGISTRY",
    "DECISION_THRESHOLD",
    "ArtifactError",
    "Comparison",
    "ConfusionCounts",
    "CronicalModelError",
    "EstimatorName",
    "EvaluationError",
    "ExperimentMetadata",
    "Metrics",
    "MissingDependencyError",
    "ModelEvaluation",
    "ModelSpec",
    "TrainedModel",
    "TrainingError",
    "UnknownModelError",
    "artifact_path",
    "available_models",
    "build_model_pipeline",
    "compare_models",
    "compute_metrics",
    "confusion_counts",
    "evaluate_model",
    "get_spec",
    "load_metadata",
    "load_model",
    "load_training_frame",
    "model_names",
    "prediction_summary",
    "pr_curve_points",
    "require_dependency",
    "render_report",
    "roc_curve_points",
    "save_model",
    "train_all_baselines",
    "train_model",
    "write_report",
]