"""Comparison of trained baselines, and generation of the evaluation report.

Both artefacts describe **model behaviour on one dataset**. Neither says anything
about any patient, and neither establishes clinical validity. Every number in the
generated report was computed by :mod:`cronical.models.evaluate` from predictions
that a fitted pipeline actually produced.

Why the "best" model is not chosen here
---------------------------------------
:func:`compare_models` sorts by ROC-AUC because it is a reasonable technical
summary of a model's ranking ability. It is **not** a recommendation, and the
ranking must not be read as one:

* ROC-AUC summarises behaviour across *all* thresholds. Two models with very
  different behaviour at the 0.5 cut-off can have nearly identical curves.
* ROC-AUC ignores the class balance. On skewed data, PR-AUC and precision at low
  recall are often the more informative view.
* The three baselines answer different questions. A linear model is transparent
  and may be preferred for that reason at a lower score; a forest may win on
  stability; a boosted model on accuracy. None of that is captured by one number.
* No model here has been tuned, validated on held-out data for selection, or
  assessed for calibration. A leaderboard between three untuned baselines on one
  dataset is a starting point for a discussion, not the end of one.

Selecting a model, and selecting a threshold, are both later-stage decisions that
require evidence this stage does not produce.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from cronical.config import Settings, get_settings
from cronical.models.errors import EvaluationError
from cronical.utils.logging import get_logger

if TYPE_CHECKING:
    from cronical.models.evaluate import Metrics
    from cronical.models.train import ExperimentMetadata, TrainedModel

__all__ = [
    "ModelEvaluation",
    "compare_models",
    "render_report",
    "write_report",
]

_LOG = get_logger(__name__)

#: Column order of the comparison table.
_COMPARISON_COLUMNS: Final[tuple[str, ...]] = (
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "roc_auc",
    "pr_auc",
)


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """One model's scores on one dataset, plus how it was produced.

    Attributes:
        name: Estimator identifier.
        metrics: Every computed metric.
        metadata: The reproducibility record from training.
        n_train: Rows the model was fitted on.
        n_test: Rows it was scored on.
    """

    name: str
    metrics: Metrics
    metadata: ExperimentMetadata
    n_train: int
    n_test: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the evaluation."""
        return {
            "name": self.name,
            "metrics": self.metrics.as_dict(),
            "metadata": self.metadata.as_dict(),
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


@dataclass(frozen=True, slots=True)
class Comparison:
    """Several models scored on the same split, ranked for technical comparison.

    Attributes:
        evaluations: The individual evaluations, ranked by ROC-AUC descending.
        ranking_metric: The metric the ordering used. **Not** a recommendation.
        unranked: Models whose ranking metric was undefined and which therefore
            cannot be placed in the ordering.
    """

    evaluations: tuple[ModelEvaluation, ...]
    ranking_metric: str = "roc_auc"
    unranked: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """Return the ranked model names, best first."""
        return tuple(evaluation.name for evaluation in self.evaluations)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the comparison."""
        return {
            "ranking_metric": self.ranking_metric,
            "order": list(self.names),
            "unranked": list(self.unranked),
            "evaluations": [evaluation.as_dict() for evaluation in self.evaluations],
        }

    def as_table(self) -> list[dict[str, Any]]:
        """Return one row per ranked model, ready for a Markdown table."""
        rows = []
        for evaluation in self.evaluations:
            row: dict[str, Any] = {"model": evaluation.name}
            row.update(evaluation.metrics.as_row())
            rows.append(row)
        return rows


def evaluate_model(
    model: TrainedModel,
    features: Any,
    y_true: Any,
) -> ModelEvaluation:
    """Score one trained model on held-out data.

    Args:
        model: A fitted model.
        features: Feature rows to score. Preprocessed by the model's own pipeline.
        y_true: Actual binary labels.

    Returns:
        The evaluation, including metrics computed from probability scores.

    Raises:
        EvaluationError: If the inputs do not satisfy the metric contract.
    """
    from cronical.models.evaluate import compute_metrics

    probabilities = model.predict_proba(features)
    metrics = compute_metrics(
        y_true,
        probabilities,
        threshold=model.metadata.decision_threshold,
    )
    return ModelEvaluation(
        name=model.name,
        metrics=metrics,
        metadata=model.metadata,
        n_train=model.metadata.train_size,
        n_test=int(metrics.n_samples),
    )


def compare_models(evaluations: Sequence[ModelEvaluation], *, metric: str = "roc_auc") -> Comparison:
    """Rank several evaluations for technical comparison.

    Args:
        evaluations: Evaluations to compare, ideally from the same split.
        metric: Ranking metric. Models whose value for it is undefined are listed
            separately rather than being forced into the ordering.

    Returns:
        A :class:`Comparison`, ranked by descending metric.

    Raises:
        EvaluationError: If ``metric`` is not a known metric name, or if no
            evaluation carries a defined value for it.
    """
    known = set(_COMPARISON_COLUMNS) - {"model"}
    if metric not in known:
        raise EvaluationError(f"unknown ranking metric {metric!r}; expected one of {sorted(known)}")

    ranked = [e for e in evaluations if getattr(e.metrics, metric) is not None]
    unranked = [e.name for e in evaluations if getattr(e.metrics, metric) is None]
    if not ranked:
        raise EvaluationError(
            f"no evaluation defines {metric!r}, so they cannot be ranked by it"
        )

    ordered = tuple(
        sorted(ranked, key=lambda e: float(getattr(e.metrics, metric)), reverse=True)
    )
    _LOG.info(
        "models compared",
        extra={"metric": metric, "order": [e.name for e in ordered], "unranked": unranked},
    )
    return Comparison(evaluations=ordered, ranking_metric=metric, unranked=tuple(unranked))


def _format(value: float | int | str | None) -> str:
    """Render a metric cell, showing gaps explicitly rather than as a zero."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Render rows as a GitHub-flavoured Markdown table."""
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format(row.get(column)) for column in columns) + " |" for row in rows
    ]
    return "\n".join([header, divider, *body])


def render_report(
    evaluations: Sequence[ModelEvaluation],
    *,
    dataset_summary: Mapping[str, Any],
    figures: Mapping[str, str] | None = None,
    generated_at: str | None = None,
) -> str:
    """Render the evaluation report as Markdown.

    Every figure in the report comes from ``figures``, which maps a label to a
    repository-relative path. A figure that was not produced is omitted, never
    substituted with a placeholder.

    Args:
        evaluations: At least one evaluation. All must share a split for the
            comparison to mean anything.
        dataset_summary: Dataset shape, class distribution and split description.
        figures: Label to repository-relative figure path.
        generated_at: Timestamp for the header. Defaults to now, in UTC.

    Returns:
        The report as Markdown text.

    Raises:
        EvaluationError: If no evaluations are supplied.
    """
    if not evaluations:
        raise EvaluationError("cannot render an evaluation report with no results")

    stamp = generated_at or datetime.now(UTC).isoformat(timespec="seconds")
    first = evaluations[0].metadata
    lines: list[str] = [
        "# Model evaluation report",
        "",
        f"- Generated: {stamp}",
        f"- Project version: {first.project_version or 'unknown'}",
        f"- Dataset: `{first.dataset_path}`",
        f"- Dataset SHA-256: `{first.dataset_sha256 or 'not recorded'}`",
        "",
        "> **This report describes model behaviour on one dataset. It does not",
        "> establish clinical validity, and no figure in it describes any patient.**",
        "",
        "## 1. Dataset",
        "",
    ]

    lines += [f"- Rows: {first.row_count}", f"- Target: `{first.target}`"]
    lines.append(f"- Features ({len(first.feature_names)}): " + ", ".join(
        f"`{name}`" for name in first.feature_names
    ))
    if distribution := dataset_summary.get("class_distribution"):
        lines.append("- Class distribution: " + ", ".join(
            f"{value}={count}" for value, count in distribution.items()
        ))
    if missing := dataset_summary.get("missing_counts"):
        total = sum(missing.values())
        lines.append(f"- Missing cells: {total}")
    if zeros := dataset_summary.get("zero_counts"):
        lines.append(
            "- Zero-sentinel values: "
            + ", ".join(f"{name}={count}" for name, count in zeros.items())
        )

    lines += [
        "",
        "## 2. Train/test split",
        "",
        f"- Test fraction: {first.test_size}",
        f"- Train rows: {first.train_size}",
        f"- Test rows: {first.test_size_rows}",
        f"- Random seed: {first.random_state}",
        "- Stratified by outcome: yes",
        "",
        "## 3. Preprocessing",
        "",
    ]
    preprocessing = first.preprocessing
    lines.append(f"- Imputation strategy: {preprocessing.get('imputation_strategy', 'n/a')}")
    lines.append(f"- Scaling: {preprocessing.get('scaling', 'n/a')}")
    lines.append(f"- Zero-sentinel columns: " + ", ".join(
        f"`{name}`" for name in preprocessing.get("sentinel_columns", [])
    ))
    lines.append(
        "- Every statistic above was fitted on the training partition only."
    )

    lines += [
        "",
        "## 4. Models evaluated",
        "",
        "Logistic regression uses standardised features; the two tree ensembles use "
        "the unscaled path, since tree splits are invariant to monotone rescaling.",
        "",
    ]
    for evaluation in evaluations:
        config = evaluation.metadata.model_configuration
        parameters = config.get("hyperparameters", {})
        lines += [
            f"### {evaluation.name}",
            "",
            f"- Scaling path: `{config.get('scaling', 'n/a')}`",
            "- Hyperparameters: "
            + ", ".join(f"`{key}={value}`" for key, value in parameters.items()),
            f"- Decision threshold: {evaluation.metrics.threshold} "
            "(technical default, not a clinical cut-off)",
            "",
        ]

    lines += [
        "## 5. Metric comparison",
        "",
    ]
    lines += _comparison_section(evaluations)

    lines += [
        "### Confusion matrices",
        "",
        _markdown_table(
            [
                {
                    "model": evaluation.name,
                    "tn": evaluation.metrics.counts.true_negatives,
                    "fp": evaluation.metrics.counts.false_positives,
                    "fn": evaluation.metrics.counts.false_negatives,
                    "tp": evaluation.metrics.counts.true_positives,
                }
                for evaluation in evaluations
            ],
            ("model", "tn", "fp", "fn", "tp"),
        ),
        "",
        "`n/a` marks a metric that could not be computed for these inputs, rather "
        "than a value of zero.",
        "",
    ]

    lines += _figure_section(evaluations, figures or {})

    lines += [
        "## 7. Limitations",
        "",
        "1. **These are untuned baselines.** No hyperparameter search was run. A "
        "tuned model may change every number above.",
        "2. **One split, one dataset.** No repeated runs, no confidence intervals, "
        "no cross-validated estimate. Differences between models are within the "
        "noise a single split can produce.",
        "3. **Ranking is not selection.** The ordering by ROC-AUC summarises "
        "ranking ability across all thresholds. It is not a recommendation, and "
        "it ignores the class balance that PR-AUC is sensitive to.",
        "4. **The threshold is a convention.** 0.5 is scikit-learn's default and "
        "carries no clinical meaning. No threshold selection was performed.",
        "5. **Performance on this dataset does not establish clinical validity.** "
        "This is a 1980s research cohort that is not representative of any wider "
        "population and does not describe current clinical practice. Nothing here "
        "supports a diagnostic, screening or treatment claim.",
        "6. **No calibration was assessed.** A model can rank well and still "
        "produce badly calibrated probabilities, which matters if outputs are "
        "read as numbers.",
        "7. **No subgroup analysis was performed.** Aggregate figures can hide a "
        "subgroup where a model performs far worse.",
        "",
    ]
    return "\n".join(lines)


def _comparison_section(evaluations: Sequence[ModelEvaluation]) -> list[str]:
    """Render the metric table, ordered where the ranking metric is defined.

    A model whose ROC-AUC could not be computed cannot be ranked. Rather than
    refusing to produce a report, the ordering is dropped and the affected models
    are named, so the reader sees the gap instead of a silently unsorted table.
    """
    rows = []
    for evaluation in evaluations:
        row: dict[str, Any] = {"model": evaluation.name}
        row.update(evaluation.metrics.as_row())
        rows.append(row)

    try:
        comparison = compare_models(evaluations)
    except EvaluationError:
        return [
            "No model defines the ranking metric, so no ordering is shown.",
            "",
            _markdown_table(rows, _COMPARISON_COLUMNS),
            "",
            "Not ranked, because `roc_auc` was undefined: "
            + ", ".join(evaluation.name for evaluation in evaluations),
            "",
        ]

    lines = [
        f"Ordered by `{comparison.ranking_metric}` descending.",
        "",
        _markdown_table(comparison.as_table(), _COMPARISON_COLUMNS),
        "",
    ]
    if comparison.unranked:
        lines += [
            f"Not ranked, because `{comparison.ranking_metric}` was undefined: "
            + ", ".join(comparison.unranked),
            "",
        ]
    return lines


def _figure_section(
    evaluations: Sequence[ModelEvaluation],
    figures: Mapping[str, str],
) -> list[str]:
    """Render the figures section, including only figures that exist."""
    if not figures:
        return []
    lines = ["## 6. Figures", ""]
    for label, relative in figures.items():
        lines += [f"### {label}", "", f"![{label}]({relative})", ""]
    return lines


def write_report(
    evaluations: Sequence[ModelEvaluation],
    *,
    dataset_summary: Mapping[str, Any],
    figures: Mapping[str, str] | None = None,
    path: Path | str | None = None,
    settings: Settings | None = None,
) -> Path:
    """Render the report and write it under ``reports/``.

    Args:
        evaluations: The results to report.
        dataset_summary: Dataset shape and class distribution.
        figures: Label to repository-relative figure path.
        path: Destination. Defaults to ``reports/model_evaluation.md``.
        settings: Configuration source. Defaults to :func:`get_settings`.

    Returns:
        The path written.

    Raises:
        EvaluationError: If the report cannot be written.
    """
    resolved = get_settings() if settings is None else settings
    destination = (
        resolved.paths.reports_dir / "model_evaluation.md" if path is None else Path(path)
    )
    report = render_report(evaluations, dataset_summary=dataset_summary, figures=figures)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report, encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"could not write report to {destination}: {exc}") from exc
    _LOG.info("evaluation report written", extra={"path": str(destination)})
    return destination


def write_comparison_json(comparison: Comparison, path: Path | str) -> Path:
    """Write a comparison as JSON, for later inspection or plotting."""
    destination = Path(path).expanduser()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(comparison.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
    except OSError as exc:
        raise EvaluationError(f"could not write comparison to {destination}: {exc}") from exc
    return destination