"""Typed configuration for the baseline estimators.

Every number a model is fitted with lives here, in one place, rather than being
scattered through the training code. That makes an experiment reproducible from
its configuration alone: the name, the seed, the hyperparameters and the
preprocessing path all travel together in a :class:`ModelSpec`.

Why three model families
------------------------
They fail differently, which is the point of comparing them:

``LogisticRegression``
    A linear baseline with an ``interpretable=True`` coefficient per feature. It
    is the floor: a more complex model that cannot beat it is not earning its
    complexity. It is also the only one of the three whose learned weights are
    directly readable.

``RandomForestClassifier``
    A bagged tree ensemble. Invariant to monotone feature rescaling, so it runs
    the unscaled preprocessing path. Robust to feature interactions and to
    outliers that would trouble a linear fit.

``XGBClassifier``
    Gradient-boosted trees, fitted sequentially and with explicit regularisation.
    Usually the strongest tabular baseline, and the most prone to overfitting if
    its capacity is left unbounded — which is why the defaults here are modest.

These are **baselines**, not a search. No hyperparameter tuning happens here.

Safety
------
Nothing in this module is a clinical threshold or a diagnostic rule. The values
are modelling choices, and the ``scaling`` field is a numerical convenience
rather than a statement about a model's suitability for any real use.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

from cronical.config import ScalingMode

__all__ = [
    "BASELINE_REGISTRY",
    "ModelSpec",
    "EstimatorName",
    "available_models",
    "get_spec",
    "model_names",
    "require_dependency",
]


class EstimatorName(StrEnum):
    """Identifiers for the baseline estimators."""

    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


#: Distributions that must be importable before a given estimator can be built.
_REQUIREMENTS: Final[Mapping[EstimatorName, tuple[str, ...]]] = MappingProxyType(
    {
        EstimatorName.LOGISTIC_REGRESSION: (),
        EstimatorName.RANDOM_FOREST: (),
        EstimatorName.XGBOOST: ("xgboost",),
    }
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """The complete, reproducible description of one baseline model.

    Attributes:
        name: Estimator identifier.
        hyperparameters: Constructor keyword arguments, with no defaults left to
            chance. Anything affecting the fitted result must be stated here.
        scaling: Which preprocessing path this model needs. Linear models need
            comparable feature scales; tree ensembles do not.
        requires: Third-party distributions needed to build this estimator.
        description: One line on why this model is in the comparison.
    """

    name: EstimatorName
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    scaling: ScalingMode = ScalingMode.NONE
    requires: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        """Freeze the hyperparameters so a spec cannot be mutated after creation."""
        object.__setattr__(
            self, "hyperparameters", MappingProxyType(dict(self.hyperparameters))
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the specification."""
        return {
            "name": self.name.value,
            "scaling": self.scaling.value,
            "hyperparameters": dict(self.hyperparameters),
            "requires": list(self.requires),
            "description": self.description,
        }


def _logistic_regression(random_state: int, n_jobs: int) -> ModelSpec:
    """Return the interpretable linear baseline.

    ``max_iter`` is raised well above the scikit-learn default because this data
    is not standardised into a very wide dynamic range by the caller, and a
    silent convergence warning would otherwise go unnoticed.
    """
    return ModelSpec(
        name=EstimatorName.LOGISTIC_REGRESSION,
        hyperparameters={
            "C": 1.0,
            "max_iter": 1_000,
            "solver": "lbfgs",
            "class_weight": None,
            "random_state": random_state,
        },
        scaling=ScalingMode.STANDARD,
        description="Interpretable linear baseline; one readable coefficient per feature.",
    )


def _random_forest(random_state: int, n_jobs: int) -> ModelSpec:
    """Return the bagged tree ensemble baseline.

    Deliberately unshallow and unsmeared: these defaults aim to be a reliable
    reference point, not the best forest this data could support.
    """
    return ModelSpec(
        name=EstimatorName.RANDOM_FOREST,
        hyperparameters={
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": random_state,
            "n_jobs": n_jobs,
        },
        scaling=ScalingMode.NONE,
        description="Bagged tree ensemble; insensitive to feature scale.",
    )


def _xgboost(random_state: int, n_jobs: int) -> ModelSpec:
    """Return the gradient-boosted tree baseline.

    ``max_depth=3`` and a low learning rate keep the model conservative: shallow
    trees with many of them are far harder to overfit than one deep tree.
    ``tree_method="hist"`` keeps fitting fast and deterministic.
    """
    return ModelSpec(
        name=EstimatorName.XGBOOST,
        hyperparameters={
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "tree_method": "hist",
            "eval_metric": "logloss",
        },
        scaling=ScalingMode.NONE,
        requires=("xgboost",),
        description="Gradient-boosted trees; sequential fit with explicit regularisation.",
    )


#: The three baselines compared in this stage, keyed by name.
BASELINE_REGISTRY: Final[Mapping[EstimatorName, ModelSpec]] = MappingProxyType(
    {
        EstimatorName.LOGISTIC_REGRESSION: _logistic_regression(0, -1),
        EstimatorName.RANDOM_FOREST: _random_forest(0, -1),
        EstimatorName.XGBOOST: _xgboost(0, -1),
    }
)

_BUILDERS = {
    EstimatorName.LOGISTIC_REGRESSION: _logistic_regression,
    EstimatorName.RANDOM_FOREST: _random_forest,
    EstimatorName.XGBOOST: _xgboost,
}


def model_names() -> tuple[str, ...]:
    """Return every baseline name, in a stable order."""
    return tuple(spec.name.value for spec in BASELINE_REGISTRY.values())


def require_dependency(name: EstimatorName) -> None:
    """Check that a baseline's third-party dependencies are importable.

    Raises:
        MissingDependencyError: If a required distribution is absent, naming it
            and the extra that provides it. Nothing is ever substituted: a
            missing estimator is an error, not an invitation to train a
            different model than the one that was asked for.
    """
    # Imported here rather than at module scope: the point is to fail only when
    # the specific estimator is requested, not merely imported.
    from importlib.util import find_spec

    from cronical.models.errors import MissingDependencyError

    spec = get_spec(name)
    for distribution in spec.requires:
        if find_spec(distribution) is None:
            raise MissingDependencyError(
                distribution=distribution,
                model=spec.name.value,
                extra="ml",
            )


def available_models() -> tuple[str, ...]:
    """Return the names of baselines whose dependencies are installed.

    Useful for reporting what a given environment can actually run, without
    triggering an error for the ones it cannot.
    """
    names = []
    for spec in BASELINE_REGISTRY.values():
        if all(_is_installed(distribution) for distribution in spec.requires):
            names.append(spec.name.value)
    return tuple(names)


def _is_installed(distribution: str) -> bool:
    """Return whether a distribution can be imported, without importing it."""
    from importlib.util import find_spec

    return find_spec(distribution) is not None


def get_spec(name: EstimatorName | str, *, random_state: int = 0, n_jobs: int = -1) -> ModelSpec:
    """Return a :class:`ModelSpec` with the seed and job count filled in.

    The registry holds specs pinned at ``random_state=0`` so it can be compared by
    identity; this function produces the copy an experiment actually runs with.

    Args:
        name: Estimator name or identifier.
        random_state: Seed for the estimator. Always set explicitly so a run is
            reproducible.
        n_jobs: Parallelism passed to estimators that support it.

    Returns:
        A new spec carrying the supplied seed.

    Raises:
        UnknownModelError: If ``name`` is not one of the baselines.
    """
    from cronical.models.errors import UnknownModelError

    try:
        resolved = EstimatorName(name)
    except ValueError as exc:
        raise UnknownModelError(f"unknown model {name!r}; expected one of {model_names()}") from exc
    return _BUILDERS[resolved](random_state=random_state, n_jobs=n_jobs)