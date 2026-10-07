"""Typed exceptions for the modelling layer.

Every failure mode the model layer can produce is represented here, so callers
can distinguish a missing dependency from an unknown model name from a dataset
that will not validate — without parsing message strings.

Hierarchy::

    CronicalModelError
    +-- UnknownModelError
    +-- MissingDependencyError
    +-- TrainingError
    +-- EvaluationError
    +-- ArtifactError

Nothing in this module describes a clinical condition. These errors report
engineering failures only.
"""

from __future__ import annotations

__all__ = [
    "ArtifactError",
    "CronicalModelError",
    "EvaluationError",
    "MissingDependencyError",
    "TrainingError",
    "UnknownModelError",
]


class CronicalModelError(Exception):
    """Base class for every error raised by :mod:`cronical.models`."""


class UnknownModelError(CronicalModelError):
    """The requested estimator is not one of the registered baselines."""


class MissingDependencyError(CronicalModelError):
    """A baseline's third-party dependency is not installed.

    Raised instead of silently substituting a different model. Training an
    estimator other than the one that was requested would produce results that
    do not correspond to the configuration being reported.

    Args:
        distribution: The distribution that could not be found.
        model: The model that needs it.
        extra: The project extra that provides it.
    """

    def __init__(self, distribution: str, *, model: str, extra: str) -> None:
        self.distribution = distribution
        self.model = model
        self.extra = extra
        super().__init__(
            f"The '{model}' baseline requires '{distribution}', which is not "
            f"installed. Install it with: pip install 'cronical[{extra}]'. "
            "No substitute model will be used."
        )


class TrainingError(CronicalModelError):
    """A model could not be trained on the supplied data."""


class EvaluationError(CronicalModelError):
    """Metrics could not be computed from the supplied inputs."""


class ArtifactError(CronicalModelError):
    """A model artifact could not be written or read."""