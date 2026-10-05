"""Modelling layer: features, estimators, training and evaluation.

Scope
-----
This package will own the machine-learning pipeline and nothing else. It must not
know about HTTP, Streamlit, or how a clinician reads a result; it accepts tidy
features and returns predictions plus honest, reproducible metrics.

Planned responsibilities
------------------------
* Column transformers for numeric and categorical features, assembled inside a
  :class:`~sklearn.pipeline.Pipeline` so cross-validation cannot leak.
* Baseline models (for example logistic regression) established *before* any
  higher-capacity model, so improvements are attributable.
* A configurable trainer returning a fitted pipeline plus a metrics report.
* Strict leakage controls: the split is created once, stratified where the
  outcome is imbalanced, and reused for every experiment.

Metrics policy
--------------
Metrics are only ever written after they have been measured by
``src/cronical/models/evaluation.py``. Placeholder numbers are forbidden; until a
model is trained the honest report states that no model has been fitted.

Status
------
Intentionally empty. The model is not implemented yet.
"""

from __future__ import annotations

__all__: list[str] = []
