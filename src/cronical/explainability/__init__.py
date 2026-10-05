"""Explainability layer: post-hoc interpretation of model behaviour.

Scope
-----
This package explains *why* a fitted model produced a prediction. It consumes a
model artifact from :mod:`cronical.models` and never trains, tunes or alters a
model. Keeping the two apart means explanations cannot quietly drift away from
the model they claim to describe.

Planned responsibilities
------------------------
* SHAP global and local attribution, with the TreeExplainer fast path for
  tree ensembles and a Kernel fallback for model-agnostic use.
* LIME tabular explanations as a secondary, model-agnostic cross-check. A
  disagreement between SHAP and LIME is surfaced to the user rather than hidden.
* Stable, serialisable explanation objects (attribution tables, waterfall and
  beeswarm payloads) so the API and dashboard never recompute attributions.
* Honest reporting: SHAP describes model behaviour, not physiology. An
  attribution of ``-0.18`` to HbA1c means "this pushed the model's output down
  by 0.18 log-odds", **not** "HbA1c is worth -0.18 to the patient".

Status
------
Intentionally empty. No explainer has been wired up yet.
"""

from __future__ import annotations

__all__: list[str] = []
