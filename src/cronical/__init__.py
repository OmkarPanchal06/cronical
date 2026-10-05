"""Cronical: an explainable-AI decision-support prototype for diabetes risk review.

.. warning::

   **This project is not a diagnostic device and it does not diagnose diabetes.**
   Cronical produces *research-grade decision support* only. Every score, risk
   band and explanation produced by this code base must be interpreted and
   verified by a qualified clinician before it influences any patient care
   decision. See :data:`CLINICAL_DISCLAIMER` and :data:`NOT_A_DIAGNOSIS`.

The package is deliberately split along architectural seams so that each
concern can evolve, test and be replaced independently:

``cronical.config``
    Environment-driven settings and the canonical filesystem layout. Every other
    module reads paths from here instead of hard-coding directories.
``cronical.data``
    Dataset acquisition, schema validation and preprocessing. Contains no
    modelling logic.
``cronical.models``
    Feature pipelines, estimators, training and evaluation. Produces an
    auditable model artifact and an honest metrics report.
``cronical.explainability``
    Post-hoc explanation of model behaviour (for example SHAP and LIME). Reads a
    fitted artifact; it never trains anything.
``cronical.clinical``
    Mapping model output onto clinician-facing decision support. This layer is
    the only place allowed to render risk language, and it is required to carry
    the disclaimer to end users.
``cronical.utils``
    Cross-cutting helpers such as structured logging.

Non-goals for this repository
-----------------------------
* It must not be used to diagnose, confirm or rule out diabetes in a real
  patient.
* It must not emit treatment, medication or dosing advice.
* It must not present unreproducible or fabricated performance numbers.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "CLINICAL_DISCLAIMER",
    "INTENDED_USE",
    "NOT_A_DIAGNOSIS",
    "__version__",
]

__version__: Final[str] = "0.1.0"

#: Short machine-readable flag for callers that must surface a banner.
NOT_A_DIAGNOSIS: Final[str] = "NOT A DIAGNOSTIC TOOL"

#: Stated purpose of the artefact, for display in the UI and the API.
INTENDED_USE: Final[str] = (
    "Educational and portfolio research artefact demonstrating explainable machine "
    "learning for diabetes risk decision support. Intended for clinician review, "
    "method demonstration and software engineering evaluation only."
)

#: Mandatory user-facing disclaimer. The dashboard and the API are both required
#: to display this text next to any prediction output.
CLINICAL_DISCLAIMER: Final[str] = (
    "AI-ASSISTED DECISION SUPPORT - NOT A DIAGNOSIS. Outputs are produced by a "
    "prototype research model and are intended only to support, never replace, "
    "professional clinical judgement. Predictions are unvalidated, may be wrong, "
    "and must be independently verified by a qualified clinician against the "
    "patient's own clinical assessment, history and examination before any care "
    "decision is made. Do not use these outputs for screening, diagnosis, "
    "treatment or dosing decisions."
)
