"""Clinical decision-support layer.

Scope
-----
This is the only layer permitted to translate model output into language that
appears in front of a clinician. It is a presentation and safety boundary, not a
diagnostic engine.

Hard rules for this package
---------------------------
1. Every rendered object carries :data:`cronical.CLINICAL_DISCLAIMER`.
2. Risk bands are **model-relative output ranges**, labelled as such. They are
   not diagnostic categories, they are not clinical thresholds, and they must
   never be presented as if they were.
3. Any threshold used here is explicitly configured and version-controlled. No
   clinical cut-off is hard-coded.
4. No treatment, medication or dosing recommendation is generated anywhere in
   this repository.
5. A missing or unavailable prediction is reported as "insufficient data"
   rather than being imputed into a confident answer.

Status
------
Intentionally empty. Risk-band mapping is not implemented yet, and no threshold
values have been chosen.
"""

from __future__ import annotations

__all__: list[str] = []
