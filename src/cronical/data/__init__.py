"""Data layer: acquisition, validation and preprocessing.

Scope
-----
This package owns everything between "a dataset exists somewhere" and "a clean,
modelling-ready table exists". It deliberately contains **no** model code, so the
same preprocessing can be reused for training, cross-validation and inference
without dragging estimators along.

Planned responsibilities
------------------------
* Dataset provenance: record the source, licence, retrieval date and checksum of
  every raw file so results stay reproducible.
* Schema validation: expected feature names, dtypes, units and missing-value
  conventions, checked with an explicit contract rather than assumed.
* A single, deterministic :func:`preprocess` entry point encapsulating cleaning,
  encoding and scaling, fitted on the training split only.
* Serialised preprocessing artifacts, so inference applies exactly the
  transformations the model was trained on.

Status
------
Intentionally empty at this stage. No dataset has been downloaded, and no
preprocessing logic has been written yet.
"""

from __future__ import annotations

__all__: list[str] = []
