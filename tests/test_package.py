"""Structural tests for the ``cronical`` package itself.

These guard the project skeleton rather than any behaviour: if a sub-package is
renamed, deleted or left unimportable, the architecture contract is broken and
these tests should fail loudly.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess

import pytest

import cronical
from cronical.config import PROJECT_ROOT

SUBPACKAGES = (
    "cronical.config",
    "cronical.data",
    "cronical.models",
    "cronical.explainability",
    "cronical.clinical",
    "cronical.utils",
    "cronical.utils.logging",
)

ROOT_DIRECTORIES = (
    "data/raw",
    "data/processed",
    "notebooks",
    "models",
    "reports/figures",
    "app",
    "api",
    "tests",
    ".github/workflows",
)


class TestPackageMetadata:
    """Package identity and safety metadata."""

    def test_version_is_declared(self) -> None:
        assert cronical.__version__.count(".") == 2

    def test_disclaimer_is_present_and_explicit(self) -> None:
        disclaimer = cronical.CLINICAL_DISCLAIMER
        assert "NOT A DIAGNOSIS" in disclaimer
        assert "clinician" in disclaimer.lower()

    def test_disclaimer_forbids_clinical_use(self) -> None:
        lowered = cronical.CLINICAL_DISCLAIMER.lower()
        for forbidden in ("screening", "diagnosis", "treatment", "dosing"):
            assert forbidden in lowered, f"disclaimer must forbid {forbidden}"

    def test_not_a_diagnosis_flag_is_exposed(self) -> None:
        assert "NOT A DIAGNOS" in cronical.NOT_A_DIAGNOSIS.upper()

    def test_intended_use_names_the_audience(self) -> None:
        assert "educational" in cronical.INTENDED_USE.lower()
        assert "clinician" in cronical.INTENDED_USE.lower()

    def test_all_is_importable(self) -> None:
        for name in cronical.__all__:
            assert hasattr(cronical, name), f"{name} exported but missing"


@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_subpackage_imports(module_name: str) -> None:
    """Every architectural seam must import cleanly on a fresh install."""
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_subpackage_has_docstring(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__doc__, f"{module_name} is missing a module docstring"


@pytest.mark.parametrize("relative", ROOT_DIRECTORIES)
def test_repository_layout(relative: str) -> None:
    """The documented directory structure must exist in the checkout."""
    assert (PROJECT_ROOT / relative).is_dir(), f"missing directory: {relative}"


def _is_git_ignored(relative: str) -> bool:
    """Ask git whether it would ignore ``relative`` within the project root."""
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "check-ignore", "-q", "--", relative],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


GITKEEP_FILES = (
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "models/.gitkeep",
    "reports/figures/.gitkeep",
)

IGNORED_ARTIFACTS = (
    "data/raw/diabetes.csv",
    "data/raw/patients.csv",
    "data/processed/features.parquet",
    "data/processed/train.csv",
    "data/raw/survey.sav",
    "models/diabetes_risk_model.joblib",
    "reports/figures/shap_beeswarm.png",
    ".env",
    "coverage.xml",
    ".venv/lib/python3.12/site-packages/x.py",
)

TRACKED_FILES = (
    "src/cronical/config.py",
    "src/cronical/utils/logging.py",
    "tests/test_package.py",
    ".env.example",
    "README.md",
    "data/raw/README.md",
)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
class TestGitignore:
    """Repository hygiene rules.

    A ``.gitignore`` that hides its own ``.gitkeep`` markers silently destroys
    the directory skeleton on the next clone, and the resulting failure surfaces
    far away from its cause. These assertions pin both halves of the contract:
    structure is tracked, content is not.
    """

    @pytest.mark.parametrize("relative", GITKEEP_FILES)
    def test_gitkeep_markers_are_tracked(self, relative: str) -> None:
        assert not _is_git_ignored(relative), f"{relative} must remain trackable"

    @pytest.mark.parametrize("relative", IGNORED_ARTIFACTS)
    def test_data_and_artifacts_are_ignored(self, relative: str) -> None:
        assert _is_git_ignored(relative), f"{relative} must be git-ignored"

    @pytest.mark.parametrize("relative", TRACKED_FILES)
    def test_source_files_are_tracked(self, relative: str) -> None:
        assert not _is_git_ignored(relative), f"{relative} must remain trackable"
