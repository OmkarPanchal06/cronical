"""Guard against documentation drifting away from the settings schema.

``.env.example`` and the README both advertise environment variables. If someone
renames a setting and forgets the docs, a developer's local setup silently stops
working. These tests make that drift a hard failure instead.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from cronical.config import ENV_PREFIX, PROJECT_ROOT, Environment, LogFormat, Settings

ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
README = PROJECT_ROOT / "README.md"

#: Any ``CRONICAL_*`` identifier, as written in prose or in a table cell.
MENTION = re.compile(rf"\b({ENV_PREFIX}[A-Z0-9_]+?)\b")

#: A real assignment line, optionally commented out: ``CRONICAL_LOG__LEVEL=X``.
ASSIGNMENT = re.compile(rf"^(?:#\s*)?({ENV_PREFIX}[A-Z0-9_]+)\s*=", re.MULTILINE)


def _documented_keys(path: Path) -> set[str]:
    """Return every ``CRONICAL_*`` variable name mentioned anywhere in ``path``."""
    if not path.is_file():
        return set()
    return set(MENTION.findall(path.read_text(encoding="utf-8")))


def _active_assignments(path: Path) -> list[str]:
    """Return the uncommented ``KEY=VALUE`` lines of ``path``."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            lines.append(stripped)
    return lines


def _nested_annotation(annotation: object) -> dict[str, Any] | None:
    """Return the fields of ``annotation`` if it is a settings model."""
    model_fields = getattr(annotation, "model_fields", None)
    return model_fields if isinstance(model_fields, dict) else None


def _nested_fields() -> set[str]:
    """Return the names of settings fields that hold another settings model."""
    return {
        name
        for name, field in Settings.model_fields.items()
        if _nested_annotation(field.annotation) is not None
    }


def _schema_keys() -> set[str]:
    """Return every environment variable the settings schema accepts.

    Nested settings are addressed with a double underscore, matching the
    ``env_nested_delimiter`` configured on :class:`Settings`.
    """
    keys = {f"{ENV_PREFIX}{name.upper()}" for name in _scalar_fields()}
    for field_name in _nested_fields():
        nested_fields = _nested_annotation(Settings.model_fields[field_name].annotation) or {}
        for nested in nested_fields:
            keys.add(f"{ENV_PREFIX}{field_name.upper()}__{nested.upper()}")
    return keys


def _scalar_fields() -> set[str]:
    """Return the names of settings fields that map to a single variable."""
    return set(Settings.model_fields) - _nested_fields()


class TestEnvExample:
    """``.env.example`` must document exactly the variables that exist."""

    def test_env_example_exists(self) -> None:
        assert ENV_EXAMPLE.is_file()

    def test_env_example_variables_all_exist_in_the_schema(self) -> None:
        unknown = _documented_keys(ENV_EXAMPLE) - _schema_keys()
        assert not unknown, f"documented but not a real setting: {sorted(unknown)}"

    def test_env_example_values_are_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every uncommented assignment in the file must load successfully."""
        assignments = _active_assignments(ENV_EXAMPLE)
        for assignment in assignments:
            monkeypatch.setenv(*assignment.split("=", 1))
        assert assignments, "expected at least one active assignment"
        Settings()  # must not raise

    def test_env_example_actually_changes_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Applying the example file must not be a silent no-op."""
        for assignment in _active_assignments(ENV_EXAMPLE):
            monkeypatch.setenv(*assignment.split("=", 1))
        assert Settings().model_version == "untrained"
        assert Settings().log.level == "INFO"


class TestReadme:
    """The README advertises environment variables too."""

    def test_readme_exists(self) -> None:
        assert README.is_file()

    def test_readme_variables_all_exist_in_the_schema(self) -> None:
        unknown = _documented_keys(README) - _schema_keys()
        assert not unknown, f"documented but not a real setting: {sorted(unknown)}"

    def test_readme_documents_every_scalar_setting(self) -> None:
        documented = _documented_keys(README)
        missing = {
            f"{ENV_PREFIX}{name.upper()}"
            for name in _scalar_fields()
            if f"{ENV_PREFIX}{name.upper()}" not in documented
        }
        assert not missing, f"settings absent from the README: {sorted(missing)}"

    def test_readme_documents_every_nested_setting(self) -> None:
        documented = _documented_keys(README)
        missing = (
            _schema_keys()
            - documented
            - {f"{ENV_PREFIX}{name.upper()}" for name in _nested_fields()}
        )
        assert not missing, f"nested settings absent from the README: {sorted(missing)}"

    def test_readme_contains_every_required_section(self) -> None:
        content = README.read_text(encoding="utf-8")
        for heading in (
            "## 1. Project purpose",
            "## 2. System architecture",
            "## 3. Planned ML pipeline",
            "## 4. Planned SHAP/LIME explainability",
            "## 5. Planned Streamlit doctor dashboard",
            "## 6. Planned FastAPI backend",
            "## 7. Safety disclaimer",
            "## 8. Local development setup",
        ):
            assert heading in content, f"missing README section: {heading}"

    def test_readme_states_this_is_not_a_diagnosis(self) -> None:
        content = README.read_text(encoding="utf-8").lower()
        assert "not a diagnostic tool" in content
        assert "does not diagnose diabetes" in content

    def test_readme_does_not_advertise_unimplemented_endpoints(self) -> None:
        """Anything presented as runnable must actually exist."""
        content = README.read_text(encoding="utf-8")
        assert "Status: not implemented" in content
        assert not (PROJECT_ROOT / "api" / "main.py").exists()
        assert not (PROJECT_ROOT / "app" / "main.py").exists()


class TestEnumCoverage:
    """Enum members documented as choices must be the real ones."""

    def test_environment_choices(self) -> None:
        assert {member.value for member in Environment} == {"local", "dev", "ci", "prod"}

    def test_log_format_choices(self) -> None:
        assert {member.value for member in LogFormat} == {"text", "json"}
