"""Tests for :mod:`cronical.config`.

These tests pin the two guarantees the rest of the project relies on: every path
is derived from ``project_root``, and configuration is fully overridable through
the environment without touching code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cronical.config import (
    ENV_PREFIX,
    PROJECT_ROOT,
    Environment,
    LogFormat,
    LogSettings,
    Paths,
    Settings,
    get_settings,
    reload_settings,
)


class TestPaths:
    """The derived directory layout."""

    def test_default_project_root_is_absolute(self) -> None:
        assert PROJECT_ROOT.is_absolute()
        assert PROJECT_ROOT.name == "cronical"

    def test_project_root_from_string_is_expanded(self) -> None:
        assert Paths(project_root="~/demo").project_root.is_absolute()

    def test_project_root_from_path_is_expanded(self) -> None:
        assert Paths(project_root=Path("~/demo")).project_root.is_absolute()

    def test_project_root_rejects_other_types(self) -> None:
        with pytest.raises(ValidationError):
            Paths(project_root=42)

    def test_project_root_rejects_none(self) -> None:
        with pytest.raises(ValidationError):
            Paths(project_root=None)

    @pytest.mark.parametrize(
        ("attribute", "suffix"),
        [
            ("raw_data_dir", ("data", "raw")),
            ("processed_data_dir", ("data", "processed")),
            ("data_dir", ("data",)),
            ("notebooks_dir", ("notebooks",)),
            ("models_dir", ("models",)),
            ("figures_dir", ("reports", "figures")),
            ("reports_dir", ("reports",)),
            ("api_dir", ("api",)),
            ("app_dir", ("app",)),
            ("tests_dir", ("tests",)),
            ("package_dir", ("src", "cronical")),
        ],
    )
    def test_directories_derive_from_root(self, attribute: str, suffix: tuple[str, ...]) -> None:
        paths = Paths(project_root=Path("/tmp/demo"))
        resolved = getattr(paths, attribute)
        assert resolved == Path("/tmp/demo").joinpath(*suffix)

    def test_as_dict_is_stringly_typed(self) -> None:
        payload = Paths(project_root=Path("/tmp/demo")).as_dict()
        assert all(isinstance(value, str) for value in payload.values())
        assert json.dumps(payload)

    def test_ensure_directories_is_idempotent(self, tmp_path: Path) -> None:
        paths = Paths(project_root=tmp_path)
        first = paths.ensure_directories()
        second = paths.ensure_directories()

        assert first, "expected the first call to create directories"
        assert second == (), "the second call must be a no-op"
        assert paths.raw_data_dir.is_dir()
        assert paths.figures_dir.is_dir()

    def test_paths_are_immutable(self) -> None:
        # mypy correctly flags this as a read-only property; the point of the
        # test is that the *runtime* also refuses the assignment.
        with pytest.raises(ValidationError):
            Paths(project_root=Path("/tmp/demo")).project_root = Path(  # type: ignore[misc]
                "/tmp/other"
            )


class TestSettings:
    """Top-level settings and their environment overrides."""

    def test_defaults_are_honest_about_model_state(self) -> None:
        settings = Settings()
        assert settings.model_version == "untrained"
        assert settings.has_trained_model is False

    def test_model_path_uses_artifact_name(self, isolated_project_root: Path) -> None:
        settings = Settings.for_root(isolated_project_root, model_artifact_name="custom.joblib")
        assert settings.model_path == isolated_project_root / "models" / "custom.joblib"

    def test_has_trained_model_requires_artifact_on_disk(self, isolated_project_root: Path) -> None:
        settings = Settings.for_root(isolated_project_root, model_version="v1")
        assert settings.has_trained_model is False

        settings.paths.models_dir.mkdir(parents=True)
        settings.model_path.write_bytes(b"placeholder")
        assert settings.has_trained_model is True

    def test_paths_follow_overridden_root(self, isolated_project_root: Path) -> None:
        settings = Settings.for_root(isolated_project_root)
        assert settings.paths.project_root == isolated_project_root
        assert settings.paths.raw_data_dir == isolated_project_root / "data" / "raw"

    def test_keyword_arguments_beat_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(f"{ENV_PREFIX}APP_NAME", "from-env")
        assert Settings(app_name="explicit").app_name == "explicit"

    def test_project_root_accepts_a_string(self) -> None:
        assert Settings(project_root="~/demo").project_root.is_absolute()

    def test_project_root_rejects_other_types(self) -> None:
        with pytest.raises(ValidationError):
            Settings(project_root=42)

    def test_flat_environment_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(f"{ENV_PREFIX}DEBUG", "true")
        monkeypatch.setenv(f"{ENV_PREFIX}API_PORT", "9000")
        settings = reload_settings()
        assert settings.debug is True
        assert settings.api_port == 9000

    def test_nested_environment_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(f"{ENV_PREFIX}LOG__LEVEL", "debug")
        monkeypatch.setenv(f"{ENV_PREFIX}LOG__FORMAT", "JSON")
        log = reload_settings().log
        assert log.level == "DEBUG"
        assert log.format is LogFormat.JSON

    def test_unknown_environment_variables_are_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(f"{ENV_PREFIX}TOTALLY_UNKNOWN", "whatever")
        assert reload_settings() is not None

    def test_invalid_value_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(f"{ENV_PREFIX}API_PORT", "not-a-port")
        with pytest.raises(ValidationError):
            reload_settings()

    def test_environment_enum(self) -> None:
        assert Settings(environment="ci").environment is Environment.CI
        assert Settings(environment="ci").is_local is False

    def test_for_root_does_not_read_the_repository_dotenv(
        self, isolated_project_root: Path
    ) -> None:
        (isolated_project_root / ".env").write_text(
            f"{ENV_PREFIX}APP_NAME=from-isolated-dotenv\n", encoding="utf-8"
        )
        settings = Settings.for_root(isolated_project_root)
        assert settings.app_name == "from-isolated-dotenv"

    def test_settings_are_immutable(self) -> None:
        with pytest.raises(ValidationError):
            Settings().app_name = "mutated"  # type: ignore[misc]


class TestLogSettings:
    """Normalisation of the nested logging configuration."""

    @pytest.mark.parametrize(
        ("raw", "expected"), [("debug", "DEBUG"), (" Info ", "INFO"), ("warning", "WARNING")]
    )
    def test_level_is_normalised(self, raw: str, expected: str) -> None:
        assert LogSettings(level=raw).level == expected

    @pytest.mark.parametrize(
        ("raw", "expected"), [("JSON", LogFormat.JSON), ("text", LogFormat.TEXT)]
    )
    def test_format_is_normalised(self, raw: str, expected: LogFormat) -> None:
        assert LogSettings(format=raw).format is expected

    def test_to_python_level_maps_to_logging_constants(self) -> None:
        import logging

        assert LogSettings(level="WARNING").to_python_level() == logging.WARNING
        assert LogSettings(level="DEBUG").to_python_level() == logging.DEBUG


class TestSettingsCache:
    """The process-wide settings singleton."""

    def test_get_settings_is_cached(self) -> None:
        assert get_settings() is get_settings()

    def test_reload_returns_a_fresh_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        before = get_settings()
        monkeypatch.setenv(f"{ENV_PREFIX}APP_NAME", "reloaded")
        after = reload_settings()
        assert after is not before
        assert after.app_name == "reloaded"


class TestDescribe:
    """Snapshotting configuration for logs and API responses."""

    def test_describe_is_json_serialisable(self, isolated_project_root: Path) -> None:
        payload = Settings.for_root(isolated_project_root).describe()
        assert json.loads(json.dumps(payload))["paths"]["raw_data_dir"]
        assert payload["project_root"] == str(isolated_project_root)

    def test_describe_redacts_sensitive_fields(
        self, isolated_project_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Settings, "SENSITIVE_FIELDS", frozenset({"app_name"}))
        payload = Settings.for_root(isolated_project_root).describe()
        assert payload["app_name"] == Settings.REDACTED

    def test_redaction_is_a_no_op_when_no_secrets_are_declared(
        self, isolated_project_root: Path
    ) -> None:
        assert frozenset() == Settings.SENSITIVE_FIELDS
        payload = Settings.for_root(isolated_project_root).describe()
        assert payload["app_name"] == "cronical"

    def test_declared_but_absent_fields_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            Settings, "SENSITIVE_FIELDS", frozenset({"a_field_that_does_not_exist"})
        )
        payload = Settings().describe()
        assert "a_field_that_does_not_exist" not in payload
