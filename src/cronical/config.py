"""Environment-driven configuration and canonical filesystem layout.

Every directory in this project is derived from :attr:`Settings.project_root` and
surfaced as a :class:`pathlib.Path`. No module outside this one should ever
hard-code a path.

Resolution order
----------------
Highest priority first:

1. Keyword arguments passed to :class:`Settings` (used heavily by tests).
2. Process environment variables prefixed with ``CRONICAL_``.
3. A ``.env`` file at the project root (see :data:`ENV_FILE`).
4. The defaults declared on :class:`Settings`.

Nested settings use a double underscore, so ``LogSettings.level`` is configured
with ``CRONICAL_LOG__LEVEL=DEBUG``.

Example:
-------
>>> from cronical.config import get_settings
>>> settings = get_settings()
>>> settings.paths.models_dir.name
'models'
"""

from __future__ import annotations

import logging
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "ENV_FILE",
    "ENV_PREFIX",
    "PACKAGE_DIR",
    "PROJECT_ROOT",
    "Environment",
    "LogFormat",
    "LogSettings",
    "Paths",
    "Settings",
    "get_settings",
    "reload_settings",
]

#: Directory containing the ``cronical`` package.
PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent

#: The ``src`` directory that acts as the import root for the package.
SRC_DIR: Final[Path] = PACKAGE_DIR.parent

#: Repository root, resolved from this file's location rather than the working
#: directory so that tooling works regardless of where it is invoked from.
PROJECT_ROOT: Final[Path] = SRC_DIR.parent

#: Prefix for every environment variable understood by this project.
ENV_PREFIX: Final[str] = "CRONICAL_"

#: Optional local overrides file. Never commit a real ``.env``.
ENV_FILE: Final[Path] = PROJECT_ROOT / ".env"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Environment(StrEnum):
    """Deployment environment for the current process."""

    LOCAL = "local"
    DEV = "dev"
    CI = "ci"
    PROD = "prod"


class LogFormat(StrEnum):
    """Rendering style for log records."""

    TEXT = "text"
    JSON = "json"


class LogSettings(BaseModel):
    """Logging behaviour, configured through ``CRONICAL_LOG__*`` variables."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: LogLevel = "INFO"
    format: LogFormat = LogFormat.TEXT
    date_format: str = "%Y-%m-%dT%H:%M:%S%z"
    text_template: str = "%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s"
    propagate: bool = False

    @field_validator("level", mode="before")
    @classmethod
    def _normalise_level(cls, value: object) -> object:
        """Upper-case the level so ``debug`` and ``DEBUG`` behave identically."""
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("format", mode="before")
    @classmethod
    def _normalise_format(cls, value: object) -> object:
        """Lower-case the format so ``JSON`` and ``json`` behave identically."""
        return value.strip().lower() if isinstance(value, str) else value

    def to_python_level(self) -> int:
        """Return the equivalent :mod:`logging` numeric level."""
        return int(getattr(logging, self.level, logging.INFO))


class Paths(BaseModel):
    """Canonical project layout, derived entirely from ``project_root``.

    The directories are exposed as read-only properties rather than stored
    fields so they can never drift out of sync with the project root.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_root: Path = Field(default=PROJECT_ROOT, description="Repository root.")

    @field_validator("project_root", mode="before")
    @classmethod
    def _coerce_root(cls, value: object) -> object:
        """Expand ``~`` and relative values into absolute paths."""
        if isinstance(value, str):
            return Path(value).expanduser()
        if isinstance(value, Path):
            return value.expanduser()
        return value

    @property
    def src_dir(self) -> Path:
        """Import root that contains the ``cronical`` package."""
        return self.project_root / "src"

    @property
    def package_dir(self) -> Path:
        """Directory of the installed ``cronical`` package."""
        return self.src_dir / "cronical"

    @property
    def data_dir(self) -> Path:
        """Root of all dataset directories."""
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        """Immutable, as-downloaded source data. Never edited in place."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Deterministic outputs of the preprocessing pipeline."""
        return self.data_dir / "processed"

    @property
    def notebooks_dir(self) -> Path:
        """Exploration and training notebooks."""
        return self.project_root / "notebooks"

    @property
    def models_dir(self) -> Path:
        """Persisted, versioned model artifacts."""
        return self.project_root / "models"

    @property
    def reports_dir(self) -> Path:
        """Generated evaluation reports."""
        return self.project_root / "reports"

    @property
    def figures_dir(self) -> Path:
        """Images and plots referenced by reports and the dashboard."""
        return self.reports_dir / "figures"

    @property
    def api_dir(self) -> Path:
        """FastAPI backend package."""
        return self.project_root / "api"

    @property
    def app_dir(self) -> Path:
        """Streamlit dashboard entry points."""
        return self.project_root / "app"

    @property
    def tests_dir(self) -> Path:
        """Automated test suite."""
        return self.project_root / "tests"

    @property
    def env_file(self) -> Path:
        """Local, uncommitted environment overrides."""
        return self.project_root / ".env"

    @property
    def managed_dirs(self) -> tuple[Path, ...]:
        """Directories that :meth:`ensure_directories` is allowed to create."""
        return (
            self.raw_data_dir,
            self.processed_data_dir,
            self.notebooks_dir,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
        )

    def as_dict(self) -> dict[str, str]:
        """Return the layout as plain strings, for logging and API responses."""
        return {
            "project_root": str(self.project_root),
            "src_dir": str(self.src_dir),
            "package_dir": str(self.package_dir),
            "data_dir": str(self.data_dir),
            "raw_data_dir": str(self.raw_data_dir),
            "processed_data_dir": str(self.processed_data_dir),
            "notebooks_dir": str(self.notebooks_dir),
            "models_dir": str(self.models_dir),
            "reports_dir": str(self.reports_dir),
            "figures_dir": str(self.figures_dir),
            "api_dir": str(self.api_dir),
            "app_dir": str(self.app_dir),
            "tests_dir": str(self.tests_dir),
            "env_file": str(self.env_file),
        }

    def ensure_directories(self) -> tuple[Path, ...]:
        """Create every managed directory and return those newly created.

        Returns:
            The directories that did not previously exist.
        """
        created: list[Path] = []
        for directory in self.managed_dirs:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created.append(directory)
        return tuple(created)


class Settings(BaseSettings):
    """Top-level application settings.

    Examples:
    --------
    >>> Settings(project_root="/tmp/demo").paths.raw_data_dir.as_posix()
    '/tmp/demo/data/raw'
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    app_name: str = "cronical"
    environment: Environment = Environment.LOCAL
    debug: bool = False
    project_root: Path = Field(
        default=PROJECT_ROOT,
        description="Repository root; every other path is derived from it.",
    )

    random_seed: int = Field(default=42, ge=0, description="Seed for reproducibility.")
    model_artifact_name: str = "diabetes_risk_model.joblib"
    model_version: str = Field(
        default="untrained",
        description=(
            "Identifier of the loaded artifact. 'untrained' means no model has been "
            "fitted yet, which is the expected state for a fresh checkout."
        ),
    )

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_reload: bool = True
    streamlit_port: int = Field(default=8501, ge=1, le=65535)

    log: LogSettings = Field(default_factory=LogSettings)

    #: Field names whose values must never appear in logs or API responses.
    #: Currently empty because the project stores no credentials; add a name
    #: here the moment a secret-backed integration is introduced.
    SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset()

    #: Placeholder substituted for redacted values.
    REDACTED: ClassVar[str] = "***redacted***"

    @field_validator("project_root", mode="before")
    @classmethod
    def _coerce_project_root(cls, value: object) -> object:
        """Expand ``~`` and relative values into absolute paths."""
        if isinstance(value, str):
            return Path(value).expanduser()
        if isinstance(value, Path):
            return value.expanduser()
        return value

    @property
    def paths(self) -> Paths:
        """Filesystem layout derived from the current project root."""
        return Paths(project_root=self.project_root)

    @property
    def model_path(self) -> Path:
        """Expected location of the persisted model artifact."""
        return self.paths.models_dir / self.model_artifact_name

    @property
    def has_trained_model(self) -> bool:
        """Whether a trained artifact is expected to exist on disk."""
        return self.model_version != "untrained" and self.model_path.is_file()

    @property
    def is_local(self) -> bool:
        """Whether this is an interactive developer machine."""
        return self.environment is Environment.LOCAL

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the configuration.

        Values listed in :attr:`SENSITIVE_FIELDS` are replaced with
        :attr:`REDACTED`, so the result is safe to log or return from an API.
        """
        payload: dict[str, Any] = self.model_dump(mode="json")
        for name in self.SENSITIVE_FIELDS:
            if name in payload:
                payload[name] = self.REDACTED
        payload["paths"] = self.paths.as_dict()
        return payload

    @classmethod
    def for_root(cls, project_root: Path | str, **overrides: Any) -> Settings:
        """Build settings rooted at ``project_root``, including its ``.env``.

        Useful for tests and for containerised runs where the code and the data
        live under different mount points.
        """
        root = Path(project_root).expanduser()
        return cls(_env_file=root / ".env", project_root=root, **overrides)


@lru_cache(maxsize=1)
def get_settings(**overrides: Any) -> Settings:
    """Return the process-wide settings singleton.

    Args:
        **overrides: Field values that take precedence over env and ``.env``.

    Returns:
        A cached :class:`Settings` instance.
    """
    return Settings(**overrides)


def reload_settings(**overrides: Any) -> Settings:
    """Clear the cache and rebuild settings, honouring the current environment.

    Intended for tests that manipulate environment variables.
    """
    get_settings.cache_clear()
    return get_settings(**overrides)
