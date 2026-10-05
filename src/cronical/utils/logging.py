"""Structured logging for the Cronical project.

A single place to obtain configured loggers so that every module logs with the
same format and level, controlled by ``CRONICAL_LOG__*`` settings.

Design notes
------------
* Every logger is a child of the ``cronical`` root logger, so one call to
  :func:`configure_logging` controls application verbosity without touching the
  interpreter-wide root logger.
* Records are written to ``stderr``. ``stdout`` is reserved for data payloads
  and for CLI output that users may want to pipe.
* Both human-readable and JSON-line formats are supported; JSON is intended for
  CI runs and future log aggregation.

Usage
-----
>>> from cronical.utils.logging import get_logger
>>> log = get_logger(__name__)
>>> log.info("pipeline stage complete", extra={"stage": "preprocess"})

Or let the first :func:`get_logger` call configure logging for you::

    from cronical.utils.logging import configure_logging
    configure_logging()
"""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING, Any

from cronical.config import LogFormat, Settings, get_settings

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "JsonFormatter",
    "configure_logging",
    "get_logger",
    "iter_logger_names",
    "owned_handlers",
    "reset_logging",
]

#: Name of the application logger. All project loggers are children of this.
ROOT_LOGGER_NAME = "cronical"

#: Attribute names that are part of the standard ``LogRecord`` contract, and so
#: must not be emitted as custom fields in the JSON payload.
_RESERVED_RECORD_FIELDS: frozenset[str] = frozenset(
    set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"asctime", "message", "taskName"}
)

#: Marker attribute set on handlers this module installs. It lets
#: :func:`configure_logging` tell "already configured by us" apart from "a third
#: party attached something", which would otherwise be silently ignored.
_OWNER_ATTR = "_cronical_owned_handler"


class JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object, one per line.

    Extra fields attached with ``logger.info(..., extra={...})`` are included
    automatically, which makes the output suitable for log aggregation.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Return ``record`` serialised as a JSON object.

        Args:
            record: The record to serialise.

        Returns:
            A JSON document with at least ``timestamp``, ``level``, ``logger``
            and ``message`` keys.
        """
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in _RESERVED_RECORD_FIELDS
            }
        )
        return json.dumps(payload, default=str, ensure_ascii=False)


def _build_handler(settings: Settings) -> logging.Handler:
    """Create the stderr handler described by ``settings``."""
    handler = logging.StreamHandler(stream=sys.stderr)
    log_settings = settings.log
    if log_settings.format is LogFormat.JSON:
        handler.setFormatter(JsonFormatter(datefmt=log_settings.date_format))
    else:
        handler.setFormatter(
            logging.Formatter(fmt=log_settings.text_template, datefmt=log_settings.date_format)
        )
    return handler


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> logging.Logger:
    """Install this module's handler on the ``cronical`` logger.

    Idempotent in the sense that repeated calls do not stack handlers. Idempotency
    is decided by whether *we* installed the handler, not merely by whether the
    logger has any handler at all: a host application or a test runner (pytest
    attaches its own capture handler) may have attached a handler already, and
    that must not be mistaken for a completed configuration. Handlers this module
    does not own are left untouched.

    Args:
        settings: Configuration to apply. Defaults to :func:`get_settings`.
        force: Reconfigure even if this module's handler is already installed.
            Use this after changing settings at runtime.

    Returns:
        The configured ``cronical`` root logger.
    """
    resolved = settings or get_settings()
    logger = logging.getLogger(ROOT_LOGGER_NAME)

    owned = [handler for handler in logger.handlers if getattr(handler, _OWNER_ATTR, False)]
    if owned and not force:
        return logger

    for handler in owned:
        logger.removeHandler(handler)
        handler.close()

    handler = _build_handler(resolved)
    setattr(handler, _OWNER_ATTR, True)
    logger.addHandler(handler)
    logger.setLevel(resolved.log.to_python_level())
    logger.propagate = resolved.log.propagate
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger namespaced under ``cronical``.

    Args:
        name: Usually ``__name__``. Names that are not already namespaced are
            prefixed with ``cronical.`` so configuration stays centralised.

    Returns:
        A ready-to-use :class:`logging.Logger`. Logging is configured on first
        use if this module has not configured it already.
    """
    if not name:
        full_name = ROOT_LOGGER_NAME
    elif name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        full_name = name
    else:
        full_name = f"{ROOT_LOGGER_NAME}.{name}"

    logger = logging.getLogger(full_name)
    if not owned_handlers():
        configure_logging()
    return logger


def owned_handlers() -> list[logging.Handler]:
    """Return the handlers this module has installed on the ``cronical`` logger.

    Empty when logging has not been configured, or when the handler has been
    removed by :func:`reset_logging`. Handy for diagnostics and for asserting in
    tests without being confused by handlers added by a test runner.

    Returns:
        A new list of owned handlers; the returned list is safe to mutate.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    return [handler for handler in logger.handlers if getattr(handler, _OWNER_ATTR, False)]


def reset_logging() -> None:
    """Detach and close every handler installed on the ``cronical`` logger.

    Primarily a test helper; it keeps repeated test runs and ``importlib``
    reloads from stacking duplicate handlers.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def iter_logger_names() -> Iterable[str]:
    """Yield the fully qualified names of live ``cronical`` loggers.

    Note that logger objects outlive :func:`reset_logging`; this reports which
    loggers have been *requested* over the process lifetime, not which ones are
    currently emitting records.

    Useful for diagnostics and for asserting in tests that a module asked for a
    namespaced logger.
    """
    prefix = f"{ROOT_LOGGER_NAME}."
    return (name for name in logging.root.manager.loggerDict if name.startswith(prefix))
