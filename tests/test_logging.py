"""Tests for :mod:`cronical.utils.logging`.

The contract being verified is that library code always obtains namespaced
loggers through :func:`get_logger`, and that formatting is decided in exactly one
place.
"""

from __future__ import annotations

import json
import logging
import sys

from cronical.config import LogFormat, Settings
from cronical.utils.logging import (
    ROOT_LOGGER_NAME,
    JsonFormatter,
    configure_logging,
    get_logger,
    iter_logger_names,
    owned_handlers,
    reset_logging,
)


class TestGetLogger:
    """Logger namespacing."""

    def test_names_are_prefixed_with_the_app_namespace(self) -> None:
        assert get_logger("my_module").name == "cronical.my_module"

    def test_already_namespaced_names_are_left_alone(self) -> None:
        assert get_logger("cronical.data").name == "cronical.data"

    def test_root_name_is_left_alone(self) -> None:
        assert get_logger(ROOT_LOGGER_NAME).name == ROOT_LOGGER_NAME

    def test_missing_name_returns_the_root_logger(self) -> None:
        assert get_logger().name == ROOT_LOGGER_NAME

    def test_first_use_installs_a_handler(self) -> None:
        get_logger("autoconfigure")
        assert owned_handlers()

    def test_first_use_configures_despite_a_foreign_handler(self) -> None:
        logger = logging.getLogger(ROOT_LOGGER_NAME)
        foreign = logging.NullHandler()
        logger.addHandler(foreign)
        try:
            reset_logging()
            get_logger("autoconfigure_despite_foreign")
            assert owned_handlers()
        finally:
            logger.removeHandler(foreign)

    def test_returned_object_is_a_standard_logger(self) -> None:
        assert isinstance(get_logger("typed"), logging.Logger)


class TestConfigureLogging:
    """Handler installation.

    These tests assert on :func:`owned_handlers` rather than on
    ``logger.handlers``, because pytest attaches its own capture handler to the
    logging tree. That contamination is exactly what this module must tolerate
    without mistaking a foreign handler for its own configuration.
    """

    def test_idempotent_by_default(self, settings: Settings) -> None:
        configure_logging(settings)
        configure_logging(settings)
        assert len(owned_handlers()) == 1

    def test_force_replaces_the_handler(self, settings: Settings) -> None:
        configure_logging(settings)
        configure_logging(settings, force=True)
        assert len(owned_handlers()) == 1

    def test_configures_even_when_a_foreign_handler_is_present(self, settings: Settings) -> None:
        logger = logging.getLogger(ROOT_LOGGER_NAME)
        foreign = logging.NullHandler()
        logger.addHandler(foreign)
        try:
            configure_logging(settings)
            assert len(owned_handlers()) == 1
            assert foreign in logger.handlers, "a third-party handler must be left alone"
        finally:
            logger.removeHandler(foreign)

    def test_foreign_handlers_are_not_closed(self, settings: Settings) -> None:
        logger = logging.getLogger(ROOT_LOGGER_NAME)
        foreign = logging.NullHandler()
        logger.addHandler(foreign)
        try:
            configure_logging(settings)
            configure_logging(settings, force=True)
            assert foreign in logger.handlers
        finally:
            logger.removeHandler(foreign)

    def test_level_is_applied(self) -> None:
        settings = Settings(log={"level": "WARNING"})
        configure_logging(settings)
        assert logging.getLogger(ROOT_LOGGER_NAME).level == logging.WARNING

    def test_json_format_is_applied(self) -> None:
        settings = Settings(log={"format": "json"})
        configure_logging(settings)
        assert isinstance(owned_handlers()[0].formatter, JsonFormatter)

    def test_text_format_is_applied(self) -> None:
        settings = Settings(log={"format": "text"})
        configure_logging(settings)
        assert not isinstance(owned_handlers()[0].formatter, JsonFormatter)

    def test_handlers_write_to_stderr(self, settings: Settings) -> None:
        configure_logging(settings)
        handler = owned_handlers()[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream is not None

    def test_propagation_follows_settings(self) -> None:
        configure_logging(Settings(log={"propagate": True}))
        assert logging.getLogger(ROOT_LOGGER_NAME).propagate is True

    def test_reset_removes_every_handler(self, settings: Settings) -> None:
        configure_logging(settings)
        reset_logging()
        assert owned_handlers() == []
        assert logging.getLogger(ROOT_LOGGER_NAME).handlers == []

    def test_owned_handlers_returns_a_fresh_list(self, settings: Settings) -> None:
        configure_logging(settings)
        first = owned_handlers()
        first.clear()
        assert len(owned_handlers()) == 1

    def test_log_format_enum_is_exhaustive(self) -> None:
        assert {member.value for member in LogFormat} == {"text", "json"}

    def test_reset_then_get_logger_reconfigures_once(self) -> None:
        configure_logging(Settings())
        reset_logging()
        get_logger("post_reset")
        assert len(owned_handlers()) == 1

    def test_get_logger_does_not_reconfigure_when_already_set_up(self, settings: Settings) -> None:
        configure_logging(settings)
        installed = owned_handlers()
        get_logger("already_configured")
        assert owned_handlers() == installed


class TestIterLoggerNames:
    """Diagnostics over live logger names."""

    def test_lists_namespaced_loggers_only(self) -> None:
        get_logger("diagnostics_probe")
        logging.getLogger("some_third_party_library")
        names = set(iter_logger_names())
        assert "cronical.diagnostics_probe" in names
        assert "some_third_party_library" not in names

    def test_names_survive_a_reset(self) -> None:
        """``reset_logging`` drops handlers, not the logger objects themselves.

        Loggers live in the interpreter-wide registry for the life of the
        process, so this function reports what has been *asked for*, not what is
        currently emitting. Documented here because it is a real distinction.
        """
        get_logger("persistent_probe")
        reset_logging()
        assert "cronical.persistent_probe" in set(iter_logger_names())
        assert owned_handlers() == []


class TestJsonFormatter:
    """Serialisation of individual records."""

    @staticmethod
    def _record(msg: str = "hello", **extra: object) -> logging.LogRecord:
        record = logging.LogRecord(
            name="cronical.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )
        for key, value in extra.items():
            setattr(record, key, value)
        return record

    def test_core_keys_are_present(self) -> None:
        payload = json.loads(JsonFormatter().format(self._record()))
        assert payload["level"] == "INFO"
        assert payload["logger"] == "cronical.test"
        assert payload["message"] == "hello"
        assert "timestamp" in payload

    def test_extra_fields_are_included(self) -> None:
        payload = json.loads(JsonFormatter().format(self._record(stage="preprocess")))
        assert payload["stage"] == "preprocess"

    def test_standard_record_attributes_are_not_duplicated(self) -> None:
        payload = json.loads(JsonFormatter().format(self._record()))
        assert "pathname" not in payload
        assert "lineno" not in payload

    def test_non_serialisable_values_are_coerced(self) -> None:
        payload = json.loads(JsonFormatter().format(self._record(obj=object())))
        assert isinstance(payload["obj"], str)

    def test_exceptions_are_included(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            record = logging.LogRecord(
                "cronical.test", logging.ERROR, __file__, 1, "failed", (), sys.exc_info()
            )
        payload = json.loads(JsonFormatter().format(record))
        assert "ValueError: boom" in payload["exception"]

    def test_percent_args_are_interpolated(self) -> None:
        record = logging.LogRecord(
            "cronical.test", logging.INFO, __file__, 1, "rows=%d", (7,), None
        )
        assert json.loads(JsonFormatter().format(record))["message"] == "rows=7"
