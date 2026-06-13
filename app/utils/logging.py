"""Project logging helpers."""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(logger_name)s | %(message)s"
_LOG_TIME_FORMAT = "%H:%M:%S"
_DEFAULT_LEVEL = "INFO"
_LOGGER_PREFIX_TO_STRIP = "app.application_layers."


class _SingleLineFormatter(logging.Formatter):
    """Formatter that keeps output compact and single-line."""

    def formatException(self, ei: Any) -> str:
        exc_type, exc, _ = ei
        return f"{exc_type.__name__}: {exc}"

    def format(self, record: logging.LogRecord) -> str:
        raw_name = record.name
        if raw_name.startswith(_LOGGER_PREFIX_TO_STRIP):
            logger_name = raw_name.removeprefix(_LOGGER_PREFIX_TO_STRIP)
        else:
            logger_name = raw_name

        record.__dict__["logger_name"] = logger_name
        formatted = super().format(record)
        return formatted.replace("\n", " | ").replace("\r", " | ")


def _resolve_level(settings: Any | None) -> int:
    if settings is None:
        return logging.INFO

    level_name = _DEFAULT_LEVEL
    app_settings = getattr(settings, "app", None)
    if app_settings is not None:
        configured = getattr(app_settings, "log_level", None)
        if isinstance(configured, str) and configured.strip():
            level_name = configured.strip().upper()

    level_value = getattr(logging, level_name, None)
    if isinstance(level_value, int):
        return level_value
    return logging.INFO


def setup_logging(settings: Any | None = None) -> None:
    """Configure root logging in an idempotent way."""
    level = _resolve_level(settings)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = _SingleLineFormatter(_LOG_FORMAT, datefmt=_LOG_TIME_FORMAT)
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        return

    for existing_handler in root_logger.handlers:
        existing_handler.setFormatter(formatter)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
