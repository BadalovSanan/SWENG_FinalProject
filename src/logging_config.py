"""Idempotent application logging configuration."""

from __future__ import annotations

import logging

from src.config import Settings, get_settings


_HANDLER_MARKER = "_async_research_assistant_handler"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(settings: Settings | None = None) -> None:
    """Configure the root logger once using the configured application level."""

    active_settings = settings or get_settings()
    root_logger = logging.getLogger()
    root_logger.setLevel(active_settings.log_level)
    if any(getattr(handler, _HANDLER_MARKER, False) for handler in root_logger.handlers):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    setattr(handler, _HANDLER_MARKER, True)
    root_logger.addHandler(handler)
