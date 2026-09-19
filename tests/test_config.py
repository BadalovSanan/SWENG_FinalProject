"""Tests for environment-backed application settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import get_settings


SETTINGS_ENV_VARS = (
    "LOG_LEVEL",
    "CACHE_DIR",
    "CACHE_TTL_SECONDS",
    "PER_SOURCE_TIMEOUT_SECONDS",
    "MAX_SOURCES_PER_QUERY",
)


@pytest.fixture(autouse=True)
def isolate_settings_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Prevent tests from reading a developer's environment or .env file."""

    get_settings.cache_clear()
    monkeypatch.chdir(tmp_path)
    for variable in SETTINGS_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)
    yield
    get_settings.cache_clear()


def test_defaults_load_correctly() -> None:
    """Defaults match the checked-in environment example."""

    settings = get_settings()

    assert settings.log_level == "INFO"
    assert settings.cache_dir == Path(".cache")
    assert settings.cache_ttl_seconds == 86_400
    assert settings.per_source_timeout_seconds == 10
    assert settings.max_sources_per_query == 3


def test_environment_overrides_are_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured environment variables override defaults."""

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("CACHE_DIR", "custom-cache")
    monkeypatch.setenv("CACHE_TTL_SECONDS", "60")
    monkeypatch.setenv("PER_SOURCE_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("MAX_SOURCES_PER_QUERY", "5")

    settings = get_settings()

    assert settings.log_level == "DEBUG"
    assert settings.cache_dir == Path("custom-cache")
    assert settings.cache_ttl_seconds == 60
    assert settings.per_source_timeout_seconds == 15
    assert settings.max_sources_per_query == 5


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("CACHE_TTL_SECONDS", "0"),
        ("CACHE_TTL_SECONDS", "-1"),
        ("CACHE_TTL_SECONDS", "not-a-number"),
        ("PER_SOURCE_TIMEOUT_SECONDS", "0"),
        ("PER_SOURCE_TIMEOUT_SECONDS", "-1"),
        ("PER_SOURCE_TIMEOUT_SECONDS", "not-a-number"),
        ("MAX_SOURCES_PER_QUERY", "0"),
        ("MAX_SOURCES_PER_QUERY", "-1"),
        ("MAX_SOURCES_PER_QUERY", "not-a-number"),
    ],
)
def test_invalid_positive_integer_settings_raise_validation_error(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    """Zero, negative, and non-numeric numeric settings are rejected."""

    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        get_settings()


def test_unrelated_environment_variables_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider credentials and other unrelated variables do not break settings."""

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-value")

    settings = get_settings()

    assert settings.log_level == "INFO"


def test_log_level_is_normalized_to_uppercase() -> None:
    """Lowercase log levels are normalized for the logging module."""

    from src.config import Settings

    assert Settings(log_level="debug").log_level == "DEBUG"


def test_invalid_log_level_is_rejected() -> None:
    """Only standard logging levels are accepted."""

    from src.config import Settings

    with pytest.raises(ValidationError):
        Settings(log_level="BANANA")
