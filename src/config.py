"""Application configuration loaded from the environment or a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated settings for the software-engineering layer."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = "INFO"
    cache_dir: Path = Path("./.cache")
    cache_ttl_seconds: int = Field(default=86_400, gt=0)
    per_source_timeout_seconds: int = Field(default=10, gt=0)
    max_sources_per_query: int = Field(default=3, gt=0)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Normalize and validate the configured logging level."""

        normalized_value = value.strip().upper()
        if normalized_value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("log_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        return normalized_value


@lru_cache
def get_settings() -> Settings:
    """Return the cached, validated application settings."""

    return Settings()
