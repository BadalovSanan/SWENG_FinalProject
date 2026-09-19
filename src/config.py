"""Application configuration loaded from the environment or a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
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


@lru_cache
def get_settings() -> Settings:
    """Return the cached, validated application settings."""

    return Settings()
