"""Filesystem-backed, TTL-aware source cache storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import tempfile
import time

from pydantic import ValidationError

from ai.schemas import Source
from src.exceptions import CacheError
from src.models import CacheKey, SourceName


logger = logging.getLogger(__name__)


class CacheStore(ABC):
    """Cache interface Person 2 codes against."""

    @abstractmethod
    def get(self, source: SourceName, query: str) -> list[Source] | None:
        """Return cached sources for a source-query pair, or None on a miss."""

    @abstractmethod
    def set(self, source: SourceName, query: str, value: list[Source]) -> None:
        """Store sources for a source-query pair."""

    @abstractmethod
    def delete(self, source: SourceName, query: str) -> None:
        """Remove any cached sources for a source-query pair."""


class FileSystemCacheStore(CacheStore):
    """Persist TTL-aware source results as one JSON file per cache key."""

    def __init__(
        self,
        cache_dir: Path,
        ttl_seconds: int,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Create a cache store without creating its directory yet."""

        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        self._cache_dir = cache_dir
        self._ttl_seconds = ttl_seconds
        self._clock = clock

    def get(self, source: SourceName, query: str) -> list[Source] | None:
        """Return a cached value, treating unreadable or invalid entries as misses."""

        key = CacheKey.from_raw(source, query)
        path = self._path_for(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.debug("Cache miss for %s", path.name)
            return None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
            self._discard_corrupt_entry(path, error)
            return None

        try:
            stored_at = payload["stored_at"]
            stored_source = payload["source"]
            stored_query = payload["query"]
            raw_sources = payload["sources"]
            if (
                isinstance(stored_at, bool)
                or not isinstance(stored_at, (int, float))
                or stored_source != key.source.value
                or stored_query != key.query
                or not isinstance(raw_sources, list)
            ):
                raise TypeError("cache entry has an invalid structure")

            if self._clock() - stored_at >= self._ttl_seconds:
                self._remove_ignoring_errors(path)
                logger.debug("Cache entry expired for %s", path.name)
                return None

            sources = [Source.model_validate(raw_source) for raw_source in raw_sources]
        except (KeyError, TypeError, ValidationError) as error:
            self._discard_corrupt_entry(path, error)
            return None

        logger.debug("Cache hit for %s", path.name)
        return sources

    def set(self, source: SourceName, query: str, value: list[Source]) -> None:
        """Atomically store sources for a source-query pair."""

        key = CacheKey.from_raw(source, query)
        path = self._path_for(key)
        temporary_path: Path | None = None
        payload = {
            "stored_at": self._clock(),
            "source": key.source.value,
            "query": key.query,
            "sources": [item.model_dump(mode="json") for item in value],
        }

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._cache_dir,
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(payload, temporary_file)
            os.replace(temporary_path, path)
        except OSError as error:
            if temporary_path is not None:
                self._remove_ignoring_errors(temporary_path)
            raise CacheError("Unable to write cache entry.") from error

        logger.debug("Wrote cache entry %s", path.name)

    def delete(self, source: SourceName, query: str) -> None:
        """Delete a cache entry if present."""

        path = self._path_for(CacheKey.from_raw(source, query))
        try:
            path.unlink()
        except FileNotFoundError:
            logger.debug("Cache entry already absent for %s", path.name)
        except OSError as error:
            raise CacheError("Unable to delete cache entry.") from error
        else:
            logger.debug("Deleted cache entry %s", path.name)

    def _path_for(self, key: CacheKey) -> Path:
        """Build the safe, deterministic filename for a canonical cache key."""

        digest_input = f"{key.source.value}:{key.query}".encode("utf-8")
        return self._cache_dir / f"{sha256(digest_input).hexdigest()}.json"

    def _discard_corrupt_entry(self, path: Path, error: Exception) -> None:
        """Log an invalid entry and attempt to remove it without raising."""

        logger.warning("Ignoring invalid cache entry %s: %s", path.name, type(error).__name__)
        self._remove_ignoring_errors(path)

    @staticmethod
    def _remove_ignoring_errors(path: Path) -> None:
        """Remove a stale or invalid file without allowing cleanup to fail reads."""

        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            return
