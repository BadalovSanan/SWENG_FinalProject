"""Tests for the filesystem-backed source cache."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from ai.schemas import Source
from src.exceptions import CacheError
from src.models import CacheKey, SourceName
from src.storage.cache_store import CacheStore, FileSystemCacheStore


class FakeClock:
    """A controllable clock for TTL tests."""

    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        """Return the current fake time."""

        return self.now

    def advance(self, seconds: float) -> None:
        """Move the fake time forward by the requested number of seconds."""

        self.now += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    """Provide a clock that tests can advance without sleeping."""

    return FakeClock()


@pytest.fixture
def cache_store(tmp_path: Path, fake_clock: FakeClock) -> FileSystemCacheStore:
    """Provide an empty cache store with a five-second TTL."""

    return FileSystemCacheStore(tmp_path / "cache", ttl_seconds=5, clock=fake_clock)


@pytest.fixture
def source_value() -> list[Source]:
    """Provide realistic sources accepted by the supplied schema."""

    return [
        Source(
            title="Photosynthesis",
            url="https://en.wikipedia.org/wiki/Photosynthesis",
            snippet="Plants convert light energy into chemical energy.",
            origin="wikipedia",
        )
    ]


def test_write_then_read_returns_equal_sources(
    cache_store: FileSystemCacheStore, source_value: list[Source]
) -> None:
    """Stored Pydantic source data is reconstructed without changes."""

    cache_store.set(SourceName.WIKI, "What is photosynthesis?", source_value)

    assert cache_store.get(SourceName.WIKI, "What is photosynthesis?") == source_value


def test_get_returns_none_when_nothing_is_stored(cache_store: FileSystemCacheStore) -> None:
    """A missing entry is a cache miss."""

    assert cache_store.get(SourceName.WEB, "missing") is None


def test_ttl_expires_and_removes_stale_entry(
    cache_store: FileSystemCacheStore,
    fake_clock: FakeClock,
    source_value: list[Source],
) -> None:
    """Entries are available before expiry and removed after expiry."""

    cache_store.set(SourceName.WIKI, "Q", source_value)
    path = cache_store._path_for(cache_store_key(SourceName.WIKI, "Q"))
    fake_clock.advance(4.9)

    assert cache_store.get(SourceName.WIKI, "Q") == source_value

    fake_clock.advance(0.2)

    assert cache_store.get(SourceName.WIKI, "Q") is None
    assert not path.exists()


def test_corrupt_entries_are_cache_misses_with_warning(
    cache_store: FileSystemCacheStore, caplog: pytest.LogCaptureFixture
) -> None:
    """Garbage and malformed entries never escape as read errors."""

    path = cache_store._path_for(cache_store_key(SourceName.WEB, "Q"))
    path.parent.mkdir()
    caplog.set_level(logging.WARNING, logger="src.storage.cache_store")

    path.write_text("not json", encoding="utf-8")
    assert cache_store.get(SourceName.WEB, "Q") is None

    path.write_text(json.dumps({"wrong": "shape"}), encoding="utf-8")
    assert cache_store.get(SourceName.WEB, "Q") is None

    path.write_text(
        json.dumps(
            {
                "stored_at": 100.0,
                "source": "web",
                "query": "q",
                "sources": [
                    {
                        "title": "Invalid origin",
                        "url": "https://example.com",
                        "snippet": "Invalid source data.",
                        "origin": "reddit",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert cache_store.get(SourceName.WEB, "Q") is None
    assert len(caplog.records) == 3
    assert all(record.levelno == logging.WARNING for record in caplog.records)


def test_equivalent_queries_share_a_cache_entry(
    cache_store: FileSystemCacheStore, source_value: list[Source]
) -> None:
    """The store relies on the shared query canonicalization function."""

    cache_store.set(SourceName.WIKI, "What is Photosynthesis?", source_value)

    assert cache_store.get(SourceName.WIKI, "  what is   photosynthesis ") == source_value


def test_different_sources_use_separate_entries(
    cache_store: FileSystemCacheStore, source_value: list[Source]
) -> None:
    """The source name participates in cache-key identity."""

    web_source = Source(
        title="Photosynthesis article",
        url="https://example.com/photosynthesis",
        snippet="A web result.",
        origin="web",
    )
    cache_store.set(SourceName.WIKI, "Q", source_value)
    cache_store.set(SourceName.WEB, "Q", [web_source])

    assert cache_store.get(SourceName.WIKI, "Q") == source_value
    assert cache_store.get(SourceName.WEB, "Q") == [web_source]


def test_empty_list_round_trips(cache_store: FileSystemCacheStore) -> None:
    """An empty source result is distinct from a cache miss."""

    cache_store.set(SourceName.ARXIV, "Q", [])

    assert cache_store.get(SourceName.ARXIV, "Q") == []


def test_second_write_overwrites_first(
    cache_store: FileSystemCacheStore, source_value: list[Source]
) -> None:
    """A newer value replaces the earlier one for the same key."""

    replacement = Source(
        title="Replacement",
        url="https://example.com/replacement",
        snippet="Replacement result.",
        origin="web",
    )
    cache_store.set(SourceName.WEB, "Q", source_value)
    cache_store.set(SourceName.WEB, "Q", [replacement])

    assert cache_store.get(SourceName.WEB, "Q") == [replacement]


def test_delete_removes_entry_and_ignores_missing_entry(
    cache_store: FileSystemCacheStore, source_value: list[Source]
) -> None:
    """Delete is safe for both present and absent cache entries."""

    cache_store.set(SourceName.WIKI, "Q", source_value)
    cache_store.delete(SourceName.WIKI, "Q")
    cache_store.delete(SourceName.WIKI, "Q")

    assert cache_store.get(SourceName.WIKI, "Q") is None


def test_new_instance_reads_persisted_entry(
    tmp_path: Path, fake_clock: FakeClock, source_value: list[Source]
) -> None:
    """Entries persist across cache-store instances."""

    cache_dir = tmp_path / "cache"
    first_store = FileSystemCacheStore(cache_dir, ttl_seconds=5, clock=fake_clock)
    first_store.set(SourceName.WIKI, "Q", source_value)
    second_store = FileSystemCacheStore(cache_dir, ttl_seconds=5, clock=fake_clock)

    assert second_store.get(SourceName.WIKI, "Q") == source_value


@pytest.mark.parametrize("ttl_seconds", [0, -1])
def test_non_positive_ttl_is_rejected(tmp_path: Path, ttl_seconds: int) -> None:
    """A cache store requires a meaningful positive TTL."""

    with pytest.raises(ValueError):
        FileSystemCacheStore(tmp_path / "cache", ttl_seconds)


def test_set_failure_raises_cache_error_with_cause(
    tmp_path: Path, source_value: list[Source]
) -> None:
    """Write failures are exposed to callers as domain cache errors."""

    cache_file = tmp_path / "cache-file"
    cache_file.write_text("not a directory", encoding="utf-8")
    cache_store = FileSystemCacheStore(cache_file, ttl_seconds=5)

    with pytest.raises(CacheError) as error_info:
        cache_store.set(SourceName.WIKI, "Q", source_value)

    assert isinstance(error_info.value.__cause__, OSError)


def test_file_system_store_implements_cache_store(cache_store: FileSystemCacheStore) -> None:
    """Person 2 can depend on the abstract cache-store interface."""

    assert isinstance(cache_store, CacheStore)


def cache_store_key(source: SourceName, query: str) -> CacheKey:
    """Build a cache key through the public model constructor."""

    return CacheKey.from_raw(source, query)
