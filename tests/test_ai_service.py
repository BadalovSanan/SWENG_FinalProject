"""Tests for the retrying AI service facade."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import pytest
from pydantic import ValidationError

import ai
from ai.providers.base import ProviderError
from ai.schemas import AnswerWithCitations, Source
from src.config import Settings
from src.logging_config import configure_logging
from src.services.ai_service import AIService


@pytest.fixture
def settings() -> Settings:
    """Provide explicit settings for facade tests."""

    return Settings(
        log_level="DEBUG",
        cache_dir=".cache",
        cache_ttl_seconds=60,
        per_source_timeout_seconds=1,
        max_sources_per_query=2,
    )


@pytest.fixture
def service(settings: Settings) -> AIService:
    """Provide a service with retry delays disabled for tests."""

    return AIService(settings, backoff_min=0, backoff_max=0, fetch_timeout=0.05)


@pytest.mark.asyncio
async def test_fetch_methods_pass_arguments_and_return_results(
    monkeypatch: pytest.MonkeyPatch, service: AIService, sample_sources: list[Source]
) -> None:
    """Each async fetch facade calls its supplied AI function at call time."""

    captured: dict[str, dict[str, Any]] = {}

    async def wiki(query: str, **kwargs: Any) -> list[Source]:
        captured["wiki"] = {"query": query, **kwargs}
        return sample_sources

    async def arxiv(query: str, **kwargs: Any) -> list[Source]:
        captured["arxiv"] = {"query": query, **kwargs}
        return sample_sources

    async def web(query: str, **kwargs: Any) -> list[Source]:
        captured["web"] = {"query": query, **kwargs}
        return sample_sources

    client = object()
    provider = object()
    monkeypatch.setattr(ai, "fetch_wikipedia", wiki)
    monkeypatch.setattr(ai, "fetch_arxiv", arxiv)
    monkeypatch.setattr(ai, "fetch_web", web)

    assert await service.fetch_wikipedia("wiki", client=client) == sample_sources
    assert await service.fetch_arxiv("arxiv", max_results=4, client=client) == sample_sources
    assert await service.fetch_web("web", provider=provider, client=client) == sample_sources
    assert captured["wiki"]["max_results"] == 2
    assert captured["wiki"]["client"] is client
    assert captured["arxiv"]["max_results"] == 4
    assert captured["arxiv"]["client"] is client
    assert captured["web"]["max_results"] == 2
    assert captured["web"]["provider"] is provider
    assert captured["web"]["client"] is client


@pytest.mark.asyncio
async def test_synthesize_passes_llm_and_runs_in_worker_thread(
    monkeypatch: pytest.MonkeyPatch, service: AIService, sample_sources: list[Source]
) -> None:
    """Synchronous synthesis is moved off the event loop and forwards its LLM."""

    main_thread = threading.get_ident()
    seen: dict[str, Any] = {}
    expected = AnswerWithCitations(question="Q", answer="Answer [1]", citations=[])

    def synthesize(question: str, sources: list[Source], *, llm: Any = None) -> AnswerWithCitations:
        seen.update(question=question, sources=sources, llm=llm, thread=threading.get_ident())
        return expected

    llm = object()
    monkeypatch.setattr(ai, "synthesize", synthesize)

    assert await service.synthesize("Q", sample_sources, llm=llm) == expected
    assert seen["question"] == "Q"
    assert seen["sources"] == sample_sources
    assert seen["llm"] is llm
    assert seen["thread"] != main_thread


@pytest.mark.asyncio
async def test_transient_failure_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, service: AIService, sample_sources: list[Source]
) -> None:
    """Provider failures are retried up to a later successful attempt."""

    calls = 0

    async def fetch(*args: Any, **kwargs: Any) -> list[Source]:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ProviderError("temporary")
        return sample_sources

    monkeypatch.setattr(ai, "fetch_wikipedia", fetch)

    assert await service.fetch_wikipedia("Q") == sample_sources
    assert calls == 3


@pytest.mark.asyncio
async def test_persistent_transient_failure_reraises_original_exception(
    monkeypatch: pytest.MonkeyPatch, service: AIService
) -> None:
    """The final ProviderError is exposed instead of a tenacity RetryError."""

    calls = 0

    async def fetch(*args: Any, **kwargs: Any) -> list[Source]:
        nonlocal calls
        calls += 1
        raise ProviderError("still unavailable")

    monkeypatch.setattr(ai, "fetch_arxiv", fetch)

    with pytest.raises(ProviderError):
        await service.fetch_arxiv("Q")
    assert calls == 3


@pytest.mark.asyncio
async def test_non_transient_value_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, service: AIService
) -> None:
    """Input and programming errors are returned to the caller immediately."""

    calls = 0

    async def fetch(*args: Any, **kwargs: Any) -> list[Source]:
        nonlocal calls
        calls += 1
        raise ValueError("bad query")

    monkeypatch.setattr(ai, "fetch_web", fetch)

    with pytest.raises(ValueError):
        await service.fetch_web("Q")
    assert calls == 1


@pytest.mark.asyncio
async def test_timeout_is_retried_and_reraised(
    monkeypatch: pytest.MonkeyPatch, service: AIService
) -> None:
    """Each timeout is transient but the final timeout reaches the caller."""

    calls = 0

    async def fetch(*args: Any, **kwargs: Any) -> list[Source]:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.1)
        return []

    monkeypatch.setattr(ai, "fetch_wikipedia", fetch)

    with pytest.raises(TimeoutError):
        await service.fetch_wikipedia("Q")
    assert calls == 3


@pytest.mark.asyncio
async def test_logging_records_retry_debug_content_and_no_secret(
    monkeypatch: pytest.MonkeyPatch,
    service: AIService,
    sample_sources: list[Source],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Logs expose operational details without leaking environment credentials."""

    calls = 0

    async def fetch(*args: Any, **kwargs: Any) -> list[Source]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError("temporary")
        return sample_sources

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret-value")
    monkeypatch.setattr(ai, "fetch_wikipedia", fetch)
    caplog.set_level(logging.DEBUG, logger="src.services.ai_service")

    await service.fetch_wikipedia("Q")

    assert any(record.levelno == logging.INFO for record in caplog.records)
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    assert any(record.levelno == logging.DEBUG for record in caplog.records)
    assert "sk-test-secret-value" not in caplog.text


def test_constructor_rejects_invalid_limits(settings: Settings) -> None:
    """Invalid attempt counts and timeouts are rejected at construction."""

    with pytest.raises(ValueError):
        AIService(settings, max_attempts=0)
    with pytest.raises(ValueError):
        AIService(settings, fetch_timeout=0)
    with pytest.raises(ValueError):
        AIService(settings, synthesis_timeout=0)


def test_configure_logging_is_idempotent(settings: Settings) -> None:
    """Repeated configuration does not add another marked stderr handler."""

    root_logger = logging.getLogger()
    configure_logging(settings)
    handler_count = sum(
        bool(getattr(handler, "_async_research_assistant_handler", False))
        for handler in root_logger.handlers
    )
    configure_logging(settings)

    assert root_logger.level == logging.DEBUG
    assert sum(
        bool(getattr(handler, "_async_research_assistant_handler", False))
        for handler in root_logger.handlers
    ) == handler_count


def test_invalid_log_level_cannot_reach_logging_configuration() -> None:
    """Settings validation prevents invalid levels from configuring logging."""

    with pytest.raises(ValidationError):
        Settings(log_level="BANANA")
