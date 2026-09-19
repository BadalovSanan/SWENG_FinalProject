"""Tests for the research assistant business logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from ai.schemas import AnswerWithCitations, Source
from src.config import Settings
from src.core.researcher import ResearchAssistant
from src.exceptions import AllSourcesFailedError, InvalidQuestionError
from src.models import DegradationNote, SourceName


def make_source(title: str,origin: str) -> Source:
    """Create a source object for researcher tests."""

    return Source(
        title=title,
        url=f"https://example.com/{title.lower()}",
        snippet=f"{title} result",
        origin=origin,
    )


def make_answer(question: str) -> AnswerWithCitations:
    """Create a small synthesized answer for tests."""

    return AnswerWithCitations(
        question=question,
        answer="Synthesized answer",
        citations=[],
    )


@pytest.fixture
def settings() -> Settings:
    """Return settings for researcher tests."""

    return Settings()


@pytest.fixture
def ai_service() -> AsyncMock:
    """Return a mocked AI service."""

    service=AsyncMock()
    service.synthesize.return_value=make_answer(
        "What is photosynthesis?"
    )
    return service


@pytest.fixture
def cache_store() -> Mock:
    """Return a mocked cache store."""

    store=Mock()
    store.get.return_value=None
    return store


@pytest.fixture
def orchestrator() -> AsyncMock:
    """Return a mocked source orchestrator."""

    service=AsyncMock()
    service.gather_sources.return_value=(
        [
            make_source("Wikipedia","wikipedia"),
            make_source("Arxiv","arxiv"),
            make_source("Web","web"),
        ],
        [],
    )
    return service


@pytest.fixture
def researcher(
    ai_service: AsyncMock,
    cache_store: Mock,
    orchestrator: AsyncMock,
    settings: Settings,
) -> ResearchAssistant:
    """Return a research assistant wired to mocked dependencies."""

    return ResearchAssistant(
        ai_service,
        cache_store,
        orchestrator,
        settings,
    )


@pytest.mark.asyncio
async def test_happy_path_fetches_and_synthesizes(
    researcher: ResearchAssistant,
    ai_service: AsyncMock,
    cache_store: Mock,
    orchestrator: AsyncMock,
) -> None:
    """A normal cache miss should fetch, cache, and synthesize."""

    result=await researcher.ask(
        "What is photosynthesis?"
    )

    assert result.answer=="Synthesized answer"

    orchestrator.gather_sources.assert_awaited_once_with(
        "what is photosynthesis",
        [
            SourceName.WIKI,
            SourceName.ARXIV,
            SourceName.WEB,
        ],
    )

    assert cache_store.set.call_count==3

    ai_service.synthesize.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_empty_question_rejected_before_dependencies(
    researcher: ResearchAssistant,
    ai_service: AsyncMock,
    cache_store: Mock,
    orchestrator: AsyncMock,
) -> None:
    """An empty question should fail before cache or network work."""

    with pytest.raises(InvalidQuestionError):
        await researcher.ask("   ")

    cache_store.get.assert_not_called()
    cache_store.set.assert_not_called()
    orchestrator.gather_sources.assert_not_awaited()
    ai_service.synthesize.assert_not_awaited()


@pytest.mark.asyncio
async def test_oversized_question_rejected_before_dependencies(
    researcher: ResearchAssistant,
    ai_service: AsyncMock,
    cache_store: Mock,
    orchestrator: AsyncMock,
) -> None:
    """An oversized question should fail before cache or network work."""

    with pytest.raises(InvalidQuestionError):
        await researcher.ask("a"*2001)

    cache_store.get.assert_not_called()
    cache_store.set.assert_not_called()
    orchestrator.gather_sources.assert_not_awaited()
    ai_service.synthesize.assert_not_awaited()


@pytest.mark.asyncio
async def test_cache_hit_skips_orchestrator(
    researcher: ResearchAssistant,
    ai_service: AsyncMock,
    cache_store: Mock,
    orchestrator: AsyncMock,
) -> None:
    """Warm cache entries should avoid source fetching."""

    wiki=make_source("Wikipedia","wikipedia")
    arxiv=make_source("Arxiv","arxiv")
    web=make_source("Web","web")

    cache_store.get.side_effect=[
        [wiki],
        [arxiv],
        [web],
    ]

    await researcher.ask(
        "What is photosynthesis?"
    )

    assert cache_store.get.call_count==3
    orchestrator.gather_sources.assert_not_awaited()
    cache_store.set.assert_not_called()

    ai_service.synthesize.assert_awaited_once_with(
        "What is photosynthesis?",
        [wiki,arxiv,web],
    )


@pytest.mark.asyncio
async def test_no_cache_bypasses_cache_reads_and_writes(
    researcher: ResearchAssistant,
    cache_store: Mock,
    orchestrator: AsyncMock,
) -> None:
    """use_cache=False should bypass both cache reads and writes."""

    await researcher.ask(
        "What is photosynthesis?",
        use_cache=False,
    )

    cache_store.get.assert_not_called()
    cache_store.set.assert_not_called()

    orchestrator.gather_sources.assert_awaited_once_with(
        "what is photosynthesis",
        [
            SourceName.WIKI,
            SourceName.ARXIV,
            SourceName.WEB,
        ],
    )


@pytest.mark.asyncio
async def test_source_filter_is_threaded_to_orchestrator(
    researcher: ResearchAssistant,
    cache_store: Mock,
    orchestrator: AsyncMock,
) -> None:
    """Requested source filters should reach the orchestrator."""

    orchestrator.gather_sources.return_value=(
        [
            make_source("Wikipedia","wikipedia"),
            make_source("Web","web"),
        ],
        [],
    )

    await researcher.ask(
        "What is photosynthesis?",
        sources=["wiki","web"],
    )

    orchestrator.gather_sources.assert_awaited_once_with(
        "what is photosynthesis",
        [
            SourceName.WIKI,
            SourceName.WEB,
        ],
    )

    assert cache_store.get.call_count==2


@pytest.mark.asyncio
async def test_partial_cache_hit_only_fetches_missing_sources(
    researcher: ResearchAssistant,
    cache_store: Mock,
    orchestrator: AsyncMock,
    ai_service: AsyncMock,
) -> None:
    """Only sources missing from the cache should be fetched."""

    wiki=make_source("Wikipedia","wikipedia")
    arxiv=make_source("Arxiv","arxiv")
    web=make_source("Web","web")

    cache_store.get.side_effect=[
        [wiki],
        None,
        None,
    ]

    orchestrator.gather_sources.return_value=(
        [arxiv,web],
        [],
    )

    await researcher.ask(
        "What is photosynthesis?"
    )

    orchestrator.gather_sources.assert_awaited_once_with(
        "what is photosynthesis",
        [
            SourceName.ARXIV,
            SourceName.WEB,
        ],
    )

    ai_service.synthesize.assert_awaited_once_with(
        "What is photosynthesis?",
        [wiki,arxiv,web],
    )


@pytest.mark.asyncio
async def test_all_sources_failed_raises_domain_error(
    researcher: ResearchAssistant,
    ai_service: AsyncMock,
    orchestrator: AsyncMock,
) -> None:
    """No available sources should raise AllSourcesFailedError."""

    orchestrator.gather_sources.return_value=(
        [],
        [
            DegradationNote(
                source=SourceName.WIKI,
                reason="Source failed.",
            ),
            DegradationNote(
                source=SourceName.ARXIV,
                reason="Source failed.",
            ),
            DegradationNote(
                source=SourceName.WEB,
                reason="Source failed.",
            ),
        ],
    )

    with pytest.raises(AllSourcesFailedError):
        await researcher.ask(
            "What is photosynthesis?"
        )

    ai_service.synthesize.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_failure_still_synthesizes(
    researcher: ResearchAssistant,
    ai_service: AsyncMock,
    orchestrator: AsyncMock,
) -> None:
    """Successful sources should still be synthesized after one failure."""

    wiki=make_source("Wikipedia","wikipedia")
    web=make_source("Web","web")

    orchestrator.gather_sources.return_value=(
        [wiki,web],
        [
            DegradationNote(
                source=SourceName.ARXIV,
                reason="Source failed with RuntimeError.",
            )
        ],
    )

    await researcher.ask(
        "What is photosynthesis?"
    )

    ai_service.synthesize.assert_awaited_once_with(
        "What is photosynthesis?",
        [wiki,web],
    )

    assert len(researcher.last_degradation_notes)==1
    assert (
        researcher.last_degradation_notes[0].source
        is SourceName.ARXIV
    )