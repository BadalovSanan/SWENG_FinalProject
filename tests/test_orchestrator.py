"""Tests for concurrent source orchestration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from ai.schemas import Source
from src.concurrency.orchestrator import SourceOrchestrator
from src.config import Settings
from src.models import SourceName


def make_source(title: str,origin: str) -> Source:
    return Source(
        title=title,
        url=f"https://example.com/{title.lower()}",
        snippet=f"{title} result",
        origin=origin,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        per_source_timeout_seconds=1,
        max_sources_per_query=3,
    )


@pytest.fixture
def ai_service() -> AsyncMock:
    service=AsyncMock()

    service.fetch_wikipedia.return_value=[
        make_source(
            "Wikipedia",
            "wikipedia",
        )
    ]

    service.fetch_arxiv.return_value=[
        make_source(
            "Arxiv",
            "arxiv",
        )
    ]

    service.fetch_web.return_value=[
        make_source(
            "Web",
            "web",
        )
    ]

    return service


@pytest.mark.asyncio
async def test_all_sources_succeed(
    ai_service: AsyncMock,
    settings: Settings,
) -> None:
    orchestrator=SourceOrchestrator(
        ai_service,
        settings,
    )

    sources,notes=await orchestrator.gather_sources(
        "What is photosynthesis?"
    )

    assert len(sources)==3
    assert notes==[]

    ai_service.fetch_wikipedia.assert_awaited_once()
    ai_service.fetch_arxiv.assert_awaited_once()
    ai_service.fetch_web.assert_awaited_once()


@pytest.mark.asyncio
async def test_one_source_failure_degrades_gracefully(
    ai_service: AsyncMock,
    settings: Settings,
) -> None:
    ai_service.fetch_arxiv.side_effect=RuntimeError(
        "arXiv unavailable"
    )

    orchestrator=SourceOrchestrator(
        ai_service,
        settings,
    )

    sources,notes=await orchestrator.gather_sources(
        "What is photosynthesis?"
    )

    assert len(sources)==2
    assert len(notes)==1

    assert notes[0].source is SourceName.ARXIV
    assert "RuntimeError" in notes[0].reason

    origins={
        source.origin
        for source in sources
    }

    assert origins=={
        "wikipedia",
        "web",
    }


@pytest.mark.asyncio
async def test_source_filter_only_calls_requested_sources(
    ai_service: AsyncMock,
    settings: Settings,
) -> None:
    orchestrator=SourceOrchestrator(
        ai_service,
        settings,
    )

    sources,notes=await orchestrator.gather_sources(
        "What is photosynthesis?",
        [
            SourceName.WIKI,
            SourceName.ARXIV,
        ],
    )

    assert len(sources)==2
    assert notes==[]

    ai_service.fetch_wikipedia.assert_awaited_once()
    ai_service.fetch_arxiv.assert_awaited_once()
    ai_service.fetch_web.assert_not_awaited()


@pytest.mark.asyncio
async def test_sources_run_concurrently(
    ai_service: AsyncMock,
    settings: Settings,
) -> None:
    active_fetches=0
    maximum_active_fetches=0

    async def concurrent_fetch(
        source: Source,
    ) -> list[Source]:
        nonlocal active_fetches,maximum_active_fetches

        active_fetches+=1

        maximum_active_fetches=max(
            maximum_active_fetches,
            active_fetches,
        )

        await asyncio.sleep(.05)

        active_fetches-=1

        return [source]

    async def fetch_wikipedia(*args,**kwargs):
        return await concurrent_fetch(
            make_source(
                "Wikipedia",
                "wikipedia",
            )
        )

    async def fetch_arxiv(*args,**kwargs):
        return await concurrent_fetch(
            make_source(
                "Arxiv",
                "arxiv",
            )
        )

    async def fetch_web(*args,**kwargs):
        return await concurrent_fetch(
            make_source(
                "Web",
                "web",
            )
        )

    ai_service.fetch_wikipedia.side_effect=fetch_wikipedia
    ai_service.fetch_arxiv.side_effect=fetch_arxiv
    ai_service.fetch_web.side_effect=fetch_web

    orchestrator=SourceOrchestrator(
        ai_service,
        settings,
    )

    sources,notes=await orchestrator.gather_sources(
        "What is photosynthesis?"
    )

    assert len(sources)==3
    assert notes==[]

    assert maximum_active_fetches==3


@pytest.mark.asyncio
async def test_slow_source_times_out_without_losing_fast_sources(
    ai_service: AsyncMock,
) -> None:
    settings=Settings(
        per_source_timeout_seconds=1,
        max_sources_per_query=3,
    )

    async def very_slow_arxiv(*args,**kwargs):
        await asyncio.sleep(2)

        return [
            make_source(
                "Arxiv",
                "arxiv",
            )
        ]

    ai_service.fetch_arxiv.side_effect=very_slow_arxiv

    orchestrator=SourceOrchestrator(
        ai_service,
        settings,
    )

    loop=asyncio.get_running_loop()
    started=loop.time()

    sources,notes=await orchestrator.gather_sources(
        "What is photosynthesis?"
    )

    elapsed=loop.time()-started

    assert elapsed<1.5

    origins={
        source.origin
        for source in sources
    }

    assert origins=={
        "wikipedia",
        "web",
    }

    assert len(notes)==1
    assert notes[0].source is SourceName.ARXIV
    assert notes[0].reason=="Source timed out."


@pytest.mark.asyncio
async def test_semaphore_limits_concurrent_source_fetches(
    ai_service: AsyncMock,
) -> None:
    """The semaphore should limit simultaneous source fetches."""

    settings=Settings(
        per_source_timeout_seconds=2,
        max_sources_per_query=2,
    )

    active_fetches=0
    maximum_active_fetches=0

    async def limited_fetch(
        source: Source,
    ) -> list[Source]:
        nonlocal active_fetches,maximum_active_fetches

        active_fetches+=1

        maximum_active_fetches=max(
            maximum_active_fetches,
            active_fetches,
        )

        await asyncio.sleep(.05)

        active_fetches-=1

        return [source]

    async def fetch_wikipedia(*args,**kwargs):
        return await limited_fetch(
            make_source(
                "Wikipedia",
                "wikipedia",
            )
        )

    async def fetch_arxiv(*args,**kwargs):
        return await limited_fetch(
            make_source(
                "Arxiv",
                "arxiv",
            )
        )

    async def fetch_web(*args,**kwargs):
        return await limited_fetch(
            make_source(
                "Web",
                "web",
            )
        )

    ai_service.fetch_wikipedia.side_effect=fetch_wikipedia
    ai_service.fetch_arxiv.side_effect=fetch_arxiv
    ai_service.fetch_web.side_effect=fetch_web

    orchestrator=SourceOrchestrator(
        ai_service,
        settings,
    )

    sources,notes=await orchestrator.gather_sources(
        "What is photosynthesis?"
    )

    assert len(sources)==3
    assert notes==[]

    assert maximum_active_fetches==2