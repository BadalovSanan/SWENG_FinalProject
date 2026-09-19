"""Concurrent source orchestration for the research assistant."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx

from ai.schemas import Source
from src.config import Settings
from src.models import DegradationNote, SourceName
from src.services.ai_service import AIService


class SourceOrchestrator:
    """Fetch research sources concurrently with graceful degradation."""

    def __init__(self,ai_service: AIService,settings: Settings) -> None:
        self._ai_service=ai_service
        self._settings=settings
        self._semaphore=asyncio.Semaphore(settings.max_sources_per_query)

    async def gather_sources(
        self,
        question: str,
        source_filter: list[SourceName] | None=None,
    ) -> tuple[list[Source],list[DegradationNote]]:
        """Fetch the requested sources concurrently."""

        selected_sources=source_filter or list(SourceName)

        async with httpx.AsyncClient() as client:
            tasks=[
                self._fetch_source(source,question,client)
                for source in selected_sources
            ]

            results=await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

        sources: list[Source]=[]
        degradation_notes: list[DegradationNote]=[]

        for source_name,result in zip(selected_sources,results):
            if isinstance(result,BaseException):
                degradation_notes.append(
                    DegradationNote(
                        source=source_name,
                        reason=self._failure_reason(result),
                    )
                )
            else:
                sources.extend(result)

        return sources,degradation_notes

    async def _fetch_source(
        self,
        source: SourceName,
        question: str,
        client: httpx.AsyncClient,
    ) -> list[Source]:
        """Fetch one source under its own timeout."""

        async with asyncio.timeout(self._settings.per_source_timeout_seconds):
            if source is SourceName.WIKI:
                return await self._ai_service.fetch_wikipedia(
                    question,
                    client=client,
                )

            if source is SourceName.ARXIV:
                return await self._ai_service.fetch_arxiv(
                    question,
                    client=client,
                )

            if source is SourceName.WEB:
                return await self._ai_service.fetch_web(
                    question,
                    client=client,
                )

        raise ValueError(f"Unsupported source: {source}")

    async def run_bounded(
        self,
        operation: Callable[[],Awaitable[object]],
    ) -> object:
        """Run one research operation under the shared concurrency bound."""

        async with self._semaphore:
            return await operation()

    @staticmethod
    def _failure_reason(error: BaseException) -> str:
        """Convert a source failure into a user-safe degradation reason."""

        if isinstance(error,TimeoutError):
            return "Source timed out."

        return f"Source failed with {type(error).__name__}."