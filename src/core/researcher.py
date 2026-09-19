"""Core research-assistant business logic."""

from __future__ import annotations

from ai.schemas import AnswerWithCitations, Source
from src.concurrency.orchestrator import SourceOrchestrator
from src.config import Settings
from src.exceptions import AllSourcesFailedError
from src.models import DegradationNote, ResearchRequest, SourceName, canonicalize_query
from src.services.ai_service import AIService
from src.storage.cache_store import CacheStore


class ResearchAssistant:
    """Coordinate validation, caching, source fetching, and synthesis."""

    def __init__(
        self,
        ai_service: AIService,
        cache_store: CacheStore,
        orchestrator: SourceOrchestrator,
        settings: Settings,
    ) -> None:
        self._ai_service=ai_service
        self._cache_store=cache_store
        self._orchestrator=orchestrator
        self._settings=settings
        self._last_degradation_notes: list[DegradationNote]=[]

    @property
    def last_degradation_notes(self) -> list[DegradationNote]:
        """Return degradation notes produced by the most recent request."""

        return list(self._last_degradation_notes)

    async def ask(
        self,
        question: str,
        sources: list[SourceName | str] | None=None,
        use_cache: bool=True,
    ) -> AnswerWithCitations:
        """Research a validated question and synthesize an answer."""

        request=ResearchRequest.create(
            question=question,
            sources=sources,
        )

        canonical_query=canonicalize_query(request.question)
        requested_sources=request.sources or list(SourceName)

        collected_sources: list[Source]=[]
        missing_sources: list[SourceName]=[]

        if use_cache:
            for source_name in requested_sources:
                cached_sources=self._cache_store.get(
                    source_name,
                    canonical_query,
                )

                if cached_sources is None:
                    missing_sources.append(source_name)
                else:
                    collected_sources.extend(cached_sources)
        else:
            missing_sources=list(requested_sources)

        degradation_notes: list[DegradationNote]=[]

        if missing_sources:
            fetched_sources,degradation_notes=(
                await self._orchestrator.gather_sources(
                    canonical_query,
                    missing_sources,
                )
            )

            collected_sources.extend(fetched_sources)

            if use_cache:
                self._cache_fetched_sources(
                    canonical_query,
                    missing_sources,
                    fetched_sources,
                    degradation_notes,
                )

        self._last_degradation_notes=degradation_notes

        if not collected_sources:
            raise AllSourcesFailedError(
                "No research sources were available."
            )

        return await self._ai_service.synthesize(
            request.question,
            collected_sources,
        )

    def _cache_fetched_sources(
        self,
        query: str,
        requested_sources: list[SourceName],
        fetched_sources: list[Source],
        degradation_notes: list[DegradationNote],
    ) -> None:
        """Cache successful fetched results under their individual sources."""

        failed_sources={
            note.source
            for note in degradation_notes
        }

        for source_name in requested_sources:
            if source_name in failed_sources:
                continue

            matching_sources=[
                source
                for source in fetched_sources
                if source.origin==source_name.origin
            ]

            self._cache_store.set(
                source_name,
                query,
                matching_sources,
            )