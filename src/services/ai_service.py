"""Retrying, timeout-aware facade around the provided AI module."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import logging
import time
from typing import Any, TypeVar

import httpx
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt, wait_exponential

from ai.providers.base import ProviderError
from ai.schemas import AnswerWithCitations, Source
from src.config import Settings


logger = logging.getLogger(__name__)
_Result = TypeVar("_Result")


class AIService:
    """Facade for ``ai.*`` calls with retry, timeout, and logging policy.

    Callers can receive ProviderError, TimeoutError, httpx.TransportError,
    retryable httpx.HTTPStatusError, or non-transient validation errors from
    the supplied AI layer after its final attempt.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        max_attempts: int = 3,
        backoff_min: float = 0.5,
        backoff_max: float = 8.0,
        fetch_timeout: float | None = None,
        synthesis_timeout: float = 60.0,
    ) -> None:
        """Create an AI facade with configurable retry and timeout limits."""

        resolved_fetch_timeout = (
            settings.per_source_timeout_seconds if fetch_timeout is None else fetch_timeout
        )
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if resolved_fetch_timeout <= 0 or synthesis_timeout <= 0:
            raise ValueError("timeouts must be greater than zero")
        self._settings = settings
        self._max_attempts = max_attempts
        self._backoff_min = backoff_min
        self._backoff_max = backoff_max
        self._fetch_timeout = resolved_fetch_timeout
        self._synthesis_timeout = synthesis_timeout

    async def fetch_wikipedia(
        self, query: str, *, max_results: int | None = None, client: Any = None
    ) -> list[Source]:
        """Fetch Wikipedia sources using the configured retry policy."""

        resolved_max_results = self._resolve_max_results(max_results)

        async def operation() -> list[Source]:
            import ai

            return await ai.fetch_wikipedia(
                query, max_results=resolved_max_results, client=client
            )

        return await self._run_async("fetch_wikipedia", len(query), operation, self._fetch_timeout)

    async def fetch_arxiv(
        self, query: str, *, max_results: int | None = None, client: Any = None
    ) -> list[Source]:
        """Fetch arXiv sources using the configured retry policy."""

        resolved_max_results = self._resolve_max_results(max_results)

        async def operation() -> list[Source]:
            import ai

            return await ai.fetch_arxiv(query, max_results=resolved_max_results, client=client)

        return await self._run_async("fetch_arxiv", len(query), operation, self._fetch_timeout)

    async def fetch_web(
        self,
        query: str,
        *,
        max_results: int | None = None,
        provider: Any = None,
        client: Any = None,
    ) -> list[Source]:
        """Fetch web sources using the configured retry policy."""

        resolved_max_results = self._resolve_max_results(max_results)

        async def operation() -> list[Source]:
            import ai

            return await ai.fetch_web(
                query,
                max_results=resolved_max_results,
                provider=provider,
                client=client,
            )

        return await self._run_async("fetch_web", len(query), operation, self._fetch_timeout)

    async def synthesize(
        self, question: str, sources: list[Source], *, llm: Any = None
    ) -> AnswerWithCitations:
        """Synthesize an answer off the event loop using the retry policy."""

        async def operation() -> AnswerWithCitations:
            import ai

            return await asyncio.to_thread(ai.synthesize, question, sources, llm=llm)

        return await self._run_async("synthesize", len(question), operation, self._synthesis_timeout)

    def _resolve_max_results(self, max_results: int | None) -> int:
        """Use the configured result limit when a method receives None."""

        return self._settings.max_sources_per_query if max_results is None else max_results

    async def _run_async(
        self,
        operation_name: str,
        input_length: int,
        operation: Callable[[], Awaitable[_Result]],
        timeout_seconds: float,
    ) -> _Result:
        """Run one asynchronous operation with transient-failure retries."""

        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(
                multiplier=self._backoff_min,
                min=self._backoff_min,
                max=self._backoff_max,
            ),
            retry=retry_if_exception(self._is_transient),
            before_sleep=lambda state: self._log_retry(operation_name, state),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                attempt_number = attempt.retry_state.attempt_number
                started_at = time.perf_counter()
                logger.info(
                    "%s attempt=%d input_length=%d",
                    operation_name,
                    attempt_number,
                    input_length,
                )
                async with asyncio.timeout(timeout_seconds):
                    result = await operation()
                duration = time.perf_counter() - started_at
                self._log_success(operation_name, attempt_number, duration, result)
                return result
        raise RuntimeError("Retry loop completed without a result")

    @staticmethod
    def _is_transient(error: BaseException) -> bool:
        """Return whether an error represents a retryable external failure."""

        if isinstance(error, (ProviderError, TimeoutError, httpx.TransportError)):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code == 429 or error.response.status_code >= 500
        return False

    @staticmethod
    def _log_retry(operation_name: str, state: RetryCallState) -> None:
        """Log a retry without including request payloads or credentials."""

        exception = state.outcome.exception() if state.outcome is not None else None
        logger.warning(
            "Retrying %s after attempt=%d error=%s",
            operation_name,
            state.attempt_number,
            type(exception).__name__ if exception is not None else "unknown",
        )

    @staticmethod
    def _log_success(
        operation_name: str, attempt_number: int, duration: float, result: object
    ) -> None:
        """Log a successful call without putting secrets in normal-level logs."""

        result_count = len(result) if isinstance(result, list) else None
        logger.info(
            "%s succeeded attempt=%d duration=%.3fs result_count=%s",
            operation_name,
            attempt_number,
            duration,
            result_count,
        )
        if isinstance(result, list):
            logger.debug(
                "%s result_sources=%s",
                operation_name,
                [(source.title, source.url) for source in result if isinstance(source, Source)],
            )
        elif isinstance(result, AnswerWithCitations):
            logger.debug("%s answer=%s", operation_name, result.answer)
