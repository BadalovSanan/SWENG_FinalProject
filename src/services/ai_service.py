"""Retrying, timeout-aware facade around the provided AI module."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import json
import logging
import re
import time
from typing import Any, TypeVar
from urllib.parse import urlsplit, urlunsplit

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

        return await self._run_async(
            "fetch_wikipedia",
            {"query": query, "max_results": resolved_max_results},
            operation,
            self._fetch_timeout,
        )

    async def fetch_arxiv(
        self, query: str, *, max_results: int | None = None, client: Any = None
    ) -> list[Source]:
        """Fetch arXiv sources using the configured retry policy."""

        resolved_max_results = self._resolve_max_results(max_results)

        async def operation() -> list[Source]:
            import ai

            return await ai.fetch_arxiv(query, max_results=resolved_max_results, client=client)

        return await self._run_async(
            "fetch_arxiv",
            {"query": query, "max_results": resolved_max_results},
            operation,
            self._fetch_timeout,
        )

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

        return await self._run_async(
            "fetch_web",
            {
                "query": query,
                "max_results": resolved_max_results,
                "provider_type": type(provider).__name__ if provider is not None else None,
            },
            operation,
            self._fetch_timeout,
        )

    async def synthesize(
        self, question: str, sources: list[Source], *, llm: Any = None
    ) -> AnswerWithCitations:
        """Synthesize an answer off the event loop using the retry policy."""

        async def operation() -> AnswerWithCitations:
            import ai

            return await asyncio.to_thread(ai.synthesize, question, sources, llm=llm)

        return await self._run_async(
            "synthesize",
            {
                "question": question,
                "source_count": len(sources),
                "sources": [self._safe_source_payload(source) for source in sources],
                "llm_type": type(llm).__name__ if llm is not None else None,
            },
            operation,
            self._synthesis_timeout,
        )

    def _resolve_max_results(self, max_results: int | None) -> int:
        """Use the configured result limit when a method receives None."""

        return self._settings.max_sources_per_query if max_results is None else max_results

    async def _run_async(
        self,
        operation_name: str,
        request_payload: dict[str, object],
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
            before_sleep=lambda state: self._log_retry(
                operation_name, request_payload, state
            ),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                attempt_number = attempt.retry_state.attempt_number
                return await self._run_attempt(
                    operation_name,
                    attempt_number,
                    request_payload,
                    operation,
                    timeout_seconds,
                )
        raise RuntimeError("Retry loop completed without a result")

    async def _run_attempt(
        self,
        operation_name: str,
        attempt_number: int,
        request_payload: dict[str, object],
        operation: Callable[[], Awaitable[_Result]],
        timeout_seconds: float,
    ) -> _Result:
        """Run and log one timed attempt without swallowing its exception."""

        safe_request = self._format_payload(request_payload)
        logger.info(
            "%s status=started attempt=%d request=%s",
            operation_name,
            attempt_number,
            self._info_request_payload(request_payload),
        )
        logger.debug("%s request_payload=%s", operation_name, safe_request)
        started_at = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_seconds):
                result = await operation()
        except Exception as error:
            duration = time.perf_counter() - started_at
            logger.info(
                "%s status=failed attempt=%d duration=%.3fs error_type=%s request=%s",
                operation_name,
                attempt_number,
                duration,
                type(error).__name__,
                self._info_request_payload(request_payload),
            )
            raise

        duration = time.perf_counter() - started_at
        self._log_success(
            operation_name, attempt_number, duration, request_payload, result
        )
        return result

    @staticmethod
    def _is_transient(error: BaseException) -> bool:
        """Return whether an error represents a retryable external failure."""

        if isinstance(error, (ProviderError, TimeoutError, httpx.TransportError)):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code == 429 or error.response.status_code >= 500
        return False

    @staticmethod
    def _log_retry(
        operation_name: str,
        request_payload: dict[str, object],
        state: RetryCallState,
    ) -> None:
        """Log a retry without including request payloads or credentials."""

        exception = state.outcome.exception() if state.outcome is not None else None
        logger.warning(
            "Retrying %s after attempt=%d error_type=%s request=%s",
            operation_name,
            state.attempt_number,
            type(exception).__name__ if exception is not None else "unknown",
            AIService._info_request_payload(request_payload),
        )

    @staticmethod
    def _log_success(
        operation_name: str,
        attempt_number: int,
        duration: float,
        request_payload: dict[str, object],
        result: object,
    ) -> None:
        """Log a successful call without putting secrets in normal-level logs."""

        result_summary = AIService._result_summary(result)
        logger.info(
            "%s status=succeeded attempt=%d duration=%.3fs request=%s result=%s",
            operation_name,
            attempt_number,
            duration,
            AIService._info_request_payload(request_payload),
            AIService._format_payload(result_summary),
        )
        logger.debug(
            "%s response_payload=%s",
            operation_name,
            AIService._format_payload(AIService._result_payload(result)),
        )

    @staticmethod
    def _info_request_payload(request_payload: dict[str, object]) -> str:
        """Return a concise safe request summary for INFO logs."""

        payload = dict(request_payload)
        payload.pop("sources", None)
        return AIService._format_payload(payload)

    @staticmethod
    def _result_summary(result: object) -> dict[str, object]:
        """Return concise safe result information for INFO logs."""

        if isinstance(result, list):
            sources = [source for source in result if isinstance(source, Source)]
            return {
                "result_count": len(result),
                "origins": [source.origin for source in sources],
                "titles": [source.title for source in sources],
            }
        if isinstance(result, AnswerWithCitations):
            return {
                "answer_length": len(result.answer),
                "citation_count": len(result.citations),
            }
        return {"result_type": type(result).__name__}

    @staticmethod
    def _result_payload(result: object) -> dict[str, object]:
        """Return the complete safe response payload for DEBUG logs."""

        if isinstance(result, list):
            return {
                "sources": [
                    AIService._safe_source_payload(source)
                    for source in result
                    if isinstance(source, Source)
                ]
            }
        if isinstance(result, AnswerWithCitations):
            return {
                "question": AIService._redact_secrets(result.question),
                "answer": AIService._redact_secrets(result.answer),
                "citations": [
                    {
                        "index": citation.index,
                        "source": AIService._safe_source_payload(citation.source),
                    }
                    for citation in result.citations
                ],
            }
        return {"result_type": type(result).__name__}

    @staticmethod
    def _safe_source_payload(source: Source) -> dict[str, str]:
        """Serialize source data for logs while removing URL credentials."""

        return {
            "title": AIService._redact_secrets(source.title),
            "url": AIService._safe_url(source.url),
            "snippet": AIService._redact_secrets(source.snippet),
            "origin": source.origin,
        }

    @staticmethod
    def _safe_url(url: str) -> str:
        """Remove URL credentials, queries, and fragments before logging."""

        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))

    @staticmethod
    def _redact_secrets(value: str) -> str:
        """Redact common credential assignments before they reach a log record."""

        return re.sub(
            r"(?i)\b(api[_ -]?key|authorization|bearer|token|secret|password)\b"
            r"\s*([=:])\s*[^\s,;]+",
            r"\1\2[REDACTED]",
            value,
        )

    @staticmethod
    def _format_payload(payload: dict[str, object]) -> str:
        """Format a safe structured payload consistently for log output."""

        return json.dumps(
            AIService._sanitize_payload(payload), ensure_ascii=False, sort_keys=True
        )

    @staticmethod
    def _sanitize_payload(value: object) -> object:
        """Recursively redact strings in a structured log payload."""

        if isinstance(value, str):
            return AIService._redact_secrets(value)
        if isinstance(value, dict):
            return {
                str(key): AIService._sanitize_payload(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [AIService._sanitize_payload(item) for item in value]
        return value
