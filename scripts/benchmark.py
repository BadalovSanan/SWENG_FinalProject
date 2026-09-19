"""Benchmark sequential versus parallel source fetching."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import httpx

PROJECT_ROOT=Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from ai.schemas import Source
from src.concurrency.orchestrator import SourceOrchestrator
from src.config import Settings
from src.models import SourceName


logger=logging.getLogger(__name__)

QUESTIONS_PATH=PROJECT_ROOT/"data"/"research_questions.json"
SIMULATED_DELAY_SECONDS=0.5


class BenchmarkAIService:
    """Provide deterministic source fetches for concurrency benchmarking."""

    async def fetch_wikipedia(
        self,
        query: str,
        *,
        max_results: int | None=None,
        client: Any=None,
    ) -> list[Source]:
        """Simulate a Wikipedia source fetch."""

        await asyncio.sleep(SIMULATED_DELAY_SECONDS)

        return [
            Source(
                title="Wikipedia benchmark result",
                url="https://example.com/wikipedia",
                snippet=query,
                origin="wikipedia",
            )
        ]

    async def fetch_arxiv(
        self,
        query: str,
        *,
        max_results: int | None=None,
        client: Any=None,
    ) -> list[Source]:
        """Simulate an arXiv source fetch."""

        await asyncio.sleep(SIMULATED_DELAY_SECONDS)

        return [
            Source(
                title="arXiv benchmark result",
                url="https://example.com/arxiv",
                snippet=query,
                origin="arxiv",
            )
        ]

    async def fetch_web(
        self,
        query: str,
        *,
        max_results: int | None=None,
        provider: Any=None,
        client: Any=None,
    ) -> list[Source]:
        """Simulate a web source fetch."""

        await asyncio.sleep(SIMULATED_DELAY_SECONDS)

        return [
            Source(
                title="Web benchmark result",
                url="https://example.com/web",
                snippet=query,
                origin="web",
            )
        ]


def load_questions() -> list[str]:
    """Load the five benchmark questions from the project dataset."""

    payload=json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )

    questions=[
        item["text"]
        for item in payload["questions"]
    ]

    if len(questions)!=5:
        raise ValueError(
            "Benchmark dataset must contain exactly five questions."
        )

    return questions


async def fetch_sequentially(
    question: str,
    ai_service: BenchmarkAIService,
) -> list[Source]:
    """Fetch all three source types one after another."""

    sources: list[Source]=[]

    async with httpx.AsyncClient() as client:
        sources.extend(
            await ai_service.fetch_wikipedia(
                question,
                client=client,
            )
        )

        sources.extend(
            await ai_service.fetch_arxiv(
                question,
                client=client,
            )
        )

        sources.extend(
            await ai_service.fetch_web(
                question,
                client=client,
            )
        )

    return sources


async def fetch_in_parallel(
    question: str,
    orchestrator: SourceOrchestrator,
) -> list[Source]:
    """Fetch all source types through the concurrent orchestrator."""

    sources,notes=await orchestrator.gather_sources(
        question,
        [
            SourceName.WIKI,
            SourceName.ARXIV,
            SourceName.WEB,
        ],
    )

    if notes:
        raise RuntimeError(
            "Benchmark source fetch unexpectedly degraded."
        )

    return sources


async def benchmark_question(
    question: str,
    ai_service: BenchmarkAIService,
    orchestrator: SourceOrchestrator,
) -> tuple[float,float]:
    """Measure sequential and parallel wall-clock time."""

    sequential_started=time.perf_counter()

    sequential_sources=await fetch_sequentially(
        question,
        ai_service,
    )

    sequential_time=time.perf_counter()-sequential_started

    parallel_started=time.perf_counter()

    parallel_sources=await fetch_in_parallel(
        question,
        orchestrator,
    )

    parallel_time=time.perf_counter()-parallel_started

    if len(sequential_sources)!=len(parallel_sources):
        raise RuntimeError(
            "Sequential and parallel runs returned different source counts."
        )

    return sequential_time,parallel_time


async def main() -> None:
    """Run the five-question concurrency benchmark."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    questions=load_questions()

    settings=Settings(
        per_source_timeout_seconds=5,
        max_sources_per_query=3,
    )

    ai_service=BenchmarkAIService()

    orchestrator=SourceOrchestrator(
        ai_service,  # type: ignore[arg-type]
        settings,
    )

    sequential_total=0.0
    parallel_total=0.0

    for index,question in enumerate(questions,start=1):
        sequential_time,parallel_time=await benchmark_question(
            question,
            ai_service,
            orchestrator,
        )

        sequential_total+=sequential_time
        parallel_total+=parallel_time

        logger.info("")
        logger.info("Question %d: %s",index,question)
        logger.info(
            "Sequential: %.2fs | Parallel: %.2fs",
            sequential_time,
            parallel_time,
        )

    speedup=(
        sequential_total/parallel_total
        if parallel_total>0
        else 0.0
    )

    logger.info("")
    logger.info("Benchmark summary")
    logger.info("-----------------")
    logger.info(
        "Sequential total: %.2fs",
        sequential_total,
    )
    logger.info(
        "Parallel total:   %.2fs",
        parallel_total,
    )
    logger.info(
        "Speedup:          %.2fx",
        speedup,
    )


if __name__=="__main__":
    asyncio.run(main())