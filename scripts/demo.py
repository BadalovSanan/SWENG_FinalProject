"""Run all five questions through the real SE pipeline using existing offline fakes.

Run with ``python scripts/demo.py``. Outputs are reproducible demonstration
fixtures, not live research: demo_ai.py includes generic and illustrative sources.
No credentials are read and no HTTP or LLM-provider requests are made.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager, ExitStack
from functools import partial
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.providers.base import LLMProvider
from ai.schemas import Source
from demo_ai import _OfflineLLM, _OfflineSources
from src.concurrency.orchestrator import SourceOrchestrator
from src.config import Settings
from src.core.researcher import ResearchAssistant
from src.exceptions import ResearchError
from src.logging_config import configure_logging
from src.models import SourceName
from src.services.ai_service import AIService
from src.storage.cache_store import FileSystemCacheStore

QUESTIONS_PATH = PROJECT_ROOT / "data" / "research_questions.json"
ARTEFACTS_DIR = PROJECT_ROOT / "artefacts"
DEMO_NOTE = (
    "Offline demo using canned sources and a templated LLM from demo_ai.py. "
    "References may be generic or illustrative; this is not verified live research."
)


def load_questions() -> list[dict]:
    """Load the dataset, requiring five questions with unique, filename-safe IDs."""

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    if len(questions) != 5:
        raise ValueError("The demo dataset must contain exactly five questions.")
    ids = [item["id"] for item in questions]
    if len(set(ids)) != len(ids) or any(
        re.fullmatch(r"[a-zA-Z0-9_-]+", value) is None for value in ids
    ):
        raise ValueError("Question IDs must be unique and safe for filenames.")
    return questions


async def _fetch_offline(
    source: SourceName, query: str, *, max_results: int = 3, **kwargs: object
) -> list[Source]:
    """Adapt the supplied canned sources to the AI fetch boundary by origin."""

    return [
        item for item in _OfflineSources.fetch(query) if item.origin == source.origin
    ][:max_results]


@contextmanager
def offline_providers(
    llm: LLMProvider | None = None,
) -> Iterator[dict[SourceName, AsyncMock]]:
    """Temporarily replace external fetches and LLM selection for this demo only.

    The AIService retry/timeout facade and ai.synthesize citation handling stay
    real. ExitStack restores all boundaries even if the demo fails.
    """

    with ExitStack() as stack:
        fetchers = {}
        for source, name in (
            (SourceName.WIKI, "fetch_wikipedia"),
            (SourceName.ARXIV, "fetch_arxiv"),
            (SourceName.WEB, "fetch_web"),
        ):
            fetchers[source] = stack.enter_context(
                patch(f"ai.{name}", new_callable=AsyncMock,
                      side_effect=partial(_fetch_offline, source))
            )
        stack.enter_context(
            patch("ai.synthesizer.get_llm", return_value=llm or _OfflineLLM())
        )
        yield fetchers


def create_demo_assistant(cache_dir: Path) -> ResearchAssistant:
    """Construct the real application with an isolated, temporary demo cache."""

    settings = Settings(
        _env_file=None,
        cache_dir=cache_dir,
        cache_ttl_seconds=86_400,
        per_source_timeout_seconds=10,
        max_sources_per_query=3,
        log_level="WARNING",
    )
    service = AIService(settings)
    cache = FileSystemCacheStore(settings.cache_dir, settings.cache_ttl_seconds)
    orchestrator = SourceOrchestrator(service, settings)
    return ResearchAssistant(service, cache, orchestrator, settings)


async def run_demo(output_dir: Path = ARTEFACTS_DIR) -> list[Path]:
    """Research each dataset question and create or refresh its named sample JSON file."""

    questions = load_questions()
    paths = [output_dir / f"{item['id']}_offline.json" for item in questions]
    # Do not follow output symlinks that could point to unrelated files.
    if any(path.is_symlink() for path in paths):
        raise FileExistsError("Demo output is a symlink; no files were overwritten.")
    output_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="researcher-demo-cache-") as cache_dir:
        assistant = create_demo_assistant(Path(cache_dir))
        with offline_providers():
            for question, path in zip(questions, paths):
                # expected_sources is dataset guidance, not a required source filter.
                result = await assistant.ask(question["text"], use_cache=True)
                if not result.answer.strip() or not result.citations:
                    raise ValueError("The offline demo did not produce a cited answer.")
                payload = {
                    "question_id": question["id"],
                    "mode": "offline",
                    "demo_note": DEMO_NOTE,
                    **result.model_dump(mode="json"),
                    "degradation_notes": [
                        note.model_dump(mode="json")
                        for note in assistant.last_degradation_notes
                    ],
                }
                with path.open("w", encoding="utf-8") as output:
                    json.dump(payload, output, ensure_ascii=False, indent=2)
                    output.write("\n")
    return paths


def main() -> int:
    """Run the offline demo with concise intentional output and handled errors."""

    configure_logging(Settings(_env_file=None, log_level="WARNING"))
    print(DEMO_NOTE)
    try:
        paths = asyncio.run(run_demo())
    except (OSError, ValueError, ResearchError) as error:
        print(f"demo: error: {error}", file=sys.stderr)
        return 1
    for path in paths:
        print(f"Saved {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
