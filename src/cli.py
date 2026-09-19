"""Command-line integration for the async research assistant."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
import re
import sys

from src.concurrency.orchestrator import SourceOrchestrator
from src.config import get_settings
from src.core.researcher import ResearchAssistant
from src.exceptions import AllSourcesFailedError, InvalidQuestionError
from src.logging_config import configure_logging
from src.models import SourceName
from src.services.ai_service import AIService
from src.storage.cache_store import FileSystemCacheStore


# Strip terminal strings (including OSC hyperlinks), CSI styling/cursor commands,
# and other escape sequences before removing remaining C0/C1 control characters.
_ANSI_ESCAPE = re.compile(
    r"(?:\x1b\]|\x9d)[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c|$)"
    r"|(?:\x1b[P^_X]|[\x90\x98\x9e\x9f])[^\x1b\x9c]*(?:\x1b\\|\x9c|$)"
    r"|(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]"
    r"|\x1b[ -/]*[@-Z\\-_]"
)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _sanitize_terminal(text: str) -> str:
    """Keep normal Unicode, newlines, tabs, citation markers, and URL text."""

    return _CONTROL_CHARACTERS.sub("", _ANSI_ESCAPE.sub("", text))


def create_assistant() -> ResearchAssistant:
    """Wire the existing application components at one patchable boundary."""

    settings = get_settings()
    configure_logging(settings)
    ai_service = AIService(settings)
    cache_store = FileSystemCacheStore(
        settings.cache_dir, settings.cache_ttl_seconds
    )
    orchestrator = SourceOrchestrator(ai_service, settings)
    return ResearchAssistant(ai_service, cache_store, orchestrator, settings)


def _parse_sources(value: str) -> list[SourceName]:
    """Parse a non-empty comma-separated filter; the request model deduplicates."""

    try:
        return [SourceName(part.strip()) for part in value.split(",")]
    except ValueError:
        allowed = ",".join(source.value for source in SourceName)
        raise argparse.ArgumentTypeError(
            f"sources must be comma-separated names from {allowed}, with no empty entries"
        ) from None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI, returning an exit code for handled domain errors."""

    parser = argparse.ArgumentParser(
        prog="researcher", description="Research a question with cited sources."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ask_parser = commands.add_parser("ask", help="research a question")
    ask_parser.add_argument("question", help="the research question")
    ask_parser.add_argument(
        "--sources", type=_parse_sources, metavar="wiki,arxiv,web",
        help="comma-separated source selection (default: all sources)",
    )
    ask_parser.add_argument(
        "--no-cache", action="store_true", help="bypass cache reads and writes"
    )
    args = parser.parse_args(argv)

    assistant = create_assistant()
    try:
        result = asyncio.run(
            assistant.ask(
                args.question, sources=args.sources, use_cache=not args.no_cache
            )
        )
    except InvalidQuestionError as error:
        print(f"researcher: error: {_sanitize_terminal(str(error))}", file=sys.stderr)
        return 2
    except AllSourcesFailedError as error:
        print(f"researcher: error: {_sanitize_terminal(str(error))}", file=sys.stderr)
        return 1

    print(_sanitize_terminal(result.answer))
    print("\nReferences:")
    for citation in sorted(result.citations, key=lambda item: item.index):
        source = citation.source
        print(_sanitize_terminal(f"  [{citation.index}] ({source.origin}) {source.title}"))
        print(_sanitize_terminal(f"      {source.url}"))

    notes = assistant.last_degradation_notes
    if notes:
        print("\nNote: Some sources were unavailable; the answer uses the remaining sources.")
        for note in notes:
            print(_sanitize_terminal(f"  {note.source.value}: {note.reason}"))
    return 0
