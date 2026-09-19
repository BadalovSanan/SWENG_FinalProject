"""Offline integration coverage for every provided question and the demo output."""

from __future__ import annotations

import json
from pathlib import Path
import re
import socket
from unittest.mock import Mock

import pytest

import ai
from ai.schemas import AnswerWithCitations, Citation, Source
from demo_ai import _OfflineSources
from scripts import demo
from src.concurrency.orchestrator import SourceOrchestrator
from src.config import Settings
from src.core.researcher import ResearchAssistant
from src.models import canonicalize_query, SourceName
from src.services.ai_service import AIService
from src.storage.cache_store import FileSystemCacheStore

QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "research_questions.json"
QUESTIONS = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    """Fail even if application degradation handling catches an attempted request."""

    blocked = Mock(side_effect=AssertionError("Network access is forbidden in this test."))
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    yield
    blocked.assert_not_called()


def make_assistant(cache_dir: Path) -> ResearchAssistant:
    """Wire real components, including disk cache, with explicit test settings."""

    settings = Settings(
        _env_file=None, cache_dir=cache_dir, cache_ttl_seconds=60,
        per_source_timeout_seconds=2, max_sources_per_query=3,
    )
    service = AIService(settings, backoff_min=0, backoff_max=0)
    return ResearchAssistant(
        service,
        FileSystemCacheStore(settings.cache_dir, settings.cache_ttl_seconds),
        SourceOrchestrator(service, settings),
        settings,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("question", QUESTIONS, ids=lambda item: item["id"])
async def test_question_pipeline_and_persistent_cache(question, tmp_path, fake_llm):
    """A fresh assistant reuses disk sources while real synthesis runs again."""

    fake_llm.response = "Offline integration answer from the supplied reference [1]."
    assistant = make_assistant(tmp_path)
    query = canonicalize_query(question["text"])
    expected_sources = [
        item
        for source in SourceName
        for item in _OfflineSources.fetch(query)
        if item.origin == source.origin
    ]

    with demo.offline_providers(fake_llm) as fetchers:
        result = await assistant.ask(question["text"])

        assert isinstance(result, AnswerWithCitations)
        assert result.question == question["text"]
        assert result.answer == fake_llm.response
        assert result.citations
        assert len(fake_llm.calls) == 1
        assert question["text"] in fake_llm.calls[0]
        for source in expected_sources:
            assert source.title in fake_llm.calls[0]
        for citation in result.citations:
            assert isinstance(citation, Citation)
            assert isinstance(citation.source, Source)
            assert 1 <= citation.index <= len(expected_sources)
            assert citation.source == expected_sources[citation.index - 1]
            assert citation.source.url.startswith("https://")
        assert {c.index for c in result.citations} == {
            int(index) for index in re.findall(r"\[(\d+)\]", result.answer)
        }
        assert assistant.last_degradation_notes == []
        for fetch in fetchers.values():
            fetch.assert_awaited_once()
            assert fetch.await_args.args == (query,)

        assert len(list(tmp_path.glob("*.json"))) == len(SourceName)
        # A different instance must read persisted cache entries, including empty lists.
        cached_result = await make_assistant(tmp_path).ask(question["text"])
        assert cached_result.model_dump() == result.model_dump()
        assert len(fake_llm.calls) == 2
        for fetch in fetchers.values():
            fetch.assert_awaited_once()


@pytest.mark.asyncio
async def test_demo_writes_five_valid_reproducible_artefacts(tmp_path):
    """Rerun the real demo in the same directory without changing unrelated files."""

    output = tmp_path / "artefacts"
    output.mkdir()
    unrelated = output / "notes.txt"
    unrelated.write_text("Keep this file.\n", encoding="utf-8")
    original_fetch = ai.fetch_wikipedia
    paths = await demo.run_demo(output)
    first_contents = {path.name: path.read_bytes() for path in paths}
    assert set(first_contents) == {f"{item['id']}_offline.json" for item in QUESTIONS}
    repeated = await demo.run_demo(output)

    assert len(QUESTIONS) == len(paths) == 5
    assert repeated == paths
    assert {path.name for path in output.iterdir()} == set(first_contents) | {"notes.txt"}
    assert ai.fetch_wikipedia is original_fetch
    assert unrelated.read_text(encoding="utf-8") == "Keep this file.\n"
    for question, path in zip(QUESTIONS, paths):
        assert path.name == f"{question['id']}_offline.json"
        assert path.read_bytes() == first_contents[path.name]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["question_id"] == question["id"]
        assert payload["question"] == question["text"]
        assert payload["mode"] == "offline"
        assert payload["demo_note"] == demo.DEMO_NOTE
        result = AnswerWithCitations.model_validate({
            key: payload[key] for key in ("question", "answer", "citations")
        })
        assert result.answer.strip()
        assert result.citations
        assert payload["degradation_notes"] == []


@pytest.mark.asyncio
async def test_demo_refuses_to_follow_output_symlinks(tmp_path):
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("Unrelated existing content", encoding="utf-8")
    existing = tmp_path / f"{QUESTIONS[-1]['id']}_offline.json"
    existing.symlink_to(unrelated)
    with pytest.raises(FileExistsError, match="symlink; no files were overwritten"):
        await demo.run_demo(tmp_path)
    assert unrelated.read_text(encoding="utf-8") == "Unrelated existing content"
    assert existing.is_symlink()
    assert set(tmp_path.iterdir()) == {existing, unrelated}
