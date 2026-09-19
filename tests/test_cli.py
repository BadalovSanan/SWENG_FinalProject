"""Offline CLI tests with mocked application boundaries."""

from __future__ import annotations

from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import AsyncMock, Mock, call

import pytest

from ai.schemas import AnswerWithCitations, Citation
from src import cli
from src.config import Settings
from src.core.researcher import ResearchAssistant
from src.exceptions import AllSourcesFailedError, InvalidQuestionError
from src.models import DegradationNote, MAX_QUESTION_LENGTH, SourceName


@pytest.fixture
def assistant(monkeypatch, sample_sources):
    fake = Mock(spec=ResearchAssistant)
    fake.ask = AsyncMock(return_value=AnswerWithCitations(
        question="What is photosynthesis?",
        answer="Plants convert light into energy [1], [2].",
        citations=[
            Citation(index=2, source=sample_sources[1]),
            Citation(index=1, source=sample_sources[0]),
        ],
    ))
    fake.last_degradation_notes = []
    monkeypatch.setattr(cli, "create_assistant", Mock(return_value=fake))
    return fake


def test_success_renders_answer_and_sorted_references(assistant, capsys):
    assert cli.main(["ask", "What is photosynthesis?"]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out == (
        "Plants convert light into energy [1], [2].\n\nReferences:\n"
        "  [1] (wikipedia) Photosynthesis (Wikipedia)\n"
        "      https://en.wikipedia.org/wiki/Photosynthesis\n"
        "  [2] (wikipedia) Calvin cycle (Wikipedia)\n"
        "      https://en.wikipedia.org/wiki/Calvin_cycle\n"
    )


@pytest.mark.parametrize(("flags", "sources", "use_cache"), [
    ([], None, True),
    (["--sources", "wiki,arxiv"], [SourceName.WIKI, SourceName.ARXIV], True),
    (["--no-cache"], None, False),
    (["--sources", "web,wiki", "--no-cache"], [SourceName.WEB, SourceName.WIKI], False),
    (["--sources", " arxiv , wiki "], [SourceName.ARXIV, SourceName.WIKI], True),
])
def test_ask_arguments(assistant, flags, sources, use_cache):
    assert cli.main(["ask", "Question?", *flags]) == 0
    assistant.ask.assert_awaited_once_with(
        "Question?", sources=sources, use_cache=use_cache
    )


@pytest.mark.parametrize("question", ["", "   ", "a" * (MAX_QUESTION_LENGTH + 1)])
def test_question_validation_stays_in_business_layer(monkeypatch, capsys, question):
    service = AsyncMock()
    cache = Mock()
    orchestrator = AsyncMock()
    real_assistant = ResearchAssistant(
        service, cache, orchestrator, Settings(_env_file=None)
    )
    monkeypatch.setattr(cli, "create_assistant", lambda: real_assistant)

    assert cli.main(["ask", question]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "research request is invalid" in output.err
    assert "Traceback" not in output.err
    assert cache.mock_calls == []
    assert orchestrator.mock_calls == []
    assert service.mock_calls == []


def test_source_deduplication_stays_in_business_layer(monkeypatch, assistant, sample_sources):
    service = AsyncMock()
    service.synthesize.return_value = assistant.ask.return_value
    orchestrator = AsyncMock()
    orchestrator.gather_sources.return_value = (sample_sources, [])
    real_assistant = ResearchAssistant(
        service, Mock(), orchestrator, Settings(_env_file=None)
    )
    monkeypatch.setattr(cli, "create_assistant", lambda: real_assistant)

    assert cli.main(["ask", "Question?", "--sources", "arxiv,wiki,arxiv", "--no-cache"]) == 0
    orchestrator.gather_sources.assert_awaited_once_with(
        "question", [SourceName.ARXIV, SourceName.WIKI]
    )


def test_all_sources_failed_is_clean_error(assistant, capsys):
    assistant.ask.side_effect = AllSourcesFailedError("No research sources were available.")
    assert cli.main(["ask", "Question?"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "researcher: error: No research sources were available.\n"


def test_degradation_notes_are_read_after_ask(assistant, capsys):
    async def ask(*args, **kwargs):
        assistant.last_degradation_notes = [
            DegradationNote(source=SourceName.ARXIV, reason="Source timed out."),
            DegradationNote(source=SourceName.WEB, reason="Source failed with RuntimeError."),
        ]
        return assistant.ask.return_value

    assistant.ask.side_effect = ask
    assert cli.main(["ask", "Question?"]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert "Plants convert light" in output.out
    assert "answer uses the remaining sources" in output.out
    assert "arxiv: Source timed out." in output.out
    assert "web: Source failed with RuntimeError." in output.out


@pytest.mark.parametrize("sources", [
    "", " ", ",", "wiki,", ",wiki", "wiki,,arxiv", "wiki, ,web",
    "unknown", "wiki,unknown", "wiki arxiv", "wiki;arxiv", "WIKI",
])
def test_invalid_source_filters(assistant, capsys, sources):
    with pytest.raises(SystemExit) as error:
        cli.main(["ask", "Question?", "--sources", sources])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "sources must be comma-separated" in output.err
    assert "Traceback" not in output.err
    cli.create_assistant.assert_not_called()


@pytest.mark.parametrize("args", [[], ["ask"], ["ask", "Question?", "--sources"]])
def test_required_arguments(assistant, capsys, args):
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    cli.create_assistant.assert_not_called()


def test_sanitizes_answer_references_and_notes(assistant, capsys):
    source = assistant.ask.return_value.citations[0].source.model_copy(update={
        "title": "\x1b[31mCafé — 植物 🌱\x1b[0m\x07",
        "url": "https://example.com/植物?q=a%20b&lang=az#part[1]\x08",
    })
    assistant.ask.return_value = AnswerWithCitations(
        question="Question?",
        answer="\x1b]0;unsafe title\x07\x1b[32mCafé [1]\x1b[0m\r\x00\n\t植物 🌱",
        citations=[Citation(index=1, source=source)],
    )
    assistant.last_degradation_notes = [
        DegradationNote(source=SourceName.ARXIV, reason="\x9b31mTimed out.\x9b0m")
    ]
    assert cli.main(["ask", "Question?"]) == 0
    output = capsys.readouterr().out
    assert "Café [1]\n\t植物 🌱" in output
    assert "Café — 植物 🌱" in output
    assert "https://example.com/植物?q=a%20b&lang=az#part[1]" in output
    assert "arxiv: Timed out." in output
    assert "unsafe title" not in output
    assert not any(char in output for char in "\x1b\x00\x07\x08\r\x9b")


@pytest.mark.parametrize("wrapped", [
    "\x1b]8;;https://example.com\x1b\\Visible [1]\x1b]8;;\x1b\\",
    "\x9d8;;https://example.com\x9cVisible [1]\x9d8;;\x9c",
    "\x1bPunsafe\x1b\\Visible [1]\x1b(B",
    "Visible [1]\x1b]0;unfinished",
])
def test_sanitizes_terminal_strings(wrapped):
    assert cli._sanitize_terminal(wrapped) == "Visible [1]"


def test_sanitizes_domain_error(assistant, capsys):
    assistant.ask.side_effect = InvalidQuestionError("\x1b[31mInvalid question.\x1b[0m")
    assert cli.main(["ask", "Question?"]) == 2
    assert capsys.readouterr().err == "researcher: error: Invalid question.\n"


def test_unexpected_errors_are_not_hidden(assistant):
    assistant.ask.side_effect = RuntimeError("programmer error")
    with pytest.raises(RuntimeError, match="programmer error"):
        cli.main(["ask", "Question?"])


def test_factory_wires_existing_components(monkeypatch, tmp_path):
    settings = Settings(_env_file=None, cache_dir=tmp_path, cache_ttl_seconds=123)
    wiring = Mock()
    names = [
        "get_settings", "configure_logging", "AIService", "FileSystemCacheStore",
        "SourceOrchestrator", "ResearchAssistant",
    ]
    for name in names:
        monkeypatch.setattr(cli, name, getattr(wiring, name))
    wiring.get_settings.return_value = settings

    assert cli.create_assistant() is wiring.ResearchAssistant.return_value
    assert wiring.mock_calls == [
        call.get_settings(),
        call.configure_logging(settings),
        call.AIService(settings),
        call.FileSystemCacheStore(tmp_path, 123),
        call.SourceOrchestrator(wiring.AIService.return_value, settings),
        call.ResearchAssistant(
            wiring.AIService.return_value, wiring.FileSystemCacheStore.return_value,
            wiring.SourceOrchestrator.return_value, settings,
        ),
    ]


@pytest.mark.parametrize("package", ["researcher", "src"])
@pytest.mark.parametrize("flags", [["--help"], ["ask", "--help"]])
def test_module_help_is_offline(package, flags):
    result = subprocess.run(
        [sys.executable, "-m", package, *flags],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0
    assert "usage: researcher" in result.stdout
    assert "ask" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("package", ["researcher", "src"])
def test_entrypoint_delegates_and_propagates_exit_code(monkeypatch, package):
    main = Mock(return_value=2)
    monkeypatch.setattr(cli, "main", main)
    with pytest.raises(SystemExit) as error:
        runpy.run_module(package, run_name="__main__")
    assert error.value.code == 2
    main.assert_called_once_with()
