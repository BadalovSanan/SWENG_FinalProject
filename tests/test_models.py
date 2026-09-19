"""Tests for research request models and SE-layer exceptions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.exceptions import (
    AllSourcesFailedError,
    CacheError,
    InvalidQuestionError,
    ResearchError,
)
from src.models import (
    MAX_QUESTION_LENGTH,
    CacheKey,
    DegradationNote,
    ResearchRequest,
    SourceName,
    canonicalize_query,
)


def test_research_request_accepts_and_strips_valid_question() -> None:
    """A valid question is retained after surrounding whitespace is removed."""

    request = ResearchRequest(question="  What is photosynthesis?  ")

    assert request.question == "What is photosynthesis?"


@pytest.mark.parametrize("question", ["", "   "])
def test_research_request_rejects_empty_question(question: str) -> None:
    """Empty and whitespace-only questions are invalid."""

    with pytest.raises(ValidationError):
        ResearchRequest(question=question)


def test_research_request_enforces_question_length_limit() -> None:
    """The documented maximum length is inclusive."""

    assert ResearchRequest(question="a" * MAX_QUESTION_LENGTH).question
    with pytest.raises(ValidationError):
        ResearchRequest(question="a" * (MAX_QUESTION_LENGTH + 1))


def test_research_request_accepts_valid_sources() -> None:
    """Known CLI source values are converted to SourceName members."""

    request = ResearchRequest(question="Q", sources=["wiki", "arxiv"])

    assert request.sources == [SourceName.WIKI, SourceName.ARXIV]


def test_research_request_rejects_invalid_source_name() -> None:
    """Only known CLI source names are accepted."""

    with pytest.raises(ValidationError):
        ResearchRequest(question="Q", sources=["wikipedia"])


def test_research_request_rejects_empty_sources() -> None:
    """An explicit source filter must contain at least one source."""

    with pytest.raises(ValidationError):
        ResearchRequest(question="Q", sources=[])


def test_research_request_deduplicates_sources_in_order() -> None:
    """The first occurrence of each requested source is kept."""

    request = ResearchRequest(
        question="Q",
        sources=["web", "wiki", "web", "arxiv", "wiki"],
    )

    assert request.sources == [SourceName.WEB, SourceName.WIKI, SourceName.ARXIV]


def test_research_request_keeps_none_sources() -> None:
    """None retains the meaning that all sources should be used."""

    assert ResearchRequest(question="Q").sources is None


def test_research_request_is_immutable() -> None:
    """Frozen request models cannot be changed after validation."""

    request = ResearchRequest(question="Q")

    with pytest.raises(ValidationError):
        request.question = "Changed"  # type: ignore[misc]


def test_create_wraps_validation_error() -> None:
    """The domain entry point hides Pydantic details but preserves the cause."""

    with pytest.raises(InvalidQuestionError) as error_info:
        ResearchRequest.create("   ")

    assert isinstance(error_info.value.__cause__, ValidationError)
    assert str(error_info.value) == "The research request is invalid."


def test_create_returns_valid_research_request() -> None:
    """The domain entry point returns a validated request on success."""

    request = ResearchRequest.create("Q", ["wiki"])

    assert request.sources == [SourceName.WIKI]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("WHAT IS photosynthesis?", "what is photosynthesis"),
        (" what is  photosynthesis ", "what is photosynthesis"),
        ("what\tis\nphotosynthesis!", "what is photosynthesis"),
        ("", ""),
    ],
)
def test_canonicalize_query(query: str, expected: str) -> None:
    """Case, whitespace, and trailing punctuation are normalized consistently."""

    assert canonicalize_query(query) == expected


def test_canonicalize_query_equates_different_looking_queries() -> None:
    """Equivalent user input produces the same cacheable query."""

    assert canonicalize_query("WHAT IS photosynthesis?") == canonicalize_query(
        "what is  photosynthesis"
    )


def test_cache_key_uses_canonical_query_for_equality_and_hashing() -> None:
    """Equivalent raw queries create equal, identically hashed cache keys."""

    first = CacheKey.from_raw(SourceName.WIKI, "WHAT IS photosynthesis?")
    second = CacheKey.from_raw(SourceName.WIKI, "what is  photosynthesis")

    assert first == second
    assert hash(first) == hash(second)


def test_cache_key_distinguishes_sources() -> None:
    """The source remains part of the cache key identity."""

    wiki_key = CacheKey.from_raw(SourceName.WIKI, "Q")
    web_key = CacheKey.from_raw(SourceName.WEB, "Q")

    assert wiki_key != web_key


def test_degradation_note_accepts_valid_data() -> None:
    """A source failure reason can be recorded for the CLI."""

    note = DegradationNote(source=SourceName.ARXIV, reason="Timed out")

    assert note.reason == "Timed out"


def test_degradation_note_rejects_empty_reason() -> None:
    """A degradation note must explain the unavailable source."""

    with pytest.raises(ValidationError):
        DegradationNote(source=SourceName.ARXIV, reason="")


@pytest.mark.parametrize(
    "exception_type",
    [InvalidQuestionError, AllSourcesFailedError, CacheError],
)
def test_domain_exceptions_share_research_error_base(
    exception_type: type[ResearchError],
) -> None:
    """All domain errors can be handled through their common base class."""

    assert issubclass(exception_type, ResearchError)
