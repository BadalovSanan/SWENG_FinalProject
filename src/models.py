"""Validated value objects for the research assistant's SE layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from src.exceptions import InvalidQuestionError


# Maximum permitted length of a stripped research question.
MAX_QUESTION_LENGTH = 2_000


class SourceName(str, Enum):
    """Source names accepted by the CLI and SE layer."""

    WIKI = "wiki"
    ARXIV = "arxiv"
    WEB = "web"

    @property
    def origin(self) -> str:
        """Return the corresponding origin value used by ``ai.schemas.Source``."""

        if self is SourceName.WIKI:
            return "wikipedia"
        return self.value


def canonicalize_query(query: str) -> str:
    """Return a lowercase, whitespace-normalized query without trailing punctuation."""

    return " ".join(query.lower().split()).rstrip("?.!")


class ResearchRequest(BaseModel):
    """An immutable, validated research question and optional source filter."""

    model_config = ConfigDict(frozen=True)

    question: str
    sources: list[SourceName] | None = None

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        """Strip and validate a supplied research question."""

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("question must be non-empty")
        if len(stripped_value) > MAX_QUESTION_LENGTH:
            raise ValueError(
                f"question must not exceed {MAX_QUESTION_LENGTH} characters"
            )
        return stripped_value

    @field_validator("sources")
    @classmethod
    def _validate_sources(cls, value: list[SourceName] | None) -> list[SourceName] | None:
        """Reject empty filters and preserve the first occurrence of each source."""

        if value is None:
            return None
        if not value:
            raise ValueError("sources must not be empty")
        return list(dict.fromkeys(value))

    @classmethod
    def create(
        cls,
        question: str,
        sources: list[SourceName | str] | None = None,
    ) -> ResearchRequest:
        """Build a request, raising a domain error when validation fails."""

        try:
            return cls.model_validate({"question": question, "sources": sources})
        except ValidationError as error:
            raise InvalidQuestionError("The research request is invalid.") from error


@dataclass(frozen=True)
class CacheKey:
    """A hashable cache key consisting of a source and canonical query."""

    source: SourceName
    query: str

    @classmethod
    def from_raw(cls, source: SourceName, query: str) -> CacheKey:
        """Create a cache key from a source and its unnormalized query."""

        return cls(source=source, query=canonicalize_query(query))


class DegradationNote(BaseModel):
    """An immutable record explaining why a source was unavailable."""

    model_config = ConfigDict(frozen=True)

    source: SourceName
    reason: str

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        """Strip and require a non-empty degradation reason."""

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("reason must be non-empty")
        return stripped_value
