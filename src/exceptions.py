"""Domain exceptions for the software-engineering layer."""

from __future__ import annotations


class ResearchError(Exception):
    """Base exception raised by the research assistant's SE-layer code."""


class InvalidQuestionError(ResearchError):
    """Raised by ResearchRequest.create when request validation fails."""


class AllSourcesFailedError(ResearchError):
    """Raised by the research service when every requested source fails."""


class CacheError(ResearchError):
    """Raised by the cache store when a cache operation cannot complete."""
