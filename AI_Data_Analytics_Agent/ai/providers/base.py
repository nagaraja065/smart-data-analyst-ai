"""
Abstract LLM Provider — Uniform Interface for AI Backends.

Defines the contract that every LLM provider must fulfill.  Concrete providers
(Claude API, local pattern engine, future OpenAI, etc.) implement `generate`
and `is_available` so the rest of the system is completely backend-agnostic.

Design Pattern: Strategy — callers program to the LLMProvider interface.
SOLID: Dependency Inversion — high-level agent depends on this abstraction,
       not on concrete API clients.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from core.logger import get_logger

logger = get_logger(__name__)


# ─── Result Dataclass ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LLMResponse:
    """Immutable container for an LLM generation result.

    Attributes:
        text:       The generated text content.
        model:      Model identifier that produced the response.
        tokens_used: Total token count (prompt + completion) if available.
        latency_ms: Wall-clock latency of the API call in milliseconds.
    """
    text: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0


# ─── Abstract Provider ───────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Abstract base class for all LLM providers.

    Every concrete provider MUST implement:
        * ``generate``    — produce text from a prompt + system instructions.
        * ``is_available`` — report whether the backend is reachable/configured.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt:        The user-facing prompt / question.
            system_prompt: Optional system-level instructions for the model.
            max_tokens:    Upper bound on tokens to generate.
            temperature:   Sampling temperature (0 = deterministic).

        Returns:
            LLMResponse with the generated text and metadata.

        Raises:
            LLMProviderError: If the backend call fails.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether this provider is configured and reachable.

        Returns:
            True if the provider can accept generate() calls right now.
        """

    # ── Convenience helpers ──────────────────────────────────────────────

    @staticmethod
    def _measure_latency(start: float) -> float:
        """Return elapsed milliseconds since *start* (``time.perf_counter``)."""
        return round((time.perf_counter() - start) * 1000, 2)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} available={self.is_available()}>"
