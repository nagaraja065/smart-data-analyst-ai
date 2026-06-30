"""
LLM Providers Sub-package — Pluggable AI Backends.

Provides a uniform interface for LLM interactions through the Strategy pattern.
Each provider implements LLMProvider ABC, enabling hot-swap between Claude API
and a fully-offline local pattern-matching engine.

Design Pattern: Strategy — interchangeable LLM backends behind a common interface.
SOLID: Open/Closed — add new providers without modifying existing code.
"""

from ai.providers.base import LLMProvider, LLMResponse
from ai.providers.claude_provider import ClaudeProvider
from ai.providers.local_provider import LocalProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ClaudeProvider",
    "LocalProvider",
]
