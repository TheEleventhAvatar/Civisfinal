"""
Abstract base class for LLM providers.

Defines the interface that all LLM providers must implement.
This abstraction allows the simulation to run with any backend
(Ollama, Gemini, or a mock for testing).
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface for language model providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the raw text response.

        Args:
            prompt: The full prompt string to send.

        Returns:
            The raw text response from the LLM.

        Raises:
            ConnectionError: If the provider cannot be reached.
            RuntimeError: If the provider returns an error.
        """
        ...