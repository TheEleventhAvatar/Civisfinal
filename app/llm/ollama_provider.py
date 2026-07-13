"""
Ollama LLM provider implementation.

Communicates with a local Ollama instance via its REST API.
"""

import json
from typing import Any

import requests

from app.llm.provider import LLMProvider
from app.config import settings


class OllamaProvider(LLMProvider):
    """Provider that calls a local Ollama instance."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 60,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to Ollama and return the generated text.

        Args:
            prompt: The full prompt string.

        Returns:
            The raw text response.

        Raises:
            ConnectionError: If the Ollama server is unreachable.
            RuntimeError: If the API returns an error.
        """
        url = f"{self.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is the server running?"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama API error: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from Ollama: {exc}") from exc