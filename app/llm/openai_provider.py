"""
OpenAI LLM provider implementation.

Communicates with OpenAI's API via the openai Python library.
"""

from app.llm.provider import LLMProvider
from app.config import settings


class OpenAIProvider(LLMProvider):
    """Provider that calls OpenAI's API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or settings.openai_api_key
        self.model_name = model or settings.openai_model

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY in .env"
            )

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to OpenAI and return the generated text.

        Args:
            prompt: The full prompt string.

        Returns:
            The raw text response.

        Raises:
            ConnectionError: If the API is unreachable.
            RuntimeError: If the API returns an error.
        """
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except ImportError as exc:
            raise RuntimeError(
                "openai is not installed. Run: pip install openai"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc