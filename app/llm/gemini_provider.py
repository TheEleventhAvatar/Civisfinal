"""
Gemini LLM provider implementation.

Communicates with Google's Gemini API via the google-generativeai library.
"""

from app.llm.provider import LLMProvider
from app.config import settings


class GeminiProvider(LLMProvider):
    """Provider that calls Google's Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model or settings.gemini_model

        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY in .env"
            )

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to Gemini and return the generated text.

        Args:
            prompt: The full prompt string.

        Returns:
            The raw text response.

        Raises:
            ConnectionError: If the API is unreachable.
            RuntimeError: If the API returns an error.
        """
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Gemini API error: {exc}") from exc