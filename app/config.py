"""
Application configuration loaded from environment variables.

Uses python-dotenv to load .env file and provides typed configuration
for the LLM provider and simulation parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


class Settings:
    """Typed configuration container for the simulation engine."""

    # LLM Provider
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()

    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    # Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Simulation
    max_ticks: int = 15
    num_citizens: int = 3
    zones: list[str] = None

    # Logging
    log_dir: str = "logs"
    log_file: str = "run.jsonl"

    def __init__(self):
        if self.zones is None:
            self.zones = ["Residential_Sector", "Market_District", "City_Park"]

    @property
    def log_path(self) -> Path:
        """Return the full path to the log file."""
        return Path(self.log_dir) / self.log_file


settings = Settings()