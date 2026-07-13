"""
Logging utilities for the simulation engine.

Provides structured JSONL logging and console output with
configurable verbosity levels. Each tick shows the full
pipeline: Raw LLM → Parsed → Validated → Mutations → State.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import settings


class SimulationLogger:
    """Handles structured logging to both console and JSONL file."""

    def __init__(self, log_path: Path | None = None):
        self.log_path = log_path or settings.log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear previous log file
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("")

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        """Append a JSON record to the log file."""
        record["_timestamp"] = datetime.utcnow().isoformat()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _console(self, message: str, level: str = "INFO") -> None:
        """Print a formatted message to stderr."""
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}", file=sys.stderr)

    def tick_start(self, tick: int) -> None:
        """Log the start of a simulation tick."""
        self._console(f"{'='*50}", level="")
        self._console(f"TICK {tick}", level="")
        self._console(f"{'='*50}", level="")
        self._write_jsonl({"event": "tick_start", "tick": tick})

    def prompt(self, tick: int, prompt_text: str) -> None:
        """Log the prompt sent to the LLM."""
        self._console(f"Prompt ({len(prompt_text)} chars)", level="PROMPT")
        self._write_jsonl({
            "event": "prompt",
            "tick": tick,
            "prompt": prompt_text,
        })

    def raw_response(self, tick: int, response: str) -> None:
        """Log the raw LLM response."""
        self._console(f"Raw LLM → {response[:150]}...", level="LLM")
        self._write_jsonl({
            "event": "raw_response",
            "tick": tick,
            "response": response,
        })

    def parsed_action(self, tick: int, action: dict[str, Any] | None) -> None:
        """Log the parsed action from the LLM."""
        self._console(f"Parsed ↓ {action}", level="PARSE")
        self._write_jsonl({
            "event": "parsed_action",
            "tick": tick,
            "action": action,
        })

    def validation_result(self, tick: int, result: dict[str, Any]) -> None:
        """Log the validation result."""
        status = "✓ ACCEPTED" if result.get("accepted") else "✗ REJECTED"
        self._console(f"Validation ↓ {status} — {result.get('reason', '')}", level="VALIDATE")
        if not result.get("accepted"):
            safe = result.get("safe_action", {})
            self._console(f"  Fallback → {safe}", level="FALLBACK")
        self._write_jsonl({
            "event": "validation",
            "tick": tick,
            "result": result,
        })

    def mutations(self, tick: int, mutations: list[dict[str, Any]]) -> None:
        """Log the mutations applied."""
        self._console(f"Mutations ({len(mutations)}) ↓", level="MUTATE")
        action_mutations = [m for m in mutations if m.get("type") != "EnergyMutation"]
        energy_mutations = [m for m in mutations if m.get("type") == "EnergyMutation"]
        if energy_mutations:
            self._console(f"  Energy decay: {len(energy_mutations)} citizens", level="MUTATE")
        for m in action_mutations:
            self._console(f"  {m.get('type')}: {m.get('description', '')}", level="MUTATE")
        self._write_jsonl({
            "event": "mutations",
            "tick": tick,
            "mutations": [dict(m) for m in mutations],
        })

    def state_snapshot(self, tick: int, state: dict[str, Any]) -> None:
        """Log the final state snapshot for a tick."""
        self._write_jsonl({
            "event": "state_snapshot",
            "tick": tick,
            "state": state,
        })

    def chronicle(self, tick: int, entry: str) -> None:
        """Log a chronicle entry."""
        self._console(f"Chronicle: {entry}", level="CHRONICLE")
        self._write_jsonl({
            "event": "chronicle",
            "tick": tick,
            "entry": entry,
        })

    def update_state(self, tick: int) -> None:
        """Log a concise state update line."""
        # Called from simulator after mutations
        pass

    def error(self, tick: int, message: str) -> None:
        """Log an error."""
        self._console(f"ERROR: {message}", level="ERROR")
        self._write_jsonl({
            "event": "error",
            "tick": tick,
            "message": message,
        })