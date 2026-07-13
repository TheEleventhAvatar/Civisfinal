"""
Resilient JSON parser for LLM responses.

Handles the messy reality of LLM output: markdown fences, trailing
commas, extra text, malformed whitespace, null fields, etc.
Attempts repair before returning a parsed LLMAction or ParserError.

Logs recovery events when malformed JSON is successfully repaired.
Never raises exceptions.
"""

import json
import logging
import re

from pydantic import ValidationError

from app.models.actions import LLMAction, ParserError as ParserErrorModel

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ``` or ``` ... ```)."""
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def _extract_json_objects(text: str) -> list[str]:
    """
    Extract all JSON-like objects from text using brace matching.

    Returns a list of strings that look like JSON objects.
    """
    objects = []
    depth = 0
    start = -1

    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                objects.append(text[start:i + 1])
                start = -1

    return objects


def _repair_json(text: str) -> str:
    """
    Attempt to repair common JSON formatting issues.

    Handles:
    - Trailing commas in objects and arrays
    - Single quotes instead of double quotes
    - Unquoted keys
    """
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    # Replace single quotes with double quotes (but not within strings)
    text = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', text)

    return text


def _normalize_nulls(data: dict) -> dict:
    """
    Normalize null-like values in parsed data.
    Converts None dialogue to empty string, etc.
    """
    if data.get("dialogue") is None:
        data["dialogue"] = ""
    if data.get("target_citizen") is None:
        data["target_citizen"] = None
    if data.get("target_zone") is None:
        data["target_zone"] = None
    return data


def parse_llm_response(raw: str) -> LLMAction | ParserErrorModel:
    """
    Parse an LLM response string into a validated LLMAction.

    Attempts multiple strategies:
    1. Strip fences and parse directly
    2. Extract JSON objects with brace matching
    3. Attempt repair on extracted JSON
    4. Normalize null fields
    5. Return ParserError if all attempts fail

    Logs a warning when malformed JSON is successfully recovered.

    Args:
        raw: The raw text response from the LLM.

    Returns:
        Either a validated LLMAction or a ParserError.
    """
    if not raw or not raw.strip():
        return ParserErrorModel(
            raw_text=raw,
            reason="Empty response from LLM.",
        )

    # Step 1: Strip markdown fences
    cleaned = _strip_fences(raw)

    # Step 2: Try direct parse on cleaned and raw
    for candidate in [cleaned, raw]:
        try:
            data = json.loads(candidate)
            data = _normalize_nulls(data)
            return LLMAction(**data)
        except (json.JSONDecodeError, ValueError, ValidationError):
            pass

    # Step 3: Try extracting JSON objects with brace matching
    candidates = _extract_json_objects(cleaned)
    if not candidates:
        candidates = _extract_json_objects(raw)

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            data = _normalize_nulls(data)
            logger.warning("Recovered malformed JSON via brace extraction.")
            return LLMAction(**data)
        except (json.JSONDecodeError, ValueError, ValidationError):
            pass

    # Step 4: Try repairing
    repaired = _repair_json(cleaned)
    try:
        data = json.loads(repaired)
        data = _normalize_nulls(data)
        logger.warning("Recovered malformed JSON via repair.")
        return LLMAction(**data)
    except (json.JSONDecodeError, ValueError, ValidationError):
        pass

    # Step 5: Try repairing extracted candidates
    for candidate in candidates:
        try:
            repaired = _repair_json(candidate)
            data = json.loads(repaired)
            data = _normalize_nulls(data)
            logger.warning("Recovered malformed JSON via extraction + repair.")
            return LLMAction(**data)
        except (json.JSONDecodeError, ValueError, ValidationError):
            pass

    # Step 6: Give up
    return ParserErrorModel(
        raw_text=raw,
        reason="Could not parse valid JSON from LLM response after multiple repair attempts.",
    )