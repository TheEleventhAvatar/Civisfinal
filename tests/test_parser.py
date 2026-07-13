"""
Tests for the LLM response parser.

Covers markdown code fences, extra text, malformed commas,
invalid schema, edge cases, and null field normalization.
"""

import pytest

from app.llm.parser import parse_llm_response, _strip_fences, _repair_json, _extract_json_objects
from app.models.actions import LLMAction, ActionType, ParserError


class TestStripFences:
    """Test markdown code fence removal."""

    def test_no_fences(self):
        """Plain JSON without fences should be unchanged."""
        text = '{"action": "MOVE", "target_zone": "City_Park"}'
        result = _strip_fences(text)
        assert result == text

    def test_json_fences(self):
        """```json ... ``` fences should be stripped."""
        text = '```json\n{"action": "MOVE"}\n```'
        result = _strip_fences(text)
        assert result == '{"action": "MOVE"}'

    def test_code_fences(self):
        """``` ... ``` fences should be stripped."""
        text = '```\n{"action": "CONVERSE"}\n```'
        result = _strip_fences(text)
        assert result == '{"action": "CONVERSE"}'

    def test_multiline_fences(self):
        """Fences with leading/trailing newlines should be handled."""
        text = '\n\n```json\n{"action": "GIFT", "target_citizen": "Bob"}\n```\n\n'
        result = _strip_fences(text)
        assert result == '{"action": "GIFT", "target_citizen": "Bob"}'


class TestExtractJsonObjects:
    """Test JSON object extraction from mixed text."""

    def test_single_object(self):
        """A single JSON object should be extracted."""
        text = 'Some text {"action": "MOVE"} trailing'
        objects = _extract_json_objects(text)
        assert len(objects) == 1
        assert '"action": "MOVE"' in objects[0]

    def test_nested_braces(self):
        """Nested objects should be correctly matched."""
        text = '{"outer": {"inner": "value"}}'
        objects = _extract_json_objects(text)
        assert len(objects) == 1
        assert objects[0] == text

    def test_no_objects(self):
        """Text without braces should return empty list."""
        objects = _extract_json_objects("Just some text here")
        assert len(objects) == 0


class TestRepairJson:
    """Test JSON repair logic."""

    def test_trailing_comma_object(self):
        """Trailing commas in objects should be removed."""
        text = '{"action": "MOVE", "target_zone": "City_Park",}'
        result = _repair_json(text)
        assert ",}" not in result
        import json
        data = json.loads(result)
        assert data["action"] == "MOVE"

    def test_trailing_comma_array(self):
        """Trailing commas in arrays should be removed."""
        text = '{"items": [1, 2, 3,]}'
        result = _repair_json(text)
        assert ",]" not in result

    def test_single_quoted_keys(self):
        """Single-quoted property names should be converted."""
        text = "{'action': 'MOVE'}"
        result = _repair_json(text)
        assert "'action'" not in result
        assert '"action"' in result


class TestParseLLMResponse:
    """Test the full parse_llm_response function."""

    def test_parse_valid_json(self):
        """Valid JSON should parse correctly."""
        raw = '{"action": "MOVE", "target_zone": "City_Park", "dialogue": ""}'
        result = parse_llm_response(raw)
        assert isinstance(result, LLMAction)
        assert result.action == ActionType.MOVE
        assert result.target_zone == "City_Park"

    def test_fenced_json(self):
        """JSON inside ```json fences should parse."""
        raw = '```json\n{"action": "CONVERSE", "dialogue": "Hello!"}\n```'
        result = parse_llm_response(raw)
        assert isinstance(result, LLMAction)
        assert result.action == ActionType.CONVERSE
        assert result.dialogue == "Hello!"

    def test_extra_markdown_before(self):
        """Extra text before JSON should be handled."""
        raw = 'Here is the response:\n{"action": "GIFT", "target_citizen": "Bob", "dialogue": "Take this."}'
        result = parse_llm_response(raw)
        assert isinstance(result, LLMAction)
        assert result.action == ActionType.GIFT

    def test_trailing_commas(self):
        """JSON with trailing commas should be repaired."""
        raw = '{"action": "STEAL", "target_citizen": "Alice", "dialogue": "",}'
        result = parse_llm_response(raw)
        assert isinstance(result, LLMAction)
        assert result.action == ActionType.STEAL

    def test_malformed_whitespace(self):
        """JSON with extra whitespace should parse."""
        raw = '{\n  "action": "MOVE",\n  "target_zone": "Market_District"\n}'
        result = parse_llm_response(raw)
        assert isinstance(result, LLMAction)
        assert result.action == ActionType.MOVE
        assert result.target_zone == "Market_District"

    def test_null_dialogue(self):
        """Null dialogue should be normalized to empty string."""
        raw = '{"action": "CONVERSE", "target_citizen": "Bob", "dialogue": null}'
        result = parse_llm_response(raw)
        assert isinstance(result, LLMAction)
        assert result.dialogue == ""

    def test_null_target_citizen(self):
        """Null target_citizen should be accepted."""
        raw = '{"action": "CONVERSE", "target_citizen": null, "dialogue": ""}'
        result = parse_llm_response(raw)
        assert isinstance(result, LLMAction)
        assert result.target_citizen is None

    def test_invalid_schema(self):
        """Invalid action type should fail schema validation."""
        raw = '{"action": "FLY", "target_citizen": "Moon"}'
        result = parse_llm_response(raw)
        assert isinstance(result, ParserError)

    def test_not_json_at_all(self):
        """Completely non-JSON text should return ParserError."""
        raw = "The citizen decides to take a walk in the park."
        result = parse_llm_response(raw)
        assert isinstance(result, ParserError)

    def test_empty_string(self):
        """Empty string should return ParserError."""
        raw = ""
        result = parse_llm_response(raw)
        assert isinstance(result, ParserError)

    def test_llm_action_with_delta(self):
        """Action with intended_state_delta should parse and preserve it."""
        raw = ('{"action": "GIFT", "target_citizen": "Charlie", "dialogue": "Here.", '
               '"target_zone": null, "intended_state_delta": {"gold": 5}}')
        result = parse_llm_response(raw)
        assert isinstance(result, LLMAction)
        assert result.action == ActionType.GIFT
        assert result.intended_state_delta == {"gold": 5}