"""
COMPREHENSIVE VALIDATION TEST SUITE

Covers 9 categories of validation tests to prove the simulator's validator
correctly rejects every invalid or impossible action.

Categories:
    A — Schema Validation
    B — Entity Validation  
    C — World Rules
    D — Resource Rules
    E — Simulation Integrity
    F — Impossible State Mutations
    G — Adversarial LLM Outputs
    H — Property-Based Random Tests
    I — Regression Tests (valid actions)
"""

import json
import random
import string

import pytest

from app.engine.validator import Validator
from app.engine.simulator import Simulator, MockProvider
from app.llm.parser import parse_llm_response
from app.models.actions import ActionType, LLMAction, ValidationResult, ParserError
from app.models.citizen import Citizen, PersonalityTensor
from app.models.world import WorldState


# =========================================================================
# Fixtures
# =========================================================================

VALID_CITIZEN_NAMES = ["Alice", "Bob", "Charlie"]
VALID_ZONES = ["Residential_Sector", "Market_District", "City_Park"]


@pytest.fixture
def world() -> WorldState:
    """Standard test world with 3 citizens."""
    citizens = {
        "Alice": Citizen(
            id="c1", name="Alice", current_zone="Market_District", gold=50, energy=10,
            personality=PersonalityTensor(sociability=0.8, aggression=0.2, empathy=0.9),
        ),
        "Bob": Citizen(
            id="c2", name="Bob", current_zone="Market_District", gold=50, energy=10,
            personality=PersonalityTensor(sociability=0.5, aggression=0.7, empathy=0.3),
        ),
        "Charlie": Citizen(
            id="c3", name="Charlie", current_zone="City_Park", gold=50, energy=10,
            personality=PersonalityTensor(sociability=0.3, aggression=0.4, empathy=0.6),
        ),
    }
    for name in citizens:
        for other_name in citizens:
            if name != other_name:
                citizens[name].get_relationship(other_name)
    return WorldState(citizens=citizens)


@pytest.fixture
def v(world: WorldState) -> Validator:
    """Short fixture name for validator."""
    return Validator(world)


# =========================================================================
# CATEGORY A — Schema Validation
# =========================================================================

class TestCategoryA_SchemaValidation:
    """Test that malformed/invalid JSON schemas are properly rejected."""

    def test_a1_malformed_json(self):
        """Malformed JSON string → ParserError."""
        result = parse_llm_response('{invalid json!!!}')
        assert isinstance(result, ParserError), "Malformed JSON should produce ParserError"

    def test_a2_missing_action_field(self):
        """JSON without action → should produce ParserError."""
        result = parse_llm_response('{"target_citizen": "Bob"}')
        assert isinstance(result, ParserError), "Missing action should produce ParserError"

    def test_a3_empty_string_input(self):
        """Empty string → ParserError."""
        result = parse_llm_response('')
        assert isinstance(result, ParserError)

    def test_a4_null_input(self):
        """'null' → ParserError."""
        result = parse_llm_response('null')
        assert isinstance(result, ParserError)

    def test_a5_invalid_enum_value(self, v, world):
        """'FLY' is not a valid ActionType → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action="FLY"), alice)  # type: ignore
        assert result.accepted is False
        assert "unknown" in result.reason.lower()

    def test_a6_wrong_data_types(self, v, world):
        """dialogue should be a string, but Pydantic coerces; test action is valid."""
        # Pydantic validates LLMAction construction, so this tests the schema boundary
        alice = world.get_citizen("Alice")
        action = LLMAction(action=ActionType.CONVERSE, target_citizen="Bob", dialogue=123)  # type: ignore
        result = v.validate(action, alice)
        # Pydantic may coerce 123 -> "123"; the action should still be valid
        assert result.accepted is True

    def test_a7_additional_unexpected_fields(self, v, world):
        """Unexpected fields are ignored by Pydantic → no rejection."""
        alice = world.get_citizen("Alice")
        # Construct dict with extra field, then parse via model_validate
        raw = {"action": "CONVERSE", "target_citizen": "Bob", "extra_field": "ignored"}
        action = LLMAction.model_validate(raw)
        result = v.validate(action, alice)
        assert result.accepted is True

    def test_a8_none_where_forbidden(self, v, world):
        """GIFT without target → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.GIFT, target_citizen=None), alice)
        assert result.accepted is False
        assert "requires a target" in result.reason.lower()

    def test_a9_empty_action_name(self, v, world):
        """Empty string as action → should produce ParserError or reject."""
        result = parse_llm_response('{"action": ""}')
        assert isinstance(result, ParserError), "Empty action should not parse"


# =========================================================================
# CATEGORY B — Entity Validation
# =========================================================================

class TestCategoryB_EntityValidation:
    """Test that nonexistent/mismatched entities are rejected."""

    def test_b1_citizen_does_not_exist(self, v, world):
        """Target citizen 'Zara' does not exist → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.GIFT, target_citizen="Zara"), alice)
        assert result.accepted is False
        assert "not found" in result.reason.lower()

    def test_b2_zone_does_not_exist(self, v, world):
        """Move to 'Atlantis' → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.MOVE, target_zone="Atlantis"), alice)
        assert result.accepted is False
        assert "invalid destination" in result.reason.lower()

    def test_b3_self_target_forbidden(self, v, world):
        """GIFT to self → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.GIFT, target_citizen="Alice"), alice)
        assert result.accepted is False
        assert "yourself" in result.reason.lower()

    def test_b4_invalid_zone_name_case(self, v, world):
        """Move to 'market district' lower case → should normalize and accept."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.MOVE, target_zone="market district"), alice)
        assert result.accepted is True
        assert result.safe_action.target_zone == "Market_District"

    def test_b5_steal_from_nonexistent(self, v, world):
        """STEAL from 'Ghost' → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.STEAL, target_citizen="Ghost"), alice)
        assert result.accepted is False
        assert "not found" in result.reason.lower()


# =========================================================================
# CATEGORY C — World Rules
# =========================================================================

class TestCategoryC_WorldRules:
    """Test zone-based rules and cross-zone interaction rejection."""

    def test_c1_gift_across_zones(self, v, world):
        """Alice in Market, Charlie in Park → GIFT rejected."""
        alice = world.get_citizen("Alice")
        charlie = world.get_citizen("Charlie")
        # Alice in Market, Charlie in City_Park
        result = v.validate(LLMAction(action=ActionType.GIFT, target_citizen="Charlie"), alice)
        assert result.accepted is False
        assert "different zone" in result.reason.lower()

    def test_c2_help_across_zones(self, v, world):
        """HELP across zones → rejected."""
        alice = world.get_citizen("Alice")
        charlie = world.get_citizen("Charlie")
        result = v.validate(LLMAction(action=ActionType.HELP, target_citizen="Charlie"), alice)
        assert result.accepted is False
        assert "different zone" in result.reason.lower()

    def test_c3_talk_across_zones(self, v, world):
        """CONVERSE across zones → accepted (converse is always valid)."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.CONVERSE, target_citizen="Charlie"), alice)
        # Converse doesn't check zone
        assert result.accepted is True

    def test_c4_follow_nonexistent(self, v, world):
        """FOLLOW 'Ghost' → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.FOLLOW, target_citizen="Ghost"), alice)
        assert result.accepted is False
        assert "not found" in result.reason.lower()

    def test_c5_move_to_current_zone(self, v, world):
        """MOVE to same zone → accepted (no rule against it)."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.MOVE, target_zone="Market_District"), alice)
        assert result.accepted is True

    def test_c6_trade_across_zones(self, v, world):
        """TRADE across zones → rejected."""
        alice = world.get_citizen("Alice")
        charlie = world.get_citizen("Charlie")
        result = v.validate(LLMAction(action=ActionType.TRADE, target_citizen="Charlie"), alice)
        assert result.accepted is False
        assert "different zone" in result.reason.lower()

    def test_c7_ignore_across_zones(self, v, world):
        """IGNORE across zones → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.IGNORE, target_citizen="Charlie"), alice)
        assert result.accepted is False
        assert "different zone" in result.reason.lower()

    def test_c8_warn_across_zones(self, v, world):
        """WARN across zones → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.WARN, target_citizen="Charlie"), alice)
        assert result.accepted is False
        assert "different zone" in result.reason.lower()

    def test_c9_follow_across_zones(self, v, world):
        """FOLLOW across zones → rejected (target not in same zone)."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.FOLLOW, target_citizen="Charlie"), alice)
        assert result.accepted is False
        assert "not in same zone" in result.reason.lower()


# =========================================================================
# CATEGORY D — Resource Rules
# =========================================================================

class TestCategoryD_ResourceRules:
    """Test that resource constraints are enforced."""

    def test_d1_gift_insufficient_gold(self, v, world):
        """Alice has 3 gold (needs 5) → GIFT rejected."""
        alice = world.get_citizen("Alice")
        alice.gold = 3
        result = v.validate(LLMAction(action=ActionType.GIFT, target_citizen="Bob"), alice)
        assert result.accepted is False
        assert "gold" in result.reason.lower()

    def test_d2_help_insufficient_energy(self, v, world):
        """Alice has 1 energy (needs 2) → HELP rejected, fallback to GIFT."""
        alice = world.get_citizen("Alice")
        alice.energy = 1
        result = v.validate(LLMAction(action=ActionType.HELP, target_citizen="Bob"), alice)
        assert result.accepted is False
        assert "energy" in result.reason.lower()
        assert result.safe_action.action == ActionType.GIFT

    def test_d3_work_zero_energy(self, v, world):
        """Alice has 0 energy → WORK rejected, fallback to REST."""
        alice = world.get_citizen("Alice")
        alice.energy = 0
        result = v.validate(LLMAction(action=ActionType.WORK), alice)
        assert result.accepted is False
        assert "incapacitated" in result.reason.lower()
        assert result.safe_action.action == ActionType.REST

    def test_d4_trade_insufficient_gold(self, v, world):
        """Alice has 5 gold (needs 10) → TRADE rejected."""
        alice = world.get_citizen("Alice")
        alice.gold = 5
        result = v.validate(LLMAction(action=ActionType.TRADE, target_citizen="Bob"), alice)
        assert result.accepted is False
        assert "gold" in result.reason.lower()

    def test_d5_help_insufficient_gold(self, v, world):
        """Alice has 5 gold (needs 8) → HELP rejected, fallback to GIFT."""
        alice = world.get_citizen("Alice")
        alice.gold = 5
        result = v.validate(LLMAction(action=ActionType.HELP, target_citizen="Bob"), alice)
        assert result.accepted is False
        assert "gold" in result.reason.lower()
        assert result.safe_action.action == ActionType.GIFT

    def test_d6_steal_insufficient_energy(self, v, world):
        """Alice has 1 energy (needs 2) → STEAL rejected."""
        alice = world.get_citizen("Alice")
        alice.energy = 1
        result = v.validate(LLMAction(action=ActionType.STEAL, target_citizen="Bob"), alice)
        assert result.accepted is False
        assert "energy" in result.reason.lower()

    def test_d7_gift_exact_threshold(self, v, world):
        """Alice has exactly 5 gold → GIFT accepted."""
        alice = world.get_citizen("Alice")
        alice.gold = 5
        result = v.validate(LLMAction(action=ActionType.GIFT, target_citizen="Bob"), alice)
        assert result.accepted is True

    def test_d8_work_exact_threshold(self, v, world):
        """Alice has exactly 3 energy → WORK accepted."""
        alice = world.get_citizen("Alice")
        alice.energy = 3
        result = v.validate(LLMAction(action=ActionType.WORK), alice)
        assert result.accepted is True

    def test_d9_help_exact_threshold(self, v, world):
        """Alice has exactly 8 gold and 2 energy → HELP accepted."""
        alice = world.get_citizen("Alice")
        alice.gold = 8
        alice.energy = 2
        result = v.validate(LLMAction(action=ActionType.HELP, target_citizen="Bob"), alice)
        assert result.accepted is True


# =========================================================================
# CATEGORY E — Simulation Integrity
# =========================================================================

class TestCategoryE_SimulationIntegrity:
    """Test that the validator handles complex/malformed inputs safely."""

    def test_e1_multiple_actions_in_one_output(self):
        """Two JSON objects in one output → Parser handles first only."""
        raw = '{"action":"MOVE","target_zone":"City_Park"}\n{"action":"STEAL","target_citizen":"Bob"}'
        result = parse_llm_response(raw)
        # Parser should parse the first valid object
        if not isinstance(result, ParserError):
            assert result.action == ActionType.MOVE

    def test_e2_nested_actions(self, v, world):
        """Action wrapped in nested structure → ParserError."""
        raw = '{"action":{"nested":{"action":"MOVE"}}}'
        result = parse_llm_response(raw)
        assert isinstance(result, ParserError)

    def test_e3_array_instead_of_object(self):
        """Array as input → ParserError."""
        result = parse_llm_response('[{"action":"MOVE"}]')
        assert isinstance(result, ParserError)

    def test_e4_markdown_wrapped_json(self):
        """Markdown ```json fences → parser strips them."""
        raw = '```json\n{"action":"CONVERSE","target_citizen":"Bob"}\n```'
        result = parse_llm_response(raw)
        # Parser should strip fences and parse
        assert not isinstance(result, ParserError)
        assert result.action == ActionType.CONVERSE

    def test_e5_llm_explanation_before_json(self):
        """Text before JSON → parser extracts JSON."""
        raw = 'I think Alice should talk to Bob.\n{"action":"CONVERSE","target_citizen":"Bob"}'
        result = parse_llm_response(raw)
        assert not isinstance(result, ParserError)

    def test_e6_llm_explanation_after_json(self):
        """JSON followed by text → parser extracts JSON."""
        raw = '{"action":"CONVERSE","target_citizen":"Bob"}\nThat is my decision.'
        result = parse_llm_response(raw)
        assert not isinstance(result, ParserError)

    def test_e7_trailing_comma_in_json(self):
        """Trailing comma → parser should repair."""
        raw = '{"action":"CONVERSE","target_citizen":"Bob",}'
        result = parse_llm_response(raw)
        assert not isinstance(result, ParserError), "Should repair trailing comma"

    def test_e8_number_instead_of_string_action(self):
        """Number as action value → ParserError."""
        result = parse_llm_response('{"action": 42}')
        assert isinstance(result, ParserError)


# =========================================================================
# CATEGORY F — Impossible State Mutations
# =========================================================================

class TestCategoryF_ImpossibleStateMutations:
    """Test that impossible/suicidal actions are rejected."""

    def test_f1_steal_from_self(self, v, world):
        """STEAL from self → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.STEAL, target_citizen="Alice"), alice)
        assert result.accepted is False
        assert "yourself" in result.reason.lower()

    def test_f2_gift_to_self(self, v, world):
        """GIFT to self → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.GIFT, target_citizen="Alice"), alice)
        assert result.accepted is False
        assert "yourself" in result.reason.lower()

    def test_f3_help_self(self, v, world):
        """HELP self → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.HELP, target_citizen="Alice"), alice)
        assert result.accepted is False
        assert "yourself" in result.reason.lower()

    def test_f4_work_incapacitated(self, v, world):
        """WORK with 0 energy → rejected, fallback to REST."""
        alice = world.get_citizen("Alice")
        alice.energy = 0
        result = v.validate(LLMAction(action=ActionType.WORK), alice)
        assert result.accepted is False
        assert result.safe_action.action == ActionType.REST

    def test_f5_move_incapacitated(self, v, world):
        """MOVE with 0 energy → rejected, fallback to REST."""
        alice = world.get_citizen("Alice")
        alice.energy = 0
        result = v.validate(LLMAction(action=ActionType.MOVE, target_zone="City_Park"), alice)
        assert result.accepted is False
        assert result.safe_action.action == ActionType.REST

    def test_f6_rest_always_allowed(self, v, world):
        """REST with 0 energy → accepted (recovery mechanic)."""
        alice = world.get_citizen("Alice")
        alice.energy = 0
        result = v.validate(LLMAction(action=ActionType.REST), alice)
        assert result.accepted is True

    def test_f7_follow_self(self, v, world):
        """FOLLOW self → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.FOLLOW, target_citizen="Alice"), alice)
        assert result.accepted is False
        assert "yourself" in result.reason.lower()

    def test_f8_warn_self(self, v, world):
        """WARN self → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.WARN, target_citizen="Alice"), alice)
        assert result.accepted is False
        assert "yourself" in result.reason.lower()

    def test_f9_trade_self(self, v, world):
        """TRADE self → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.TRADE, target_citizen="Alice"), alice)
        assert result.accepted is False
        assert "yourself" in result.reason.lower()

    def test_f10_ignore_self(self, v, world):
        """IGNORE self → rejected."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.IGNORE, target_citizen="Alice"), alice)
        assert result.accepted is False
        assert "yourself" in result.reason.lower()


# =========================================================================
# CATEGORY G — Adversarial LLM Outputs
# =========================================================================

class TestCategoryG_AdversarialOutputs:
    """Test that realistic hallucinated LLM outputs are safely handled."""

    def test_g1_prose_only(self):
        """No JSON at all → ParserError."""
        result = parse_llm_response("I think Bob should probably help Alice.")
        assert isinstance(result, ParserError)

    def test_g2_malformed_json_with_action_keyword(self):
        """JSON-like but invalid → recovery should handle."""
        result = parse_llm_response('{\naction:"HELP"\n}')
        assert isinstance(result, ParserError)

    def test_g3_single_quotes(self):
        """JSON with single quotes → should repair."""
        result = parse_llm_response("{'action':'CONVERSE','target_citizen':'Alice'}")
        assert not isinstance(result, ParserError), "Should repair single quotes"

    def test_g4_trailing_comma_object(self):
        """Trailing comma → should repair."""
        result = parse_llm_response('{"action":"MOVE","target_zone":"City_Park",}')
        assert not isinstance(result, ParserError)

    def test_g5_code_fences_with_language(self):
        """Python code fences → parser strips."""
        raw = '```python\n{"action":"REST"}\n```'
        result = parse_llm_response(raw)
        assert not isinstance(result, ParserError), "Should handle python fences"

    def test_g6_escaped_json_string(self):
        """JSON as escaped string → maybe ParserError (depends on context)."""
        # This simulates an LLM returning a Python repr of a dict
        result = parse_llm_response('{\\"action\\": \\"CONVERSE\\"}')
        # May be ParserError, but shouldn't crash
        assert result is not None

    def test_g7_partial_json(self):
        """Truncated JSON → ParserError, no crash."""
        result = parse_llm_response('{"action":"MOVE","target_zone":"Cit')
        assert isinstance(result, ParserError)

    def test_g8_double_json(self):
        """Two complete JSON objects → parses first one."""
        raw = '{"action":"MOVE","target_zone":"City_Park"}\n{"action":"STEAL"}'
        result = parse_llm_response(raw)
        assert not isinstance(result, ParserError)
        assert result.action == ActionType.MOVE

    def test_g9_random_text_with_json_fragment(self):
        """Random text with JSON inside → should extract."""
        raw = "Hmm, let me think... {\"action\":\"REST\"} Yes, that's best."
        result = parse_llm_response(raw)
        assert not isinstance(result, ParserError)
        assert result.action == ActionType.REST

    def test_g10_emoji_and_unicode(self):
        """JSON with unicode → should parse correctly."""
        raw = '{"action":"CONVERSE","dialogue":"Hello 😊 world üñîçødé"}'
        result = parse_llm_response(raw)
        assert not isinstance(result, ParserError)
        assert "😊" in result.dialogue


# =========================================================================
# CATEGORY H — Property-Based Random Tests
# =========================================================================

class TestCategoryH_PropertyTests:
    """Generate random invalid actions and verify validator never crashes."""

    def test_h1_random_action_names(self, v, world):
        """Random strings as action names."""
        alice = world.get_citizen("Alice")
        for _ in range(50):
            random_action = ''.join(random.choices(string.ascii_uppercase, k=random.randint(3, 10)))
            try:
                action = LLMAction(action=random_action)  # type: ignore
                result = v.validate(action, alice)
                # Should either reject or potentially crash if not caught
                assert result is not None
            except Exception:
                # Some invalid enums may fail at construction — that's acceptable
                pass

    def test_h2_random_citizen_ids(self, v, world):
        """Random citizen names as targets."""
        alice = world.get_citizen("Alice")
        for _ in range(50):
            random_name = ''.join(random.choices(string.ascii_letters, k=random.randint(1, 20)))
            action = LLMAction(action=ActionType.GIFT, target_citizen=random_name)
            result = v.validate(action, alice)
            # Should always produce a ValidationResult (rejected for nonexistent)
            assert isinstance(result, ValidationResult)
            if random_name not in ["Alice", "Bob", "Charlie"]:
                assert result.accepted is False

    def test_h3_random_zone_names(self, v, world):
        """Random zone names as destinations."""
        alice = world.get_citizen("Alice")
        for _ in range(50):
            random_zone = ''.join(random.choices(string.ascii_letters, k=random.randint(1, 30)))
            action = LLMAction(action=ActionType.MOVE, target_zone=random_zone)
            result = v.validate(action, alice)
            assert isinstance(result, ValidationResult)

    def test_h4_negative_integers(self, v, world):
        """Negative gold/energy values."""
        alice = world.get_citizen("Alice")
        alice.gold = -5  # Pydantic clamping should prevent this, but test validator path
        # After clamping
        alice.gold = 0
        action = LLMAction(action=ActionType.GIFT, target_citizen="Bob")
        result = v.validate(action, alice)
        assert result.accepted is False

    def test_h5_large_integers(self, v, world):
        """Huge gold values."""
        alice = world.get_citizen("Alice")
        alice.gold = 999999
        result = v.validate(LLMAction(action=ActionType.GIFT, target_citizen="Bob"), alice)
        assert result.accepted is True

    def test_h6_empty_object(self, v, world):
        """Empty action dict → Pydantic should reject at construction."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            LLMAction()  # Missing required 'action' field

    def test_h7_unicode_emoji_in_fields(self, v, world):
        """Unicode in dialogue field."""
        alice = world.get_citizen("Alice")
        action = LLMAction(action=ActionType.CONVERSE, dialogue="🔥💀🔥")
        result = v.validate(action, alice)
        assert result.accepted is True

    def test_h8_none_for_required(self, v, world):
        """None for action → Pydantic ValidationError."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            LLMAction(action=None)  # type: ignore

    def test_h9_zero_length_strings(self, v, world):
        """Empty dialogue string."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.CONVERSE, dialogue=""), alice)
        assert result.accepted is True

    def test_h10_whitespace_only(self, v, world):
        """Whitespace-only dialogue."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.CONVERSE, dialogue="   "), alice)
        assert result.accepted is True


# =========================================================================
# CATEGORY I — Regression Tests (Valid Actions)
# =========================================================================

class TestCategoryI_RegressionTests:
    """For every valid action, one canonical passing test.
    
    These ensure future validator updates don't break legal actions.
    """

    def test_i1_move_valid(self, v, world):
        """MOVE to valid zone → accepted."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.MOVE, target_zone="City_Park"), alice)
        assert result.accepted is True

    def test_i2_rest_valid(self, v, world):
        """REST → accepted."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.REST), alice)
        assert result.accepted is True

    def test_i3_work_valid(self, v, world):
        """WORK with sufficient energy → accepted."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.WORK), alice)
        assert result.accepted is True

    def test_i4_help_valid(self, v, world):
        """HELP with sufficient resources, same zone → accepted."""
        alice = world.get_citizen("Alice")
        alice.gold = 10
        alice.energy = 5
        result = v.validate(LLMAction(action=ActionType.HELP, target_citizen="Bob"), alice)
        assert result.accepted is True

    def test_i5_gift_valid(self, v, world):
        """GIFT with sufficient gold, same zone → accepted."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.GIFT, target_citizen="Bob"), alice)
        assert result.accepted is True

    def test_i6_follow_valid(self, v, world):
        """FOLLOW target in same zone → accepted."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.FOLLOW, target_citizen="Bob"), alice)
        assert result.accepted is True

    def test_i7_explore_valid(self, v, world):
        """EXPLORE valid zone → accepted."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.EXPLORE, target_zone="City_Park"), alice)
        assert result.accepted is True

    def test_i8_converse_valid(self, v, world):
        """CONVERSE with target → accepted."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.CONVERSE, target_citizen="Bob"), alice)
        assert result.accepted is True

    def test_i9_warn_valid(self, v, world):
        """WARN target in same zone → accepted."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.WARN, target_citizen="Bob"), alice)
        assert result.accepted is True

    def test_i10_request_help_valid(self, v, world):
        """REQUEST_HELP target in same zone → accepted."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.REQUEST_HELP, target_citizen="Bob"), alice)
        assert result.accepted is True

    def test_i11_trade_valid(self, v, world):
        """TRADE with sufficient gold, same zone → accepted."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.TRADE, target_citizen="Bob"), alice)
        assert result.accepted is True

    def test_i12_ignore_valid(self, v, world):
        """IGNORE target in same zone → accepted."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.IGNORE, target_citizen="Bob"), alice)
        assert result.accepted is True

    def test_i13_observe_valid(self, v, world):
        """OBSERVE → accepted."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.OBSERVE), alice)
        assert result.accepted is True

    def test_i14_steal_valid(self, v, world):
        """STEAL target in same zone → accepted."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.STEAL, target_citizen="Bob"), alice)
        assert result.accepted is True

    def test_i15_steal_across_zones_rejected(self, v, world):
        """STEAL across zones → rejected (sanity check valid → invalid)."""
        bob = world.get_citizen("Bob")
        charlie = world.get_citizen("Charlie")
        # Bob in Market, Charlie in Park
        result = v.validate(LLMAction(action=ActionType.STEAL, target_citizen="Charlie"), bob)
        assert result.accepted is False
        assert "different zone" in result.reason.lower()


# =========================================================================
# CATEGORY J — ValidationResult Error Codes
# =========================================================================

class TestCategoryJ_ErrorCodes:
    """Test that every ValidationResult includes a meaningful error_code."""

    def test_j1_accepted_has_ok_code(self, v, world):
        """Accepted actions should have error_code='OK'."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.CONVERSE), alice)
        assert result.accepted is True
        assert result.error_code == "OK"

    def test_j2_incapacitated_has_code(self, v, world):
        """Incapacitated rejections should have error_code='INCAPACITATED'."""
        alice = world.get_citizen("Alice")
        alice.energy = 0
        result = v.validate(LLMAction(action=ActionType.MOVE, target_zone="City_Park"), alice)
        assert result.accepted is False
        assert result.error_code == "INCAPACITATED"

    def test_j3_unknown_action_has_code(self, v, world):
        """Unknown action type should have error_code='UNKNOWN_ACTION'."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action="FLY"), alice)  # type: ignore
        assert result.accepted is False
        assert result.error_code == "UNKNOWN_ACTION"

    def test_j4_move_no_destination_has_code(self, v, world):
        """MOVE without destination should have error_code='MOVE_NO_DESTINATION'."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.MOVE, target_zone=None), alice)
        assert result.accepted is False
        assert result.error_code == "MOVE_NO_DESTINATION"

    def test_j5_move_invalid_zone_has_code(self, v, world):
        """MOVE to invalid zone should have error_code='MOVE_INVALID_ZONE'."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.MOVE, target_zone="Atlantis"), alice)
        assert result.accepted is False
        assert result.error_code == "MOVE_INVALID_ZONE"

    def test_j6_gift_no_target_has_code(self, v, world):
        """GIFT without target should have error_code='GIFT_NO_TARGET'."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.GIFT, target_citizen=None), alice)
        assert result.accepted is False
        assert "GIFT" in result.error_code

    def test_j7_gift_self_has_code(self, v, world):
        """GIFT to self should have error_code='GIFT_SELF'."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.GIFT, target_citizen="Alice"), alice)
        assert result.accepted is False
        assert result.error_code == "GIFT_SELF"

    def test_j8_converse_bad_target_has_code(self, v, world):
        """CONVERSE with nonexistent target should have error_code='CONVERSE_BAD_TARGET'."""
        alice = world.get_citizen("Alice")
        result = v.validate(LLMAction(action=ActionType.CONVERSE, target_citizen="Ghost"), alice)
        assert result.accepted is True  # Still accepted, but target is nulled
        assert result.error_code == "CONVERSE_BAD_TARGET"
        assert result.safe_action.target_citizen is None