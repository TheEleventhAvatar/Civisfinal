"""
Tests for the Validator class.

Covers schema validation, action validation, resource validation,
zone validation, and target validation.
"""

import pytest

from app.engine.validator import Validator
from app.models.actions import ActionType, LLMAction
from app.models.citizen import Citizen, PersonalityTensor
from app.models.world import WorldState


@pytest.fixture
def world() -> WorldState:
    """Create a standard test world with 3 citizens."""
    citizens = {
        "Alice": Citizen(
            id="c1",
            name="Alice",
            current_zone="Market_District",
            gold=50,
            energy=10,
            personality=PersonalityTensor(sociability=0.8, aggression=0.2, empathy=0.9),
        ),
        "Bob": Citizen(
            id="c2",
            name="Bob",
            current_zone="Market_District",
            gold=50,
            energy=10,
            personality=PersonalityTensor(sociability=0.5, aggression=0.7, empathy=0.3),
        ),
        "Charlie": Citizen(
            id="c3",
            name="Charlie",
            current_zone="City_Park",
            gold=50,
            energy=10,
            personality=PersonalityTensor(sociability=0.3, aggression=0.4, empathy=0.6),
        ),
    }
    # Initialize relationships using names as keys
    for name in citizens:
        for other_name in citizens:
            if name != other_name:
                citizens[name].get_relationship(other_name)
    return WorldState(citizens=citizens)


@pytest.fixture
def validator(world: WorldState) -> Validator:
    """Create a validator for the test world."""
    return Validator(world)


class TestMoveValidation:
    """Test MOVE action validation rules."""

    def test_valid_move(self, validator: Validator, world: WorldState):
        """MOVE should be accepted with a valid destination zone in target_zone."""
        action = LLMAction(
            action=ActionType.MOVE,
            target_zone="City_Park",
        )
        bob = world.get_citizen("Bob")
        result = validator.validate(action, bob)
        assert result.accepted is True
        assert result.safe_action.target_zone == "City_Park"

    def test_invalid_move(self, validator: Validator, world: WorldState):
        """MOVE should be rejected with an invalid destination."""
        action = LLMAction(
            action=ActionType.MOVE,
            target_zone="Nowhere",
        )
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        assert result.accepted is False
        assert "invalid destination" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE

    def test_move_normalizes_zone_name(self, validator: Validator, world: WorldState):
        """MOVE should normalize zone names (spaces to underscores, title case)."""
        action = LLMAction(
            action=ActionType.MOVE,
            target_zone="city park",
        )
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        assert result.accepted is True
        assert result.safe_action.target_zone == "City_Park"

    def test_move_no_destination(self, validator: Validator, world: WorldState):
        """MOVE should be rejected when no destination is specified."""
        action = LLMAction(
            action=ActionType.MOVE,
            target_zone=None,
        )
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        assert result.accepted is False
        assert "target_zone" in result.reason.lower()


class TestGiftValidation:
    """Test GIFT action validation rules."""

    def test_valid_gift(self, validator: Validator, world: WorldState):
        """GIFT should be accepted when target is in same zone and actor has gold."""
        action = LLMAction(
            action=ActionType.GIFT,
            target_citizen="Bob",
            dialogue="Here's some gold.",
        )
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        assert result.accepted is True
        assert "validated" in result.reason.lower()

    def test_invalid_gift_no_gold(self, validator: Validator, world: WorldState):
        """GIFT should be rejected when actor has insufficient gold."""
        bob = world.get_citizen("Bob")
        bob.gold = 0

        action = LLMAction(
            action=ActionType.GIFT,
            target_citizen="Alice",
            dialogue="I have nothing to give.",
        )
        result = validator.validate(action, bob)
        assert result.accepted is False
        assert "gold" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE

    def test_invalid_gift_different_zone(self, validator: Validator, world: WorldState):
        """GIFT should be rejected when target is in a different zone."""
        action = LLMAction(
            action=ActionType.GIFT,
            target_citizen="Charlie",
            dialogue="I found you!",
        )
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        assert result.accepted is False
        assert "different zone" in result.reason.lower()

    def test_invalid_gift_no_target(self, validator: Validator, world: WorldState):
        """GIFT should be rejected when no target is specified."""
        action = LLMAction(
            action=ActionType.GIFT,
            target_citizen=None,
            dialogue="For no one.",
        )
        bob = world.get_citizen("Bob")
        result = validator.validate(action, bob)
        assert result.accepted is False
        assert "target" in result.reason.lower()


class TestStealValidation:
    """Test STEAL action validation rules."""

    def test_valid_steal(self, validator: Validator, world: WorldState):
        """STEAL should be accepted when target is in same zone."""
        action = LLMAction(
            action=ActionType.STEAL,
            target_citizen="Bob",
        )
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        assert result.accepted is True

    def test_invalid_steal_different_zone(self, validator: Validator, world: WorldState):
        """STEAL should be rejected when target is in a different zone."""
        action = LLMAction(
            action=ActionType.STEAL,
            target_citizen="Alice",
        )
        charlie = world.get_citizen("Charlie")
        result = validator.validate(action, charlie)
        assert result.accepted is False
        assert "different zone" in result.reason.lower()

    def test_invalid_steal_nonexistent_target(self, validator: Validator, world: WorldState):
        """STEAL should be rejected when target does not exist."""
        action = LLMAction(
            action=ActionType.STEAL,
            target_citizen="Ghost",
        )
        bob = world.get_citizen("Bob")
        result = validator.validate(action, bob)
        assert result.accepted is False
        assert "not found" in result.reason.lower()

    def test_invalid_steal_no_target(self, validator: Validator, world: WorldState):
        """STEAL should be rejected when no target is specified."""
        action = LLMAction(
            action=ActionType.STEAL,
            target_citizen=None,
        )
        bob = world.get_citizen("Bob")
        result = validator.validate(action, bob)
        assert result.accepted is False
        assert "target" in result.reason.lower()


class TestConverseValidation:
    """Test CONVERSE action validation rules."""

    def test_converse_always_accepted(self, validator: Validator, world: WorldState):
        """CONVERSE should always be accepted regardless of context."""
        action = LLMAction(
            action=ActionType.CONVERSE,
            target_citizen="Bob",
            dialogue="Hello!",
        )
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        assert result.accepted is True

    def test_converse_no_target(self, validator: Validator, world: WorldState):
        """CONVERSE should be accepted even without a target."""
        action = LLMAction(
            action=ActionType.CONVERSE,
            target_citizen=None,
            dialogue="Is anyone there?",
        )
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        assert result.accepted is True


class TestHallucinatedGold:
    """Test that hallucinated state deltas are properly ignored."""

    def test_llm_cannot_inject_gold(self, validator: Validator, world: WorldState):
        """
        The LLM cannot inject gold through intended_state_delta.
        The validator ignores intended_state_delta entirely.
        """
        action = LLMAction(
            action=ActionType.CONVERSE,
            target_citizen="Bob",
            dialogue="I wish I had more gold.",
            intended_state_delta={"gold": 1000},
        )
        bob = world.get_citizen("Bob")
        result = validator.validate(action, bob)

        assert result.accepted is True
        assert result.safe_action.action == ActionType.CONVERSE
        assert bob.gold == 50

    def test_hallucinated_gold_injection_rejected(self, validator: Validator, world: WorldState):
        """
        Even if the LLM tries GIFT with no gold, the validator catches it.
        """
        alice = world.get_citizen("Alice")
        alice.gold = 0

        action = LLMAction(
            action=ActionType.GIFT,
            target_citizen="Bob",
            dialogue="Here's gold I don't have.",
            intended_state_delta={"gold_decrease": 100, "gold_increase_for_bob": 100},
        )
        result = validator.validate(action, alice)
        assert result.accepted is False
        assert "gold" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE