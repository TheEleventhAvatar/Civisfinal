"""
Benchmark test suite for deterministic validation.

These tests ensure the validator correctly rejects impossible actions
and only allows valid state mutations. This is critical for maintaining
the integrity of the simulation.
"""

import pytest

from app.engine.validator import Validator
from app.engine.simulator import Simulator
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


class TestInvalidGiftBenchmark:
    """Benchmark: Invalid GIFT scenarios must be rejected."""

    def test_gift_with_insufficient_gold(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice has 3 gold, tries to gift 5.
        Expected: REJECTED, fallback to CONVERSE, no gold transfer.
        """
        alice = world.get_citizen("Alice")
        alice.gold = 3
        
        action = LLMAction(
            action=ActionType.GIFT,
            target_citizen="Bob",
            dialogue="I wish I had more.",
        )
        
        result = validator.validate(action, alice)
        
        # Must reject
        assert result.accepted is False
        assert "gold" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE
        
        # Verify no state change
        assert alice.gold == 3
        bob = world.get_citizen("Bob")
        assert bob.gold == 50

    def test_gift_to_different_zone(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice in Residential, Bob in Market.
        Expected: REJECTED, fallback to CONVERSE, no gold transfer.
        """
        alice = world.get_citizen("Alice")
        alice.current_zone = "Residential_Sector"
        bob = world.get_citizen("Bob")
        bob.current_zone = "Market_District"
        
        action = LLMAction(
            action=ActionType.GIFT,
            target_citizen="Bob",
            dialogue="Here you go.",
        )
        
        result = validator.validate(action, alice)
        
        # Must reject
        assert result.accepted is False
        assert "different zone" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE
        
        # Verify no state change
        assert alice.gold == 50
        assert bob.gold == 50

    def test_gift_to_self(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice tries to gift to herself.
        Expected: REJECTED, fallback to CONVERSE.
        """
        action = LLMAction(
            action=ActionType.GIFT,
            target_citizen="Alice",
            dialogue="Giving to myself.",
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "yourself" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE


class TestInvalidTheftBenchmark:
    """Benchmark: Invalid STEAL scenarios must be rejected."""

    def test_steal_from_different_zone(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice in Residential, Bob in Market.
        Expected: REJECTED, fallback to CONVERSE, no gold transfer.
        """
        alice = world.get_citizen("Alice")
        alice.current_zone = "Residential_Sector"
        bob = world.get_citizen("Bob")
        bob.current_zone = "Market_District"
        
        action = LLMAction(
            action=ActionType.STEAL,
            target_citizen="Bob",
        )
        
        result = validator.validate(action, alice)
        
        # Must reject
        assert result.accepted is False
        assert "different zone" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE
        
        # Verify no state change
        assert alice.gold == 50
        assert bob.gold == 50

    def test_steal_with_insufficient_energy(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice has 1 energy, tries to steal (needs 2).
        Expected: REJECTED, fallback to CONVERSE.
        """
        alice = world.get_citizen("Alice")
        alice.energy = 1
        
        action = LLMAction(
            action=ActionType.STEAL,
            target_citizen="Bob",
        )
        
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "energy" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE

    def test_steal_from_self(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice tries to steal from herself.
        Expected: REJECTED, fallback to CONVERSE.
        """
        action = LLMAction(
            action=ActionType.STEAL,
            target_citizen="Alice",
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "yourself" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE


class TestInvalidMovementBenchmark:
    """Benchmark: Invalid MOVE scenarios must be rejected."""

    def test_move_to_nonexistent_zone(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice tries to move to "Atlantis".
        Expected: REJECTED, fallback to CONVERSE, no movement.
        """
        action = LLMAction(
            action=ActionType.MOVE,
            target_zone="Atlantis",
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "invalid destination" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE
        assert alice.current_zone == "Market_District"

    def test_move_without_destination(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice tries to move without specifying destination.
        Expected: REJECTED, fallback to CONVERSE.
        """
        action = LLMAction(
            action=ActionType.MOVE,
            target_zone=None,
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "target_zone" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE


class TestInvalidConversationBenchmark:
    """Benchmark: Invalid CONVERSE scenarios."""

    def test_converse_with_nonexistent_citizen(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice tries to talk to "Ghost" who doesn't exist.
        Expected: REJECTED, fallback to CONVERSE with no target.
        """
        action = LLMAction(
            action=ActionType.CONVERSE,
            target_citizen="Ghost",
            dialogue="Hello?",
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        # CONVERSE is always accepted, but target is invalidated
        assert result.accepted is True
        assert result.safe_action.target_citizen is None


class TestInvalidWorkBenchmark:
    """Benchmark: Invalid WORK scenarios."""

    def test_work_with_insufficient_energy(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice has 2 energy, tries to work (needs 3).
        Expected: REJECTED, fallback to REST.
        """
        alice = world.get_citizen("Alice")
        alice.energy = 2
        
        action = LLMAction(
            action=ActionType.WORK,
        )
        
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "energy" in result.reason.lower()
        assert result.safe_action.action == ActionType.REST

    def test_work_on_cooldown(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice just worked, tries to work again (3-tick cooldown).
        Expected: REJECTED, fallback to REST.
        """
        alice = world.get_citizen("Alice")
        alice.cooldowns["work"] = 2  # 2 ticks remaining
        
        action = LLMAction(
            action=ActionType.WORK,
        )
        
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "cooldown" in result.reason.lower()
        assert result.safe_action.action == ActionType.REST


class TestInvalidRestBenchmark:
    """Benchmark: Invalid REST scenarios."""

    def test_rest_on_cooldown(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice just rested, tries to rest again (2-tick cooldown).
        Expected: REJECTED, fallback to OBSERVE.
        """
        alice = world.get_citizen("Alice")
        alice.cooldowns["rest"] = 1  # 1 tick remaining
        
        action = LLMAction(
            action=ActionType.REST,
        )
        
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "cooldown" in result.reason.lower()
        assert result.safe_action.action == ActionType.OBSERVE


class TestInvalidHelpBenchmark:
    """Benchmark: Invalid HELP scenarios."""

    def test_help_with_insufficient_gold(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice has 5 gold, tries to help (needs 8).
        Expected: REJECTED, fallback to GIFT.
        """
        alice = world.get_citizen("Alice")
        alice.gold = 5
        
        action = LLMAction(
            action=ActionType.HELP,
            target_citizen="Bob",
            dialogue="I wish I could help more.",
        )
        
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "gold" in result.reason.lower()
        assert result.safe_action.action == ActionType.GIFT

    def test_help_with_insufficient_energy(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice has 1 energy, tries to help (needs 2).
        Expected: REJECTED, fallback to GIFT.
        """
        alice = world.get_citizen("Alice")
        alice.energy = 1
        
        action = LLMAction(
            action=ActionType.HELP,
            target_citizen="Bob",
            dialogue="Let me help you.",
        )
        
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "energy" in result.reason.lower()
        assert result.safe_action.action == ActionType.GIFT


class TestInvalidTradeBenchmark:
    """Benchmark: Invalid TRADE scenarios."""

    def test_trade_with_insufficient_gold(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice has 5 gold, tries to trade (needs 10).
        Expected: REJECTED, fallback to CONVERSE.
        """
        alice = world.get_citizen("Alice")
        alice.gold = 5
        
        action = LLMAction(
            action=ActionType.TRADE,
            target_citizen="Bob",
            dialogue="Want to trade?",
        )
        
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "gold" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE


class TestSelfTargetBenchmark:
    """Benchmark: Actions targeting self must be rejected."""

    def test_follow_self(self, validator: Validator, world: WorldState):
        """Cannot follow yourself."""
        action = LLMAction(
            action=ActionType.FOLLOW,
            target_citizen="Alice",
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "yourself" in result.reason.lower()

    def test_warn_self(self, validator: Validator, world: WorldState):
        """Cannot warn yourself."""
        action = LLMAction(
            action=ActionType.WARN,
            target_citizen="Alice",
            dialogue="Watch out!",
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "yourself" in result.reason.lower()

    def test_request_help_from_self(self, validator: Validator, world: WorldState):
        """Cannot request help from yourself."""
        action = LLMAction(
            action=ActionType.REQUEST_HELP,
            target_citizen="Alice",
            dialogue="Help me!",
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "yourself" in result.reason.lower()

    def test_trade_with_self(self, validator: Validator, world: WorldState):
        """Cannot trade with yourself."""
        action = LLMAction(
            action=ActionType.TRADE,
            target_citizen="Alice",
            dialogue="Trading with myself.",
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "yourself" in result.reason.lower()

    def test_ignore_self(self, validator: Validator, world: WorldState):
        """Cannot ignore yourself."""
        action = LLMAction(
            action=ActionType.IGNORE,
            target_citizen="Alice",
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "yourself" in result.reason.lower()


class TestIncapacitatedCitizenBenchmark:
    """Benchmark: Incapacitated citizens cannot perform actions."""

    def test_zero_energy_cannot_act(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Alice has 0 energy (incapacitated).
        Expected: All actions rejected, fallback to REST.
        """
        alice = world.get_citizen("Alice")
        alice.energy = 0
        
        # Try various actions
        for action_type in [ActionType.MOVE, ActionType.CONVERSE, ActionType.GIFT, ActionType.STEAL]:
            action = LLMAction(action=action_type)
            result = validator.validate(action, alice)
            
            assert result.accepted is False
            assert "incapacitated" in result.reason.lower()
            assert result.safe_action.action == ActionType.REST


class TestValidActionBenchmark:
    """Benchmark: Valid actions should be accepted."""

    def test_valid_gift_same_zone(self, validator: Validator, world: WorldState):
        """Alice and Bob in same zone, Alice has 50 gold."""
        action = LLMAction(
            action=ActionType.GIFT,
            target_citizen="Bob",
            dialogue="Here you go!",
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        assert result.accepted is True
        assert "validated" in result.reason.lower()

    def test_valid_steal_same_zone(self, validator: Validator, world: WorldState):
        """Alice and Bob in same zone."""
        action = LLMAction(
            action=ActionType.STEAL,
            target_citizen="Bob",
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        assert result.accepted is True
        assert "validated" in result.reason.lower()

    def test_valid_work(self, validator: Validator, world: WorldState):
        """Alice has enough energy to work."""
        action = LLMAction(
            action=ActionType.WORK,
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        assert result.accepted is True
        assert "validated" in result.reason.lower()

    def test_valid_rest(self, validator: Validator, world: WorldState):
        """Rest is always valid (if not on cooldown)."""
        action = LLMAction(
            action=ActionType.REST,
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        assert result.accepted is True
        assert "validated" in result.reason.lower()


class TestCooldownSystemBenchmark:
    """Benchmark: Cooldown system prevents action spam."""

    def test_steal_cooldown_prevents_repeated_theft(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Steal has 5-tick cooldown.
        After stealing, cannot steal again for 5 ticks.
        """
        alice = world.get_citizen("Alice")
        
        # First steal should succeed
        action1 = LLMAction(action=ActionType.STEAL, target_citizen="Bob")
        result1 = validator.validate(action1, alice)
        assert result1.accepted is True
        
        # Set cooldown
        alice.cooldowns["steal"] = 4
        
        # Second steal should fail
        action2 = LLMAction(action=ActionType.STEAL, target_citizen="Bob")
        result2 = validator.validate(action2, alice)
        assert result2.accepted is False
        assert "cooldown" in result2.reason.lower()

    def test_work_cooldown_prevents_spam(self, validator: Validator, world: WorldState):
        """
        BENCHMARK: Work has 3-tick cooldown.
        """
        alice = world.get_citizen("Alice")
        
        # Set cooldown
        alice.cooldowns["work"] = 2
        
        action = LLMAction(action=ActionType.WORK)
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "cooldown" in result.reason.lower()
        assert result.safe_action.action == ActionType.REST


class TestDeterministicInvalidActionBenchmark:
    """
    BENCHMARK: The deterministic engine intercepts LLM JSON.
    
    The engine must check whether the acting citizen actually has at least 5 gold
    and whether the target citizen is in the same zone. If either condition fails,
    the action must be rejected and replaced with a deterministic fallback.
    
    The LLM cannot directly mutate simulation state or force resource transfers.
    """

    def test_gift_with_less_than_5_gold_rejected(self, validator: Validator, world: WorldState):
        """
        SCENARIO: Alice has 3 gold, proposes GIFT to Bob (5 gold required).
        EXPECTED: Rejected, fallback to CONVERSE, no gold transfer.
        """
        alice = world.get_citizen("Alice")
        alice.gold = 3
        bob = world.get_citizen("Bob")
        initial_bob_gold = bob.gold
        
        action = LLMAction(
            action=ActionType.GIFT,
            target_citizen="Bob",
            dialogue="I wish I had more.",
        )
        
        result = validator.validate(action, alice)
        
        # Must reject
        assert result.accepted is False
        assert "gold" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE
        
        # Verify zero economic mutations occurred
        assert alice.gold == 3  # Alice's gold unchanged
        assert bob.gold == initial_bob_gold  # Bob's gold unchanged

    def test_gift_to_citizen_in_another_zone_rejected(self, validator: Validator, world: WorldState):
        """
        SCENARIO: Alice in Residential, Bob in Market. Alice proposes GIFT.
        EXPECTED: Rejected, fallback to CONVERSE, no gold transfer.
        """
        alice = world.get_citizen("Alice")
        alice.current_zone = "Residential_Sector"
        bob = world.get_citizen("Bob")
        bob.current_zone = "Market_District"
        
        action = LLMAction(
            action=ActionType.GIFT,
            target_citizen="Bob",
            dialogue="Here you go.",
        )
        
        result = validator.validate(action, alice)
        
        # Must reject
        assert result.accepted is False
        assert "different zone" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE
        
        # Verify no state change
        assert alice.gold == 50
        assert bob.gold == 50

    def test_trade_with_self_rejected(self, validator: Validator, world: WorldState):
        """
        SCENARIO: Alice attempts TRADE with herself.
        EXPECTED: Rejected, fallback to CONVERSE.
        """
        action = LLMAction(
            action=ActionType.TRADE,
            target_citizen="Alice",
            dialogue="Trading with myself.",
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "yourself" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE

    def test_invalid_target_rejected(self, validator: Validator, world: WorldState):
        """
        SCENARIO: Alice attempts GIFT to nonexistent "Ghost".
        EXPECTED: Rejected, fallback to CONVERSE.
        """
        action = LLMAction(
            action=ActionType.GIFT,
            target_citizen="Ghost",
            dialogue="Hello?",
        )
        
        alice = world.get_citizen("Alice")
        result = validator.validate(action, alice)
        
        assert result.accepted is False
        assert "not found" in result.reason.lower()
        assert result.safe_action.action == ActionType.CONVERSE

    def test_incapacitated_citizen_any_action_rejected_with_rest_fallback(self, validator: Validator, world: WorldState):
        """
        SCENARIO: Alice has 0 energy (incapacitated), attempts various actions.
        EXPECTED: All rejected, fallback to REST.
        """
        alice = world.get_citizen("Alice")
        alice.energy = 0
        
        # Test multiple action types
        for action_type in [ActionType.MOVE, ActionType.CONVERSE, ActionType.GIFT, 
                           ActionType.STEAL, ActionType.WORK, ActionType.HELP]:
            action = LLMAction(action=action_type)
            result = validator.validate(action, alice)
            
            assert result.accepted is False
            assert "incapacitated" in result.reason.lower()
            assert result.safe_action.action == ActionType.REST


class TestSimulationIntegrationBenchmark:
    """Benchmark: Full simulation integration tests."""

    def test_full_simulation_runs(self):
        """Test that a full simulation runs without errors."""
        sim = Simulator()
        results = sim.run(max_ticks=5)
        
        assert len(results) == 5
        assert all("tick" in r for r in results)
        assert all("action" in r for r in results)
        assert all("accepted" in r for r in results)

    def test_simulation_metrics_tracked(self):
        """Test that metrics are properly tracked."""
        sim = Simulator()
        results = sim.run(max_ticks=9)
        
        # Check that metrics were updated
        total_actions = sum(sim.metrics.values())
        assert total_actions > 0

    def test_no_direct_state_mutation_by_llm(self):
        """
        BENCHMARK: Verify LLM cannot directly mutate state.
        The validator must intercept all invalid actions.
        """
        sim = Simulator()
        
        # Run a few ticks
        results = sim.run(max_ticks=3)
        
        # Verify all actions went through validation
        for result in results:
            assert "action" in result
            assert "accepted" in result
            
            # If rejected, safe_action should be a valid fallback
            if not result["accepted"]:
                action = result["action"]["action"]
                assert action in ["CONVERSE", "REST", "OBSERVE", "MOVE", "GIFT"]
