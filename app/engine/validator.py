"""
Validation layer for LLM-proposed actions.

The validator is the only trusted authority in the simulation.
It checks every proposed action against the current world state
and replaces invalid actions with a safe default (CONVERSE).

The validator NEVER modifies state itself. It only produces
ValidationResult objects.
"""

from app.models.actions import (
    ActionType,
    LLMAction,
    ValidationResult,
)
from app.models.citizen import Citizen
from app.models.world import WorldState
from app.config import settings
from app.engine.rules import (
    MIN_ENERGY_FOR_WORK,
    MIN_ENERGY_FOR_EXPLORE,
    MIN_ENERGY_FOR_STEAL,
    MIN_GOLD_FOR_GIFT,
    MIN_GOLD_FOR_HELP,
    MIN_GOLD_FOR_TRADE,
    HELP_ENERGY_AMOUNT,
    COOLDOWN_REST,
    COOLDOWN_WORK,
    COOLDOWN_STEAL,
)


class Validator:
    """
    Validates LLM-proposed actions against the current world state.

    Responsibilities:
    - Schema validation (handled by Pydantic before reaching here)
    - Action-specific rule validation
    - Resource validation (gold, energy)
    - Zone validation (target must be in same zone, destination must exist)
    - Target validation (target must be a real citizen)

    If any validation fails, the action is replaced with CONVERSE
    and the reason is recorded.
    """

    def __init__(self, world: WorldState):
        self.world = world

    def validate(self, action: LLMAction, actor: Citizen) -> ValidationResult:
        """
        Validate a proposed action against the current world state.

        Args:
            action: The LLM-proposed action.
            actor: The citizen performing the action.

        Returns:
            A ValidationResult indicating acceptance or rejection
            with a safe fallback action.
        """
        action_type = action.action

        # REST is always allowed, even when incapacitated (it's the recovery mechanic)
        if action_type == ActionType.REST:
            return self._validate_rest(action, actor)
        
        # Check if actor is alive for all non-REST actions
        if actor.energy <= 0:
            return ValidationResult(
                accepted=False,
                reason=f"{actor.name} is incapacitated (energy={actor.energy}). Cannot perform actions. REST to recover.",
                error_code="INCAPACITATED",
                safe_action=LLMAction(action=ActionType.REST),
            )

        # Route to specific validator
        if action_type == ActionType.MOVE:
            return self._validate_move(action, actor)
        elif action_type == ActionType.EXPLORE:
            return self._validate_explore(action, actor)
        elif action_type == ActionType.FOLLOW:
            return self._validate_follow(action, actor)
        elif action_type == ActionType.CONVERSE:
            return self._validate_converse(action, actor)
        elif action_type == ActionType.GIFT:
            return self._validate_gift(action, actor)
        elif action_type == ActionType.HELP:
            return self._validate_help(action, actor)
        elif action_type == ActionType.REQUEST_HELP:
            return self._validate_request_help(action, actor)
        elif action_type == ActionType.WARN:
            return self._validate_warn(action, actor)
        elif action_type == ActionType.TRADE:
            return self._validate_trade(action, actor)
        elif action_type == ActionType.IGNORE:
            return self._validate_ignore(action, actor)
        elif action_type == ActionType.WORK:
            return self._validate_work(action, actor)
        elif action_type == ActionType.STEAL:
            return self._validate_steal(action, actor)
        elif action_type == ActionType.REST:
            return self._validate_rest(action, actor)
        elif action_type == ActionType.OBSERVE:
            return self._validate_observe(action, actor)
        else:
            return ValidationResult(
                accepted=False,
                reason=f"Unknown action type: {action_type}",
                error_code="UNKNOWN_ACTION",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

    def _validate_move(self, action: LLMAction, actor: Citizen) -> ValidationResult:
        """Validate MOVE action: destination must be a valid zone."""
        destination = action.target_zone

        if destination is None or destination.strip() == "":
            return ValidationResult(
                accepted=False,
                reason="MOVE requires a destination zone in target_zone field.",
                error_code="MOVE_NO_DESTINATION",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        # Normalize: zones in settings are in Title_Case
        normalized_dest = destination.strip().replace(" ", "_").title()
        valid_zones = settings.zones

        if normalized_dest not in valid_zones:
            return ValidationResult(
                accepted=False,
                reason=f"Invalid destination '{destination}'. Valid zones: {', '.join(valid_zones)}",
                error_code="MOVE_INVALID_ZONE",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        # Create a corrected action with normalized zone
        corrected = action.model_copy(update={"target_zone": normalized_dest})
        return ValidationResult(
            accepted=True,
            reason="Move validated.",
            error_code="OK",
            safe_action=corrected,
        )

    def _validate_converse(self, action: LLMAction, actor: Citizen) -> ValidationResult:
        """Validate CONVERSE action: target is optional, invalidate nonexistent targets."""
        target_name = action.target_citizen
        
        # If target specified, validate it exists
        if target_name is not None:
            target = self.world.get_citizen(target_name)
            if target is None:
                # Invalidate the target but still accept the action
                corrected = action.model_copy(update={"target_citizen": None})
                return ValidationResult(
                    accepted=True,
                    reason=f"Target citizen '{target_name}' not found. Converse accepted without target.",
                    error_code="CONVERSE_BAD_TARGET",
                    safe_action=corrected,
                )
        
        return ValidationResult(
            accepted=True,
            reason="Converse accepted.",
            error_code="OK",
            safe_action=action,
        )

    def _validate_gift(self, action: LLMAction, actor: Citizen) -> ValidationResult:
        """Validate GIFT action: must have target, same zone, enough gold."""
        target_name = action.target_citizen
        if target_name is None:
            return ValidationResult(
                accepted=False,
                reason="GIFT requires a target citizen.",
                error_code="GIFT_NO_TARGET",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        # Cannot gift to self
        if target_name == actor.name:
            return ValidationResult(
                accepted=False,
                reason="Cannot gift to yourself.",
                error_code="GIFT_SELF",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        target = self.world.get_citizen(target_name)
        if target is None:
            return ValidationResult(
                accepted=False,
                reason=f"Target citizen '{target_name}' not found. Valid citizens: {self._valid_citizen_names()}",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        # Check same zone
        if target.current_zone != actor.current_zone:
            return ValidationResult(
                accepted=False,
                reason=f"Cannot gift to '{target_name}' in different zone '{target.current_zone}'. "
                       f"Both must be in '{actor.current_zone}'.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        # Check enough gold
        if actor.gold < MIN_GOLD_FOR_GIFT:
            return ValidationResult(
                accepted=False,
                reason=f"Not enough gold to gift. Has {actor.gold}, needs at least {MIN_GOLD_FOR_GIFT}.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        return ValidationResult(
            accepted=True,
            reason="Gift validated.",
            safe_action=action,
        )

    def _validate_steal(self, action: LLMAction, actor: Citizen) -> ValidationResult:
        """Validate STEAL action: must have target in same zone, enough energy, not on cooldown."""
        target_name = action.target_citizen
        if target_name is None:
            return ValidationResult(
                accepted=False,
                reason="STEAL requires a target citizen.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        # Cannot steal from self
        if target_name == actor.name:
            return ValidationResult(
                accepted=False,
                reason="Cannot steal from yourself.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        target = self.world.get_citizen(target_name)
        if target is None:
            return ValidationResult(
                accepted=False,
                reason=f"Target citizen '{target_name}' not found. Valid citizens: {self._valid_citizen_names()}",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        # Check same zone
        if target.current_zone != actor.current_zone:
            return ValidationResult(
                accepted=False,
                reason=f"Cannot steal from '{target_name}' in different zone '{target.current_zone}'. "
                       f"Both must be in '{actor.current_zone}'.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        # Check enough energy
        if actor.energy < MIN_ENERGY_FOR_STEAL:
            return ValidationResult(
                accepted=False,
                reason=f"Not enough energy to steal. Has {actor.energy}, needs at least {MIN_ENERGY_FOR_STEAL}.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        # Check cooldown
        if "steal" in actor.cooldowns and actor.cooldowns["steal"] > 0:
            return ValidationResult(
                accepted=False,
                reason=f"Steal is on cooldown for {actor.cooldowns['steal']} more ticks.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        return ValidationResult(
            accepted=True,
            reason="Steal validated.",
            safe_action=action,
        )

    def _validate_work(self, action: LLMAction, actor: Citizen) -> ValidationResult:
        """Validate WORK action: must have enough energy, not on cooldown."""
        if actor.energy < MIN_ENERGY_FOR_WORK:
            return ValidationResult(
                accepted=False,
                reason=f"Not enough energy to work. Has {actor.energy}, needs at least {MIN_ENERGY_FOR_WORK}.",
                safe_action=LLMAction(action=ActionType.REST),
            )

        if "work" in actor.cooldowns and actor.cooldowns["work"] > 0:
            return ValidationResult(
                accepted=False,
                reason=f"Work is on cooldown for {actor.cooldowns['work']} more ticks.",
                safe_action=LLMAction(action=ActionType.REST),
            )

        return ValidationResult(
            accepted=True,
            reason="Work validated.",
            safe_action=action,
        )

    def _validate_rest(self, action: LLMAction, actor: Citizen) -> ValidationResult:
        """Validate REST action: check cooldown."""
        if "rest" in actor.cooldowns and actor.cooldowns["rest"] > 0:
            return ValidationResult(
                accepted=False,
                reason=f"Rest is on cooldown for {actor.cooldowns['rest']} more ticks.",
                safe_action=LLMAction(action=ActionType.OBSERVE),
            )

        return ValidationResult(
            accepted=True,
            reason="Rest validated.",
            safe_action=action,
        )

    def _validate_explore(self, action: LLMAction, actor: Citizen) -> ValidationResult:
        """Validate EXPLORE action: must have enough energy, valid destination."""
        if actor.energy < MIN_ENERGY_FOR_EXPLORE:
            return ValidationResult(
                accepted=False,
                reason=f"Not enough energy to explore. Has {actor.energy}, needs at least {MIN_ENERGY_FOR_EXPLORE}.",
                safe_action=LLMAction(action=ActionType.MOVE),
            )

        destination = action.target_zone
        if destination is None:
            return ValidationResult(
                accepted=False,
                reason="EXPLORE requires a destination zone in target_zone field.",
                safe_action=LLMAction(action=ActionType.MOVE),
            )

        normalized_dest = destination.strip().replace(" ", "_").title()
        valid_zones = settings.zones

        if normalized_dest not in valid_zones:
            return ValidationResult(
                accepted=False,
                reason=f"Invalid destination '{destination}'. Valid zones: {', '.join(valid_zones)}",
                safe_action=LLMAction(action=ActionType.MOVE),
            )

        return ValidationResult(
            accepted=True,
            reason="Explore validated.",
            safe_action=action.model_copy(update={"target_zone": normalized_dest}),
        )

    def _validate_follow(self, action: LLMAction, actor: Citizen) -> ValidationResult:
        """Validate FOLLOW action: target must exist and be in same zone."""
        target_name = action.target_citizen
        if target_name is None:
            return ValidationResult(
                accepted=False,
                reason="FOLLOW requires a target citizen.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        if target_name == actor.name:
            return ValidationResult(
                accepted=False,
                reason="Cannot follow yourself.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        target = self.world.get_citizen(target_name)
        if target is None:
            return ValidationResult(
                accepted=False,
                reason=f"Target citizen '{target_name}' not found.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        if target.current_zone != actor.current_zone:
            return ValidationResult(
                accepted=False,
                reason=f"Cannot follow '{target_name}' - not in same zone.",
                safe_action=LLMAction(action=ActionType.MOVE),
            )

        return ValidationResult(
            accepted=True,
            reason="Follow validated.",
            safe_action=action,
        )

    def _validate_help(self, action: LLMAction, actor: Citizen) -> ValidationResult:
        """Validate HELP action: must have enough gold and energy, target in same zone."""
        target_name = action.target_citizen
        if target_name is None:
            return ValidationResult(
                accepted=False,
                reason="HELP requires a target citizen.",
                safe_action=LLMAction(action=ActionType.GIFT),
            )

        if target_name == actor.name:
            return ValidationResult(
                accepted=False,
                reason="Cannot help yourself.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        target = self.world.get_citizen(target_name)
        if target is None:
            return ValidationResult(
                accepted=False,
                reason=f"Target citizen '{target_name}' not found.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        if target.current_zone != actor.current_zone:
            return ValidationResult(
                accepted=False,
                reason=f"Cannot help '{target_name}' in different zone.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        if actor.gold < MIN_GOLD_FOR_HELP:
            return ValidationResult(
                accepted=False,
                reason=f"Not enough gold to help. Has {actor.gold}, needs {MIN_GOLD_FOR_HELP}.",
                safe_action=LLMAction(action=ActionType.GIFT),
            )

        if actor.energy < HELP_ENERGY_AMOUNT:
            return ValidationResult(
                accepted=False,
                reason=f"Not enough energy to help. Has {actor.energy}, needs {HELP_ENERGY_AMOUNT}.",
                safe_action=LLMAction(action=ActionType.GIFT),
            )

        return ValidationResult(
            accepted=True,
            reason="Help validated.",
            safe_action=action,
        )

    def _validate_request_help(self, action: LLMAction, actor: Citizen) -> ValidationResult:
        """Validate REQUEST_HELP action: target must be in same zone."""
        target_name = action.target_citizen
        if target_name is None:
            return ValidationResult(
                accepted=False,
                reason="REQUEST_HELP requires a target citizen.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        if target_name == actor.name:
            return ValidationResult(
                accepted=False,
                reason="Cannot request help from yourself.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        target = self.world.get_citizen(target_name)
        if target is None:
            return ValidationResult(
                accepted=False,
                reason=f"Target citizen '{target_name}' not found.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        if target.current_zone != actor.current_zone:
            return ValidationResult(
                accepted=False,
                reason=f"Cannot request help from '{target_name}' in different zone.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        return ValidationResult(
            accepted=True,
            reason="Request help validated.",
            safe_action=action,
        )

    def _validate_warn(self, action: LLMAction, actor: Citizen) -> ValidationResult:
        """Validate WARN action: target must be in same zone."""
        target_name = action.target_citizen
        if target_name is None:
            return ValidationResult(
                accepted=False,
                reason="WARN requires a target citizen.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        if target_name == actor.name:
            return ValidationResult(
                accepted=False,
                reason="Cannot warn yourself.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        target = self.world.get_citizen(target_name)
        if target is None:
            return ValidationResult(
                accepted=False,
                reason=f"Target citizen '{target_name}' not found.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        if target.current_zone != actor.current_zone:
            return ValidationResult(
                accepted=False,
                reason=f"Cannot warn '{target_name}' in different zone.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        return ValidationResult(
            accepted=True,
            reason="Warn validated.",
            safe_action=action,
        )

    def _validate_trade(self, action: LLMAction, actor: Citizen) -> ValidationResult:
        """Validate TRADE action: must have enough gold, target in same zone."""
        target_name = action.target_citizen
        if target_name is None:
            return ValidationResult(
                accepted=False,
                reason="TRADE requires a target citizen.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        if target_name == actor.name:
            return ValidationResult(
                accepted=False,
                reason="Cannot trade with yourself.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        target = self.world.get_citizen(target_name)
        if target is None:
            return ValidationResult(
                accepted=False,
                reason=f"Target citizen '{target_name}' not found.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        if target.current_zone != actor.current_zone:
            return ValidationResult(
                accepted=False,
                reason=f"Cannot trade with '{target_name}' in different zone.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        if actor.gold < MIN_GOLD_FOR_TRADE:
            return ValidationResult(
                accepted=False,
                reason=f"Not enough gold to trade. Has {actor.gold}, needs {MIN_GOLD_FOR_TRADE}.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        return ValidationResult(
            accepted=True,
            reason="Trade validated.",
            safe_action=action,
        )

    def _validate_ignore(self, action: LLMAction, actor: Citizen) -> ValidationResult:
        """Validate IGNORE action: target must be in same zone."""
        target_name = action.target_citizen
        if target_name is None:
            return ValidationResult(
                accepted=False,
                reason="IGNORE requires a target citizen.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        if target_name == actor.name:
            return ValidationResult(
                accepted=False,
                reason="Cannot ignore yourself.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        target = self.world.get_citizen(target_name)
        if target is None:
            return ValidationResult(
                accepted=False,
                reason=f"Target citizen '{target_name}' not found.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        if target.current_zone != actor.current_zone:
            return ValidationResult(
                accepted=False,
                reason=f"Cannot ignore '{target_name}' in different zone.",
                safe_action=LLMAction(action=ActionType.CONVERSE),
            )

        return ValidationResult(
            accepted=True,
            reason="Ignore validated.",
            safe_action=action,
        )

    def _validate_observe(self, action: LLMAction, actor: Citizen) -> ValidationResult:
        """Validate OBSERVE action: always valid."""
        return ValidationResult(
            accepted=True,
            reason="Observe validated.",
            safe_action=action,
        )

    def _valid_citizen_names(self) -> str:
        """Return a comma-separated string of valid citizen names."""
        return ", ".join(sorted(c.name for c in self.world.citizens.values()))
