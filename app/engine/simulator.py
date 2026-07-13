"""
Main simulation engine that orchestrates the entire simulation flow.

The simulator follows the strict architecture:
    CLI -> LLM -> Parser -> Validator -> Mutations -> State Store -> Chronicle

The LLM NEVER modifies state directly. It only proposes actions.
Only the deterministic validator may mutate the simulation.
"""

import sys
from typing import Any, Union

from app.config import settings
from app.engine.chronicle import generate_chronicle
from app.engine.mutations import (
    GoldMutation,
    EnergyMutation,
    MoveMutation,
    RelationshipMutation,
    DialogueMutation,
    RestMutation,
    WorkMutation,
    ObserveMutation,
    ExploreMutation,
    HelpMutation,
    TradeMutation,
    RequestHelpMutation,
    WarnMutation,
    FollowMutation,
    IgnoreMutation,
    Mutation,
    mutations_to_dicts,
)
from app.engine.rules import (
    ENERGY_DECAY_PER_TICK,
    GIFT_GOLD_AMOUNT,
    STEAL_GOLD_AMOUNT,
    HELP_GOLD_AMOUNT,
    HELP_ENERGY_AMOUNT,
    WORK_GOLD_EARNED,
    WORK_ENERGY_COST,
    REST_ENERGY_GAIN,
    TRADE_GOLD_AMOUNT,
    ZONE_REST_BONUS,
    ZONE_WORK_BONUS,
    ZONE_SOCIAL_BONUS,
    REL_AFFINITY_GIFT,
    REL_TRUST_GIFT,
    REL_AFFINITY_STEAL,
    REL_TRUST_STEAL,
    REL_AFFINITY_CONVERSE,
    REL_TRUST_CONVERSE,
    REL_AFFINITY_HELP,
    REL_TRUST_HELP,
    REL_AFFINITY_WARN,
    REL_TRUST_WARN,
    REL_AFFINITY_TRADE,
    REL_TRUST_TRADE,
    REL_AFFINITY_REQUEST_HELP,
    REL_TRUST_REQUEST_HELP,
    REL_AFFINITY_IGNORE,
    REL_TRUST_IGNORE,
    COOLDOWN_REST,
    COOLDOWN_WORK,
    COOLDOWN_STEAL,
)
from app.engine.validator import Validator
from app.llm.parser import parse_llm_response
from app.llm.prompt_builder import build_prompt
from app.llm.provider import LLMProvider
from app.logger import SimulationLogger
from app.models.actions import (
    ActionType,
    LLMAction,
    ParserError,
    ValidationResult,
)
from app.models.citizen import Citizen, PersonalityTensor
from app.models.world import WorldState


class MockProvider(LLMProvider):
    """A mock LLM provider for testing without a real LLM."""

    def __init__(self, action: LLMAction | None = None):
        self.action = action or LLMAction(
            action=ActionType.CONVERSE,
            dialogue="Testing the simulation.",
        )

    def generate(self, prompt: str) -> str:
        """Return the pre-configured action as a JSON string."""
        return self.action.model_dump_json(exclude_none=True)


InjectionScenario = dict[str, Any]


class Simulator:
    """
    Orchestrates the simulation loop.

    The simulator owns the WorldState and coordinates the flow:
    energy decay -> citizen selection -> prompt -> LLM -> parse
    -> validate -> mutate -> chronicle -> log
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        logger: SimulationLogger | None = None,
        injection_scenarios: list[InjectionScenario] | None = None,
    ):
        self.logger = logger or SimulationLogger()
        self.llm_provider = llm_provider or self._create_provider()
        self.world = self._create_initial_state()
        self.validator = Validator(self.world)
        self.injection_scenarios = injection_scenarios or []
        self.rejection_reasons: dict[str, int] = {}
        self.metrics: dict[str, Union[int, float]] = {
            "moves": 0,
            "explores": 0,
            "follows": 0,
            "conversations": 0,
            "gifts": 0,
            "helps": 0,
            "requests_help": 0,
            "warns": 0,
            "trades": 0,
            "ignores": 0,
            "works": 0,
            "rests": 0,
            "observes": 0,
            "thefts": 0,
            "accepted_actions": 0,
            "rejected_actions": 0,
            "total_gold_gifted": 0,
            "total_gold_stolen": 0,
            "total_gold_traded": 0,
            "total_gold_earned": 0,
        }

    def _create_provider(self) -> LLMProvider:
        """Create the appropriate LLM provider based on configuration."""
        provider_name = settings.llm_provider

        if provider_name == "ollama":
            from app.llm.ollama_provider import OllamaProvider
            return OllamaProvider()
        elif provider_name == "gemini":
            from app.llm.gemini_provider import GeminiProvider
            return GeminiProvider()
        elif provider_name == "openai":
            from app.llm.openai_provider import OpenAIProvider
            return OpenAIProvider()
        else:
            print(
                f"Unknown LLM provider '{provider_name}'. "
                f"Falling back to MockProvider.",
                file=sys.stderr,
            )
            return MockProvider()

    def _create_initial_state(self) -> WorldState:
        """Create the initial world state with citizens and zones."""
        citizens_data = [
            {
                "id": "citizen_1",
                "name": "Alice",
                "current_zone": "Residential_Sector",
                "gold": 50,
                "energy": 10,
                "personality": PersonalityTensor(
                    sociability=0.8,
                    aggression=0.2,
                    empathy=0.9,
                ),
            },
            {
                "id": "citizen_2",
                "name": "Bob",
                "current_zone": "Market_District",
                "gold": 50,
                "energy": 10,
                "personality": PersonalityTensor(
                    sociability=0.5,
                    aggression=0.7,
                    empathy=0.3,
                ),
            },
            {
                "id": "citizen_3",
                "name": "Charlie",
                "current_zone": "City_Park",
                "gold": 50,
                "energy": 10,
                "personality": PersonalityTensor(
                    sociability=0.3,
                    aggression=0.4,
                    empathy=0.6,
                ),
            },
        ]

        citizens: dict[str, Citizen] = {}
        for data in citizens_data:
            citizen = Citizen(**data)
            # Use citizen name as the canonical key
            citizens[citizen.name] = citizen

        # Initialize relationships using names as canonical identifiers
        citizen_names = list(citizens.keys())
        for name in citizen_names:
            for other_name in citizen_names:
                if name != other_name:
                    citizens[name].get_relationship(other_name)

        return WorldState(citizens=citizens)

    def _decay_energy(self, active_citizen: Citizen) -> list[Mutation]:
        """Apply energy decay ONLY to the active citizen.
        
        Only the citizen who gets to act this tick loses energy from
        the passage of time. Non-active citizens conserve their energy
        between turns. This prevents the death spiral where all citizens
        reach zero simultaneously and spend the rest of the simulation
        cycling through forced REST recovery.
        """
        mutations: list[Mutation] = []
        if active_citizen.energy > 0:
            active_citizen.energy -= ENERGY_DECAY_PER_TICK
            mutations.append(EnergyMutation(
                citizen_name=active_citizen.name,
                delta=-ENERGY_DECAY_PER_TICK,
                description=f"Energy decay for {active_citizen.name}",
            ))
        return mutations

    def _select_active_citizen(self) -> Citizen:
        """
        Select the active citizen for this tick using round-robin.

        Returns the citizen with the lowest tick count in history.
        """
        # Simple round-robin: cycle through citizens by name
        names = sorted(self.world.citizens.keys())
        tick = self.world.tick
        name = names[tick % len(names)]
        return self.world.citizens[name]

    def _derive_mutations(
        self,
        action: LLMAction,
        actor: Citizen,
    ) -> list[Mutation]:
        """
        Derive deterministic mutations from a validated action.

        This is where the actual state change logic lives. The LLM's
        intended_state_delta is COMPLETELY IGNORED.

        Args:
            action: The validated action.
            actor: The citizen performing the action.

        Returns:
            A list of immutable Mutation objects.
        """
        mutations: list[Mutation] = []
        action_type = action.action

        if action_type == ActionType.MOVE:
            destination = action.target_zone or "Residential_Sector"
            mutations.append(MoveMutation(
                citizen_name=actor.name,
                destination=destination,
                description=f"{actor.name} moves to {destination}",
            ))

        elif action_type == ActionType.EXPLORE:
            destination = action.target_zone or "Residential_Sector"
            mutations.append(ExploreMutation(
                citizen_name=actor.name,
                destination=destination,
                energy_cost=2,
                description=f"{actor.name} explores {destination}",
            ))
            mutations.append(EnergyMutation(
                citizen_name=actor.name,
                delta=-2,
                description=f"{actor.name} spends energy exploring",
            ))

        elif action_type == ActionType.FOLLOW:
            target_name = action.target_citizen
            if target_name:
                target = self.world.get_citizen(target_name)
                if target:
                    mutations.append(FollowMutation(
                        citizen_name=actor.name,
                        target_name=target_name,
                        description=f"{actor.name} follows {target_name}",
                    ))
                    mutations.append(MoveMutation(
                        citizen_name=actor.name,
                        destination=target.current_zone,
                        description=f"{actor.name} moves to {target.current_zone}",
                    ))

        elif action_type == ActionType.CONVERSE:
            target_name = action.target_citizen
            # Apply zone social bonus
            social_bonus = ZONE_SOCIAL_BONUS.get(actor.current_zone, 0.0)
            affinity_delta = REL_AFFINITY_CONVERSE + social_bonus
            
            if target_name:
                mutations.append(RelationshipMutation(
                    citizen_name=actor.name,
                    target_name=target_name,
                    affinity_delta=affinity_delta,
                    trust_delta=REL_TRUST_CONVERSE,
                    description=f"{actor.name} converses with {target_name}",
                ))
            if action.dialogue:
                mutations.append(DialogueMutation(
                    citizen_name=actor.name,
                    dialogue=action.dialogue,
                    target_name=target_name or "",
                    description=f"{actor.name} speaks",
                ))

        elif action_type == ActionType.GIFT:
            target_name = action.target_citizen
            if target_name:
                # Transfer gold
                mutations.append(GoldMutation(
                    citizen_name=actor.name,
                    delta=-GIFT_GOLD_AMOUNT,
                    description=f"{actor.name} gives {GIFT_GOLD_AMOUNT} gold to {target_name}",
                ))
                mutations.append(GoldMutation(
                    citizen_name=target_name,
                    delta=GIFT_GOLD_AMOUNT,
                    description=f"{target_name} receives {GIFT_GOLD_AMOUNT} gold from {actor.name}",
                ))
                # Update relationships
                mutations.append(RelationshipMutation(
                    citizen_name=actor.name,
                    target_name=target_name,
                    affinity_delta=REL_AFFINITY_GIFT,
                    trust_delta=REL_TRUST_GIFT,
                    description=f"{actor.name} gains affinity/trust with {target_name}",
                ))
                # Dialogue
                if action.dialogue:
                    mutations.append(DialogueMutation(
                        citizen_name=actor.name,
                        dialogue=action.dialogue,
                        target_name=target_name,
                        description=f"{actor.name} speaks to {target_name}",
                    ))

        elif action_type == ActionType.HELP:
            target_name = action.target_citizen
            if target_name:
                # Give more resources than a simple gift
                mutations.append(GoldMutation(
                    citizen_name=actor.name,
                    delta=-HELP_GOLD_AMOUNT,
                    description=f"{actor.name} helps {target_name} with {HELP_GOLD_AMOUNT} gold",
                ))
                mutations.append(GoldMutation(
                    citizen_name=target_name,
                    delta=HELP_GOLD_AMOUNT,
                    description=f"{target_name} receives {HELP_GOLD_AMOUNT} gold from {actor.name}",
                ))
                mutations.append(EnergyMutation(
                    citizen_name=target_name,
                    delta=HELP_ENERGY_AMOUNT,
                    description=f"{target_name} gains {HELP_ENERGY_AMOUNT} energy from {actor.name}'s help",
                ))
                # Update relationships
                mutations.append(RelationshipMutation(
                    citizen_name=actor.name,
                    target_name=target_name,
                    affinity_delta=REL_AFFINITY_HELP,
                    trust_delta=REL_TRUST_HELP,
                    description=f"{actor.name} gains significant affinity/trust with {target_name}",
                ))
                # Dialogue
                if action.dialogue:
                    mutations.append(DialogueMutation(
                        citizen_name=actor.name,
                        dialogue=action.dialogue,
                        target_name=target_name,
                        description=f"{actor.name} helps {target_name}",
                    ))

        elif action_type == ActionType.REQUEST_HELP:
            target_name = action.target_citizen
            if target_name:
                # Requesting help slightly reduces affinity (you're admitting weakness)
                mutations.append(RelationshipMutation(
                    citizen_name=actor.name,
                    target_name=target_name,
                    affinity_delta=REL_AFFINITY_REQUEST_HELP,
                    trust_delta=REL_TRUST_REQUEST_HELP,
                    description=f"{actor.name} requests help from {target_name}",
                ))
                if action.dialogue:
                    mutations.append(DialogueMutation(
                        citizen_name=actor.name,
                        dialogue=action.dialogue,
                        target_name=target_name,
                        description=f"{actor.name} requests help",
                    ))

        elif action_type == ActionType.WARN:
            target_name = action.target_citizen
            if target_name:
                # Warning increases trust slightly
                mutations.append(RelationshipMutation(
                    citizen_name=actor.name,
                    target_name=target_name,
                    affinity_delta=REL_AFFINITY_WARN,
                    trust_delta=REL_TRUST_WARN,
                    description=f"{actor.name} warns {target_name}",
                ))
                if action.dialogue:
                    mutations.append(DialogueMutation(
                        citizen_name=actor.name,
                        dialogue=action.dialogue,
                        target_name=target_name,
                        description=f"{actor.name} warns {target_name}",
                    ))

        elif action_type == ActionType.TRADE:
            target_name = action.target_citizen
            if target_name:
                # Trade: both give and receive gold (simplified: actor gives TRADE_GOLD_AMOUNT)
                mutations.append(GoldMutation(
                    citizen_name=actor.name,
                    delta=-TRADE_GOLD_AMOUNT,
                    description=f"{actor.name} trades {TRADE_GOLD_AMOUNT} gold with {target_name}",
                ))
                mutations.append(GoldMutation(
                    citizen_name=target_name,
                    delta=TRADE_GOLD_AMOUNT,
                    description=f"{target_name} receives {TRADE_GOLD_AMOUNT} gold from trade with {actor.name}",
                ))
                mutations.append(RelationshipMutation(
                    citizen_name=actor.name,
                    target_name=target_name,
                    affinity_delta=REL_AFFINITY_TRADE,
                    trust_delta=REL_TRUST_TRADE,
                    description=f"{actor.name} trades with {target_name}",
                ))
                if action.dialogue:
                    mutations.append(DialogueMutation(
                        citizen_name=actor.name,
                        dialogue=action.dialogue,
                        target_name=target_name,
                        description=f"{actor.name} trades with {target_name}",
                    ))

        elif action_type == ActionType.IGNORE:
            target_name = action.target_citizen
            if target_name:
                # Ignoring someone reduces relationship
                mutations.append(RelationshipMutation(
                    citizen_name=actor.name,
                    target_name=target_name,
                    affinity_delta=REL_AFFINITY_IGNORE,
                    trust_delta=REL_TRUST_IGNORE,
                    description=f"{actor.name} ignores {target_name}",
                ))

        elif action_type == ActionType.WORK:
            # Earn gold with zone bonus, spend energy
            zone_bonus = ZONE_WORK_BONUS.get(actor.current_zone, 0)
            total_gold = WORK_GOLD_EARNED + zone_bonus
            mutations.append(WorkMutation(
                citizen_name=actor.name,
                gold_earned=total_gold,
                energy_cost=WORK_ENERGY_COST,
                description=f"{actor.name} works in {actor.current_zone} and earns {total_gold} gold",
            ))
            mutations.append(GoldMutation(
                citizen_name=actor.name,
                delta=total_gold,
                description=f"{actor.name} earns {total_gold} gold from work",
            ))
            mutations.append(EnergyMutation(
                citizen_name=actor.name,
                delta=-WORK_ENERGY_COST,
                description=f"{actor.name} spends {WORK_ENERGY_COST} energy working",
            ))

        elif action_type == ActionType.STEAL:
            target_name = action.target_citizen
            if target_name:
                # Transfer gold
                mutations.append(GoldMutation(
                    citizen_name=actor.name,
                    delta=STEAL_GOLD_AMOUNT,
                    description=f"{actor.name} steals {STEAL_GOLD_AMOUNT} gold from {target_name}",
                ))
                mutations.append(GoldMutation(
                    citizen_name=target_name,
                    delta=-STEAL_GOLD_AMOUNT,
                    description=f"{target_name} loses {STEAL_GOLD_AMOUNT} gold to {actor.name}",
                ))
                # Update relationships (from target's perspective toward actor)
                mutations.append(RelationshipMutation(
                    citizen_name=target_name,
                    target_name=actor.name,
                    affinity_delta=REL_AFFINITY_STEAL,
                    trust_delta=REL_TRUST_STEAL,
                    description=f"{target_name} loses affinity/trust with {actor.name}",
                ))

        elif action_type == ActionType.REST:
            # Recover energy with zone bonus
            zone_bonus = ZONE_REST_BONUS.get(actor.current_zone, 0)
            total_energy = REST_ENERGY_GAIN + zone_bonus
            mutations.append(RestMutation(
                citizen_name=actor.name,
                energy_gained=total_energy,
                description=f"{actor.name} rests in {actor.current_zone} and recovers {total_energy} energy",
            ))
            mutations.append(EnergyMutation(
                citizen_name=actor.name,
                delta=total_energy,
                description=f"{actor.name} gains {total_energy} energy from rest",
            ))

        elif action_type == ActionType.OBSERVE:
            # Observe doesn't change state, just records the action
            target_name = action.target_citizen
            mutations.append(ObserveMutation(
                citizen_name=actor.name,
                zone_observed=actor.current_zone,
                description=f"{actor.name} observes the {actor.current_zone}",
            ))

        return mutations

    def _apply_mutations(self, mutations: list[Mutation]) -> None:
        """
        Apply validated mutations to the world state.

        This is the ONLY place where WorldState is modified.
        """
        for mutation in mutations:
            if isinstance(mutation, GoldMutation):
                citizen = self.world.get_citizen(mutation.citizen_name)
                if citizen:
                    citizen.gold = max(0, citizen.gold + mutation.delta)

            elif isinstance(mutation, EnergyMutation):
                citizen = self.world.get_citizen(mutation.citizen_name)
                if citizen:
                    citizen.energy = max(0, citizen.energy + mutation.delta)

            elif isinstance(mutation, MoveMutation):
                citizen = self.world.get_citizen(mutation.citizen_name)
                if citizen:
                    citizen.current_zone = mutation.destination

            elif isinstance(mutation, ExploreMutation):
                citizen = self.world.get_citizen(mutation.citizen_name)
                if citizen:
                    citizen.current_zone = mutation.destination

            elif isinstance(mutation, FollowMutation):
                # Follow is handled by the MoveMutation that should accompany it
                pass

            elif isinstance(mutation, RelationshipMutation):
                citizen = self.world.get_citizen(mutation.citizen_name)
                if citizen:
                    rel = citizen.get_relationship(mutation.target_name)
                    rel.affinity += mutation.affinity_delta
                    rel.trust += mutation.trust_delta
                    rel.clamp()

            # All other mutations (Dialogue, Rest, Work, Observe, Help, etc.) are informational only

    def _update_cooldowns(self, actor: Citizen, action: LLMAction) -> None:
        """Update cooldowns after an action is performed."""
        action_type = action.action.value.lower()
        
        # Set cooldown based on action type
        if action_type == "rest":
            actor.cooldowns["rest"] = COOLDOWN_REST
        elif action_type == "work":
            actor.cooldowns["work"] = COOLDOWN_WORK
        elif action_type == "steal":
            actor.cooldowns["steal"] = COOLDOWN_STEAL
        
        # Decrement all cooldowns
        for key in list(actor.cooldowns.keys()):
            if actor.cooldowns[key] > 0:
                actor.cooldowns[key] -= 1
            if actor.cooldowns[key] <= 0:
                del actor.cooldowns[key]

    def run_tick(self) -> dict[str, Any]:
        """
        Execute one simulation tick.

        Returns a dict with the tick results for logging.
        """
        self.world.tick += 1
        tick = self.world.tick
        self.logger.tick_start(tick)

        # 1. Select active citizen
        actor = self._select_active_citizen()

        # 2. Decay energy (only for the active citizen)
        energy_mutations = self._decay_energy(actor)
        self.logger._console(f"Active citizen: {actor.name} (zone: {actor.current_zone})")

        # 3. Build prompt
        prompt = build_prompt(actor, self.world)
        self.logger.prompt(tick, prompt)

        # 4. Check for injection scenarios
        injection = self._get_injection(tick, actor.name)
        if injection:
            raw_response = injection["raw_response"]
            self.logger._console(f"[INJECTION] Overriding LLM for tick {tick}: {injection['description']}")
        else:
            # 4. Call LLM
            try:
                raw_response = self.llm_provider.generate(prompt)
            except (ConnectionError, RuntimeError) as exc:
                self.logger.error(tick, f"LLM error: {exc}")
                raw_response = '{"action": "CONVERSE", "dialogue": "System error, falling back to converse."}'
        self.logger.raw_response(tick, raw_response)

        # 5. Parse JSON
        parsed = parse_llm_response(raw_response)
        if isinstance(parsed, ParserError):
            self.logger.error(tick, f"Parse error: {parsed.reason}")
            safe_action = LLMAction(action=ActionType.CONVERSE, dialogue="Fallback after parse failure.")
            accepted = False
            reason = f"Parse failed: {parsed.reason}"
            validation = ValidationResult(
                accepted=False,
                reason=reason,
                error_code="PARSE_ERROR",
                safe_action=safe_action,
            )
        else:
            self.logger.parsed_action(tick, parsed.model_dump(mode="json"))
            # 6. Validate
            validation = self.validator.validate(parsed, actor)
            accepted = validation.accepted
            reason = validation.reason
            safe_action = validation.safe_action

        self.logger.validation_result(tick, {
            "accepted": accepted,
            "reason": reason,
            "safe_action": safe_action.model_dump(mode="json", exclude_none=True),
        })

        # 7. Derive mutations from the validated action
        action_mutations = self._derive_mutations(safe_action, actor)
        all_mutations = energy_mutations + action_mutations

        # 8. Apply mutations
        self._apply_mutations(all_mutations)

        # 8.5 Update cooldowns
        self._update_cooldowns(actor, safe_action)

        # Log mutations
        mutation_dicts = mutations_to_dicts(all_mutations)
        self.logger.mutations(tick, mutation_dicts)

        # 9. Generate chronicle (REST is always shown as accepted since it's the recovery mechanic)
        chronicle_accepted = accepted or (safe_action.action == ActionType.REST)
        chronicle_entry = generate_chronicle(
            tick=tick,
            actor=actor,
            action=safe_action,
            accepted=chronicle_accepted,
            world=self.world,
        )
        self.logger.chronicle(tick, chronicle_entry)

        # 10. Save state snapshot
        state_dict = self.world.model_dump_simple()
        self.logger.state_snapshot(tick, state_dict)

        # Record in history
        self.world.history.append({
            "tick": tick,
            "actor": actor.name,
            "action": safe_action.action.value,
            "accepted": accepted,
            "chronicle": chronicle_entry,
        })

        # Update interaction memory
        target_name = safe_action.target_citizen
        if accepted and target_name:
            actor.recent_interactions.append({
                "tick": tick,
                "action": safe_action.action.value,
                "target": target_name,
                "dialogue": safe_action.dialogue if safe_action.dialogue else "",
            })
            # Keep only last 10 interactions
            if len(actor.recent_interactions) > 10:
                actor.recent_interactions.pop(0)

        # Update metrics
        action_type = safe_action.action.value
        if action_type == "MOVE":
            self.metrics["moves"] += 1
        elif action_type == "EXPLORE":
            self.metrics["explores"] += 1
        elif action_type == "FOLLOW":
            self.metrics["follows"] += 1
        elif action_type == "CONVERSE":
            self.metrics["conversations"] += 1
        elif action_type == "GIFT":
            self.metrics["gifts"] += 1
            # Track gold gifted
            for mutation in action_mutations:
                if isinstance(mutation, GoldMutation) and mutation.delta < 0:
                    self.metrics["total_gold_gifted"] += abs(mutation.delta)
        elif action_type == "HELP":
            self.metrics["helps"] += 1
            for mutation in action_mutations:
                if isinstance(mutation, GoldMutation) and mutation.delta < 0:
                    self.metrics["total_gold_gifted"] += abs(mutation.delta)
        elif action_type == "REQUEST_HELP":
            self.metrics["requests_help"] += 1
        elif action_type == "WARN":
            self.metrics["warns"] += 1
        elif action_type == "TRADE":
            self.metrics["trades"] += 1
            for mutation in action_mutations:
                if isinstance(mutation, GoldMutation) and mutation.delta < 0:
                    self.metrics["total_gold_traded"] += abs(mutation.delta)
        elif action_type == "IGNORE":
            self.metrics["ignores"] += 1
        elif action_type == "WORK":
            self.metrics["works"] += 1
            for mutation in action_mutations:
                if isinstance(mutation, GoldMutation) and mutation.delta > 0:
                    self.metrics["total_gold_earned"] += mutation.delta
        elif action_type == "REST":
            self.metrics["rests"] += 1
        elif action_type == "OBSERVE":
            self.metrics["observes"] += 1
        elif action_type == "STEAL":
            self.metrics["thefts"] += 1
            # Track gold stolen
            for mutation in action_mutations:
                if isinstance(mutation, GoldMutation) and mutation.delta > 0:
                    self.metrics["total_gold_stolen"] += mutation.delta

        if accepted:
            self.metrics["accepted_actions"] += 1
        else:
            self.metrics["rejected_actions"] += 1
            # Track rejection reasons
            error_code = getattr(validation, 'error_code', 'UNKNOWN')
            self.rejection_reasons[error_code] = self.rejection_reasons.get(error_code, 0) + 1

        return {
            "tick": tick,
            "actor": actor.name,
            "action": safe_action.model_dump(mode="json", exclude_none=True),
            "accepted": accepted,
            "mutations": mutation_dicts,
            "chronicle": chronicle_entry,
        }

    def _print_metrics(self) -> None:
        """Print simulation metrics to stderr."""
        print("\n" + "=" * 60, file=sys.stderr)
        print("SIMULATION METRICS", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        
        total_actions = sum([
            self.metrics["moves"],
            self.metrics["explores"],
            self.metrics["follows"],
            self.metrics["conversations"],
            self.metrics["gifts"],
            self.metrics["helps"],
            self.metrics["requests_help"],
            self.metrics["warns"],
            self.metrics["trades"],
            self.metrics["ignores"],
            self.metrics["works"],
            self.metrics["rests"],
            self.metrics["observes"],
            self.metrics["thefts"],
        ])
        
        print(f"\nAction Distribution (Total: {total_actions}):", file=sys.stderr)
        print(f"  Moves: {self.metrics['moves']}", file=sys.stderr)
        print(f"  Explores: {self.metrics['explores']}", file=sys.stderr)
        print(f"  Follows: {self.metrics['follows']}", file=sys.stderr)
        print(f"  Conversations: {self.metrics['conversations']}", file=sys.stderr)
        print(f"  Gifts: {self.metrics['gifts']}", file=sys.stderr)
        print(f"  Helps: {self.metrics['helps']}", file=sys.stderr)
        print(f"  Requests for Help: {self.metrics['requests_help']}", file=sys.stderr)
        print(f"  Warns: {self.metrics['warns']}", file=sys.stderr)
        print(f"  Trades: {self.metrics['trades']}", file=sys.stderr)
        print(f"  Ignores: {self.metrics['ignores']}", file=sys.stderr)
        print(f"  Works: {self.metrics['works']}", file=sys.stderr)
        print(f"  Rests: {self.metrics['rests']}", file=sys.stderr)
        print(f"  Observes: {self.metrics['observes']}", file=sys.stderr)
        print(f"  Thefts: {self.metrics['thefts']}", file=sys.stderr)
        
        print(f"\nValidation:", file=sys.stderr)
        print(f"  Accepted Actions: {self.metrics['accepted_actions']}", file=sys.stderr)
        print(f"  Rejected Actions: {self.metrics['rejected_actions']}", file=sys.stderr)
        if total_actions > 0:
            rejection_rate = (self.metrics['rejected_actions'] / total_actions) * 100
            print(f"  Rejection Rate: {rejection_rate:.1f}%", file=sys.stderr)
        
        print(f"\nEconomic Activity:", file=sys.stderr)
        print(f"  Total Gold Gifted: {self.metrics['total_gold_gifted']}", file=sys.stderr)
        print(f"  Total Gold Traded: {self.metrics['total_gold_traded']}", file=sys.stderr)
        print(f"  Total Gold Earned: {self.metrics['total_gold_earned']}", file=sys.stderr)
        print(f"  Total Gold Stolen: {self.metrics['total_gold_stolen']}", file=sys.stderr)
        
        # Calculate average trust and affinity
        total_trust = 0.0
        total_affinity = 0.0
        relationship_count = 0
        
        for citizen in self.world.citizens.values():
            for rel in citizen.relationships.values():
                total_trust += rel.trust
                total_affinity += rel.affinity
                relationship_count += 1
        
        if relationship_count > 0:
            avg_trust = total_trust / relationship_count
            avg_affinity = total_affinity / relationship_count
            print(f"\nSocial Metrics:", file=sys.stderr)
            print(f"  Average Trust: {avg_trust:.3f}", file=sys.stderr)
            print(f"  Average Affinity: {avg_affinity:.3f}", file=sys.stderr)
        
        print("=" * 60, file=sys.stderr)

    def run(self, max_ticks: int | None = None) -> list[dict[str, Any]]:
        """
        Run the full simulation for the specified number of ticks.

        Args:
            max_ticks: Number of ticks to run. Defaults to settings.max_ticks.

        Returns:
            A list of tick result dicts.
        """
        max_ticks = max_ticks or settings.max_ticks
        results: list[dict[str, Any]] = []

        print(f"Starting simulation: {max_ticks} ticks, {len(self.world.citizens)} citizens", file=sys.stderr)
        print(f"Zones: {', '.join(settings.zones)}", file=sys.stderr)
        print(f"LLM Provider: {type(self.llm_provider).__name__}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        for _ in range(max_ticks):
            result = self.run_tick()
            results.append(result)

        self._print_metrics()
        self._print_rejection_summary()
        print("Simulation complete.", file=sys.stderr)

        return results

    def _get_injection(self, tick: int, actor_name: str) -> InjectionScenario | None:
        """Check if there's an injection scenario for this tick."""
        for scenario in self.injection_scenarios:
            if scenario.get("tick") == tick:
                # Apply any actor state overrides
                if "set_actor_gold" in scenario:
                    actor = self.world.get_citizen(actor_name)
                    if actor:
                        actor.gold = scenario["set_actor_gold"]
                if "set_actor_energy" in scenario:
                    actor = self.world.get_citizen(actor_name)
                    if actor:
                        actor.energy = scenario["set_actor_energy"]
                return scenario
        return None

    def _print_rejection_summary(self) -> None:
        """Print summary of all rejection reasons."""
        if not self.rejection_reasons:
            return
        
        print("\n" + "=" * 60, file=sys.stderr)
        print("REJECTION SUMMARY", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for reason, count in sorted(self.rejection_reasons.items()):
            print(f"  {reason}: {count}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
