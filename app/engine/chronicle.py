"""
Chronicle engine for generating deterministic prose descriptions
of simulation events.

The chronicle does NOT use the LLM. It produces clinical, dry,
post-apocalyptic historian prose with slight dark irony.

Accepted actions get detailed entries. Rejected actions get brief,
dismissive entries emphasizing the system's rejection of LLM fiction.
"""

from app.models.actions import ActionType, LLMAction
from app.models.citizen import Citizen
from app.models.world import WorldState


def _format_gold(n: int) -> str:
    """Format gold amount with appropriate phrasing."""
    return f"{n} gold"


def generate_chronicle(
    tick: int,
    actor: Citizen,
    action: LLMAction,
    accepted: bool,
    world: WorldState,
) -> str:
    """
    Generate a deterministic chronicle entry for an action.

    Args:
        tick: The current simulation tick.
        actor: The citizen who performed the action.
        action: The validated action that was executed.
        accepted: Whether the action was accepted or replaced.
        world: The world state after mutation.

    Returns:
        A prose chronicle entry string.
    """
    # REST is always a legitimate recovery action, even when it's a fallback
    if action.action == ActionType.REST:
        return (
            f"Tick {tick}: {actor.name} rested in the {actor.current_zone} "
            f"and recovered energy. A strategic pause in the endless "
            f"struggle for survival. Sometimes the wisest action is "
            f"to simply... stop."
        )
    
    if not accepted:
        return (
            f"Tick {tick}: {actor.name} attempted economic fiction in the "
            f"{actor.current_zone}. Reality declined the proposal with "
            f"bureaucratic finality."
        )

    action_type = action.action
    target_name = action.target_citizen
    target = world.get_citizen(target_name) if target_name else None

    if action_type == ActionType.MOVE:
        return (
            f"Tick {tick}: {actor.name} migrated to the {action.target_zone} "
            f"carrying {_format_gold(actor.gold)} with {actor.energy} energy. "
            f"The city's endless corridors claimed another transient soul."
        )

    elif action_type == ActionType.EXPLORE:
        return (
            f"Tick {tick}: {actor.name} cautiously explored {action.target_zone} "
            f"from {actor.current_zone}. Energy was expended in the pursuit "
            f"of knowledge. Whether the knowledge was worth the cost remains "
            f"subject to interpretation."
        )

    elif action_type == ActionType.FOLLOW:
        if target:
            return (
                f"Tick {tick}: {actor.name} shadowed {target_name} to the "
                f"{actor.current_zone}. A subtle dance of proximity and "
                f"intention. Whether this leads to connection or collision "
                f"remains to be seen."
            )
        return (
            f"Tick {tick}: {actor.name} attempted to follow a phantom. "
            f"The void did not reciprocate the interest."
        )

    elif action_type == ActionType.CONVERSE:
        if target_name and target:
            dialogue_snippet = action.dialogue[:60] if action.dialogue else "exchanged pleasantries"
            zone_bonus = 0.05 if actor.current_zone == "City_Park" else 0.0
            bonus_text = f" The park's tranquility enhanced the exchange (+{zone_bonus:.0%})." if zone_bonus > 0 else ""
            return (
                f"Tick {tick}: {actor.name} engaged {target_name} in "
                f"conversation at the {actor.current_zone}. "
                f'"{dialogue_snippet}"{bonus_text} Words hung in the air, '
                f"carrying the weight of uncertain meaning."
            )
        else:
            return (
                f"Tick {tick}: {actor.name} spoke to the void at the "
                f"{actor.current_zone}. The silence answered in kind. "
                f"Sometimes the only available audience is oneself."
            )

    elif action_type == ActionType.GIFT:
        if target:
            return (
                f"Tick {tick}: {actor.name} bestowed five gold upon "
                f"{target_name} in the {actor.current_zone}. A rare moment "
                f"of generosity in the wasteland. The social fabric "
                f"strengthened by measurable margins."
            )
        return (
            f"Tick {tick}: {actor.name} attempted to give gold to a "
            f"non-existent entity. The gold remained in place. Reality 1, "
            f"Fiction 0."
        )

    elif action_type == ActionType.HELP:
        if target:
            return (
                f"Tick {tick}: {actor.name} provided substantial aid to "
                f"{target_name} in the {actor.current_zone}. Eight gold and "
                f"two energy units were transferred. This level of sacrifice "
                f"represents true compassion. Trust and affinity increased "
                f"significantly."
            )
        return (
            f"Tick {tick}: {actor.name} attempted to help a ghost. "
            f"Good intentions alone cannot alter reality."
        )

    elif action_type == ActionType.REQUEST_HELP:
        if target:
            return (
                f"Tick {tick}: {actor.name} vulnerably requested assistance "
                f"from {target_name} in the {actor.current_zone}. Admitting "
                f"weakness is a risk. The social calculus shifted slightly."
            )
        return (
            f"Tick {tick}: {actor.name} cried out to the emptiness. "
            f"Only echoes responded."
        )

    elif action_type == ActionType.WARN:
        if target:
            return (
                f"Tick {tick}: {actor.name} issued a warning to {target_name} "
                f"in the {actor.current_zone}. Information shared is a gift. "
                f"Trust increased marginally."
            )
        return (
            f"Tick {tick}: {actor.name} warned the void. The void was "
            f"already aware."
        )

    elif action_type == ActionType.TRADE:
        if target:
            return (
                f"Tick {tick}: {actor.name} engaged in commerce with "
                f"{target_name} in the {actor.current_zone}. Ten gold changed "
                f"hands in a mutually beneficial exchange. Economic cooperation "
                f"strengthened their bond."
            )
        return (
            f"Tick {tick}: {actor.name} attempted to trade with reality "
            f"itself. The market did not cooperate."
        )

    elif action_type == ActionType.IGNORE:
        if target:
            return (
                f"Tick {tick}: {actor.name} deliberately ignored {target_name} "
                f"in the {actor.current_zone}. A calculated social withdrawal. "
                f"Relations cooled by design."
            )
        return (
            f"Tick {tick}: {actor.name} ignored the void. The void, "
            f"characteristically, did not notice."
        )

    elif action_type == ActionType.WORK:
        return (
            f"Tick {tick}: {actor.name} labored in the {actor.current_zone} "
            f"and earned {_format_gold(8)}. Sweat and toil converted to "
            f"currency. Three energy units were expended in the process. "
            f"Honest work builds foundations."
        )

    elif action_type == ActionType.STEAL:
        if target:
            return (
                f"Tick {tick}: {actor.name} relieved {target_name} of "
                f"{_format_gold(5)} in the {actor.current_zone}. "
                f"Trust responded exactly as expected — a significant decline "
                f"recorded. The social fabric frayed another thread."
            )
        return (
            f"Tick {tick}: {actor.name} attempted to steal from shadows. "
            f"The shadows had no gold. The universe recorded this as "
            f"intent without impact."
        )

    elif action_type == ActionType.REST:
        return (
            f"Tick {tick}: {actor.name} rested in the {actor.current_zone} "
            f"and recovered energy. A strategic pause in the endless "
            f"struggle for survival. Sometimes the wisest action is "
            f"to simply... stop."
        )

    elif action_type == ActionType.OBSERVE:
        return (
            f"Tick {tick}: {actor.name} observed the {actor.current_zone} "
            f"with quiet attention. Information was gathered, though no "
            f"immediate action was taken. Knowledge is power, even when "
            f"it appears passive."
        )

    return (
        f"Tick {tick}: {actor.name} performed an unclassifiable action "
        f"in the {actor.current_zone}. The chronicle is confused."
    )
