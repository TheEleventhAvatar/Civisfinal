"""
Prompt builder for constructing LLM prompts from simulation state.

Constructs a structured prompt that includes the citizen's current
situation, personality, relationships, the required JSON schema,
and explicit behavioral guidance.

Personality traits influence the prompt description to encourage
the LLM toward fitting actions without hardcoding probabilities.
"""

from app.models.world import WorldState
from app.models.citizen import Citizen
from app.config import settings
from app.engine.rules import REST_ENERGY_GAIN


def _describe_personality(citizen: Citizen) -> str:
    """Convert personality tensor to a descriptive paragraph with behavioral guidance."""
    p = citizen.personality
    traits = []

    if p.sociability >= 0.7:
        traits.append("highly sociable and enjoys interacting with others")
    elif p.sociability <= 0.3:
        traits.append("reserved and prefers solitude, but will act if necessary")
    else:
        traits.append("moderately sociable, comfortable with both solitude and company")

    if p.aggression >= 0.7:
        traits.append("quick to aggression and willing to take what they want by force")
    elif p.aggression <= 0.3:
        traits.append("peaceful and avoids conflict whenever possible")
    else:
        traits.append("generally even-tempered, but capable of standing their ground")

    if p.empathy >= 0.7:
        traits.append("deeply empathetic and considerate of others' feelings — inclined to give")
    elif p.empathy <= 0.3:
        traits.append("lacks empathy and is indifferent to others' suffering — may exploit others")
    else:
        traits.append("moderately empathetic, weighing self-interest against social bonds")

    return f"{citizen.name} is {', '.join(traits)}."


def _describe_relationships(citizen: Citizen, world: WorldState) -> str:
    """Build a summary of the citizen's relationships."""
    if not citizen.relationships:
        return f"{citizen.name} has no established relationships yet."

    lines = []
    for target_name, rel in citizen.relationships.items():
        target = world.get_citizen(target_name)
        display_name = target.name if target else target_name
        affinity_desc = (
            "friendly" if rel.affinity > 0.3
            else "hostile" if rel.affinity < -0.3
            else "neutral"
        )
        trust_desc = (
            "trusting" if rel.trust > 0.6
            else "distrustful" if rel.trust < 0.3
            else "cautious"
        )
        
        # Add status tags if target is distressed or poor
        status_tags = []
        if target:
            if target.is_distressed:
                status_tags.append("DISTRESSED")
            if target.is_poor:
                status_tags.append("POOR")
        
        status_str = f" [{', '.join(status_tags)}]" if status_tags else ""
        
        lines.append(
            f"  {display_name}{status_str}: {affinity_desc} (affinity {rel.affinity:.2f}), "
            f"{trust_desc} (trust {rel.trust:.2f})"
        )
    return f"{citizen.name}'s relationships:\n" + "\n".join(lines)


def _describe_citizen_status(citizen: Citizen) -> str:
    """Describe a citizen's current status with tags."""
    status_tags = []
    if citizen.is_distressed:
        status_tags.append("DISTRESSED")
    if citizen.is_poor:
        status_tags.append("POOR")
    
    if status_tags:
        return f"{citizen.name}: [{', '.join(status_tags)}] (Gold: {citizen.gold}, Energy: {citizen.energy})"
    return f"{citizen.name}: (Gold: {citizen.gold}, Energy: {citizen.energy})"


def _get_recent_events(world: WorldState, num_events: int = 5) -> str:
    """Get the last N chronicle events for context."""
    if not world.history:
        return "No recent events."
    
    recent = world.history[-num_events:]
    events = []
    for entry in recent:
        events.append(f"Tick {entry['tick']}: {entry['chronicle']}")
    return "\n".join(events)


def _get_interaction_summary(citizen: Citizen) -> str:
    """Get a summary of recent interactions for memory."""
    if not citizen.recent_interactions:
        return "No recent interactions."
    
    interactions = []
    for interaction in citizen.recent_interactions[-5:]:  # Last 5 interactions
        action = interaction.get("action", "unknown")
        target = interaction.get("target", "unknown")
        tick = interaction.get("tick", 0)
        interactions.append(f"  Tick {tick}: {action} with {target}")
    
    return "\n".join(interactions)


def _get_zone_population(world: WorldState, zone: str) -> str:
    """Get the population of a zone."""
    citizens_in_zone = world.get_citizens_in_zone(zone)
    if not citizens_in_zone:
        return f"{zone} is empty."
    
    names = [c.name for c in citizens_in_zone]
    return f"{zone} contains: {', '.join(names)}"


def _behavioral_guidance(citizen: Citizen, world: WorldState, nearby_count: int, nearby_names: list[str]) -> str:
    """Provide personality-aligned behavioral suggestions.

    These guide the LLM's reasoning without hardcoding action probabilities.
    """
    parts = []

    # Zone-specific guidance
    zone_guidance = {
        "Residential_Sector": "Residential areas offer peaceful REST with excellent energy recovery (+2 bonus).",
        "Market_District": "The Market is where commerce happens. WORK here earns bonus gold (+3). TRADE opportunities abound.",
        "City_Park": "The Park's tranquility boosts social interactions (+0.05 affinity) and provides mild rest bonuses (+1)."
    }
    zone_tip = zone_guidance.get(citizen.current_zone, "")
    if zone_tip:
        parts.append(f"Zone Context: {zone_tip}")

    # Self-status guidance
    if citizen.is_distressed:
        parts.append(
            f"CRITICAL: {citizen.name} is DISTRESSED (energy={citizen.energy}). "
            "Survival is paramount. REST immediately to recover energy. "
            "If others are nearby, consider REQUEST_HELP. "
            f"Remember: REST recovers {REST_ENERGY_GAIN} energy plus zone bonuses."
        )
    elif citizen.energy <= 4:
        parts.append(
            f"Energy is running low ({citizen.energy}). "
            "Plan strategically: REST soon to recover, or find a safe zone. "
            "Avoid expensive actions until energy is restored."
        )
    
    if citizen.is_poor:
        parts.append(
            f"{citizen.name} is POOR (gold={citizen.gold}). "
            "WORK to earn money (costs 2 energy, earns 8+ gold). "
            "Market District offers best earnings. Ask for help if needed."
        )
    elif citizen.gold >= 30:
        parts.append(
            f"{citizen.name} has sufficient gold ({citizen.gold}). "
            "Consider HELP (8 gold + 2 energy) or GIFT (5 gold) to build relationships. "
            "Generosity creates allies."
        )

    # Proximity guidance — nearby citizens dramatically shift the action space
    if nearby_count > 0:
        nearby_str = ", ".join(nearby_names)
        parts.append(
            f"There are people nearby ({nearby_str}). "
            "This is an opportunity for meaningful interaction. "
            "Consider your goals before acting."
        )

        # Check for distressed/poor nearby citizens
        nearby_citizens = [world.get_citizen(name) for name in nearby_names]
        distressed_nearby = [c.name for c in nearby_citizens if c and c.is_distressed]
        poor_nearby = [c.name for c in nearby_citizens if c and c.is_poor]
        
        if distressed_nearby:
            parts.append(
                f"Nearby citizens in distress: {', '.join(distressed_nearby)}. "
                f"Empathetic citizens should consider HELP or GIFT. "
                f"REQUEST_HELP is also an option if you're in need."
            )
        
        if poor_nearby:
            parts.append(
                f"Nearby citizens with little gold: {', '.join(poor_nearby)}. "
                f"They may appreciate GIFT, HELP, or TRADE opportunities."
            )

        # Personality-informed preference hints (NOT hardcoded probabilities)
        if citizen.personality.empathy >= 0.7:
            parts.append(
                f"{citizen.name}'s strong empathy makes GIFT, HELP, and supportive CONVERSE feel natural. "
                f"STEAL would violate personal values."
            )
        
        # Strong HELP/GIFT bias for highly empathetic citizens with resources
        if (
            citizen.personality.empathy > 0.8
            and citizen.personality.aggression < 0.4
            and citizen.gold >= 15
        ):
            nearby_citizens = [world.get_citizen(name) for name in nearby_names]
            struggling_targets = [
                c.name for c in nearby_citizens
                if c and (c.is_distressed or c.is_poor)
            ]
            if struggling_targets:
                targets_str = ", ".join(struggling_targets)
                parts.append(
                    f"With means and compassion, {citizen.name} could assist {targets_str}. "
                    f"HELP (8g + 2e) is most impactful. GIFT (5g) is also meaningful."
                )
        
        # Work ethic for certain personalities
        if citizen.personality.aggression < 0.3 and citizen.personality.sociability < 0.4:
            parts.append(
                f"{citizen.name}'s introspective nature makes WORK productive. "
                f"Earn gold through labor, especially in Market District."
            )
        
        if citizen.personality.aggression >= 0.7 and citizen.personality.empathy < 0.5:
            parts.append(
                f"{citizen.name}'s aggressive nature may favor STEAL or WARN. "
                f"STEAL has 5-tick cooldown and destroys trust. Choose wisely."
            )
        
        if citizen.personality.sociability >= 0.7:
            parts.append(
                f"Given {citizen.name}'s sociable nature, CONVERSE is natural. "
                f"Consider WARN to build trust, or TRADE for economic cooperation."
            )
    else:
        parts.append(
            f"{citizen.name} is alone. Strategic options: "
            f"MOVE to find company, WORK to earn gold (Market District best), "
            f"REST to recover energy (Residential best), EXPLORE for opportunities, "
            f"or OBSERVE to gather information."
        )

    # Relationship-based guidance
    for target_name, rel in citizen.relationships.items():
        if rel.affinity > 0.5 and nearby_count > 0 and target_name in nearby_names:
            parts.append(
                f"Strong positive bond with {target_name} (affinity {rel.affinity:.2f}). "
                f"CONVERSE, GIFT, or HELP would deepen this connection."
            )
        if rel.affinity < -0.3 and rel.trust < 0.2 and nearby_count > 0 and target_name in nearby_names:
            parts.append(
                f"Trust with {target_name} is near zero (trust {rel.trust:.2f}). "
                f"Consider IGNORE to avoid conflict, or WARN to establish boundaries."
            )

    # Cooldown awareness
    if "steal" in citizen.cooldowns and citizen.cooldowns["steal"] > 0:
        parts.append(
            f"STEAL on cooldown ({citizen.cooldowns['steal']} ticks remaining)."
        )
    if "work" in citizen.cooldowns and citizen.cooldowns["work"] > 0:
        parts.append(
            f"WORK on cooldown ({citizen.cooldowns['work']} ticks remaining). Consider REST or social actions."
        )
    if "rest" in citizen.cooldowns and citizen.cooldowns["rest"] > 0:
        parts.append(
            f"REST on cooldown ({citizen.cooldowns['rest']} ticks remaining). Choose other actions."
        )

    # Memory-based guidance
    if citizen.recent_interactions:
        recent_thefts = [i for i in citizen.recent_interactions if i["action"] == "STEAL" and i["target"] == citizen.name]
        recent_help = [i for i in citizen.recent_interactions if i["action"] == "HELP"]
        
        if recent_thefts:
            parts.append(
                f"WARNING: {len(recent_thefts)} recent theft(s) victimized you. "
                f"Consider STEAL retaliation, WARN others, or avoid the perpetrator."
            )
        
        if recent_help:
            parts.append(
                f"You've received help recently ({len(recent_help)} time(s)). "
                f"Consider GIFT or HELP in return to maintain goodwill."
            )

    return "\n".join(parts)


def build_prompt(citizen: Citizen, world: WorldState) -> str:
    """
    Build a complete prompt for the LLM based on current simulation state.

    Includes the full action schema with examples, valid citizen names,
    personality-driven behavioral guidance, and strict output formatting.

    Args:
        citizen: The active citizen who will decide an action.
        world: The current world state.

    Returns:
        A formatted prompt string.
    """
    nearby = world.get_citizens_in_zone(citizen.current_zone, exclude_name=citizen.name)
    nearby_names = [c.name for c in nearby]
    nearby_str = ", ".join(nearby_names) if nearby_names else "no one"
    valid_citizen_names = ", ".join(sorted(c.name for c in world.citizens.values()))
    zones_str = ", ".join(settings.zones)

    behavioral = _behavioral_guidance(citizen, world, len(nearby_names), nearby_names)
    recent_events = _get_recent_events(world, num_events=5)
    interaction_summary = _get_interaction_summary(citizen)
    zone_population = _get_zone_population(world, citizen.current_zone)

    prompt = f"""You are controlling {citizen.name} in a post-apocalyptic city simulation. Your decisions determine survival and social standing. Think carefully about your current situation, goals, and relationships before acting.

=== Current State ===
Tick: {world.tick}
Zone: {citizen.current_zone}
Zone Population: {zone_population}
Nearby: {nearby_str}

=== Citizen Status ===
{_describe_citizen_status(citizen)}

=== Nearby Citizens ===
{chr(10).join(_describe_citizen_status(world.get_citizen(name)) for name in nearby_names) if nearby_names else "No one nearby."}

=== Personality ===
{_describe_personality(citizen)}

=== Relationships ===
{_describe_relationships(citizen, world)}

=== Recent Interactions ===
{interaction_summary}

=== Recent Events (Chronicle) ===
{recent_events}

=== Decision Guidance ===
{behavioral}

=== Available Actions ===
- MOVE: Travel to a different zone.
  Available zones: {zones_str}
  Use target_zone to specify destination. target_citizen should be null.

- EXPLORE: Carefully explore a zone (costs 2 energy, reveals opportunities).
  Use target_zone to specify destination.

- CONVERSE: Talk to someone nearby (or to yourself if alone).
  Use target_citizen to specify who to talk to, or null for solo speech.

- GIFT: Give 5 gold to someone nearby.
  Requires target_citizen and at least 5 gold. Both must be in same zone.

- HELP: Give 8 gold and 2 energy to help someone (stronger than GIFT).
  Requires target_citizen, 8 gold, and 2 energy. Both must be in same zone.

- REQUEST_HELP: Ask someone for help (shows vulnerability).
  Requires target_citizen in same zone.

- WARN: Warn someone about danger (builds trust).
  Requires target_citizen in same zone.

- TRADE: Trade 10 gold with someone (mutual economic benefit).
  Requires target_citizen and 10 gold. Both must be in same zone.

- FOLLOW: Follow someone to their zone.
  Requires target_citizen in same zone.

- IGNORE: Intentionally ignore someone (damages relationship).
  Requires target_citizen in same zone.

- WORK: Earn 8 gold (costs 3 energy, 3-tick cooldown).
  No target needed.

- REST: Recover 3 energy (2-tick cooldown).
  No target needed.

- OBSERVE: Observe your surroundings (informational).
  No target needed.

- STEAL: Take 5 gold from someone nearby.
  Requires target_citizen in same zone. 5-tick cooldown. Damages trust.

=== Valid Citizens (use EXACTLY these names) ===
{valid_citizen_names}

=== JSON Schema ===
Return ONLY valid JSON. No markdown. No comments. No explanations. No surrounding text.

Example MOVE: {{"action":"MOVE","target_citizen":null,"target_zone":"Market_District","dialogue":"","intended_state_delta":{{"description":"Moving"}}}}

Example CONVERSE: {{"action":"CONVERSE","target_citizen":"Alice","target_zone":null,"dialogue":"Hello there","intended_state_delta":{{}}}}

Example GIFT: {{"action":"GIFT","target_citizen":"Charlie","target_zone":null,"dialogue":"Take this","intended_state_delta":{{"description":"Gift"}}}}

Example STEAL: {{"action":"STEAL","target_citizen":"Bob","target_zone":null,"dialogue":"","intended_state_delta":{{"description":"Theft"}}}}"""
    return prompt
