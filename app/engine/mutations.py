"""
Immutable mutation objects representing validated state changes.

Each mutation is a dataclass-like record of a single atomic state
change. The simulator applies these to the WorldState after validation.

This separation ensures that validation and state mutation are
completely decoupled.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Mutation:
    """Base mutation type - all mutations share a description."""
    description: str = ""


@dataclass(frozen=True)
class GoldMutation(Mutation):
    """Change a citizen's gold by a delta (positive or negative)."""
    citizen_name: str = ""
    delta: int = 0


@dataclass(frozen=True)
class EnergyMutation(Mutation):
    """Change a citizen's energy by a delta."""
    citizen_name: str = ""
    delta: int = 0


@dataclass(frozen=True)
class MoveMutation(Mutation):
    """Move a citizen to a new zone."""
    citizen_name: str = ""
    destination: str = ""


@dataclass(frozen=True)
class RelationshipMutation(Mutation):
    """Modify a citizen's relationship with another citizen."""
    citizen_name: str = ""
    target_name: str = ""
    affinity_delta: float = 0.0
    trust_delta: float = 0.0


@dataclass(frozen=True)
class DialogueMutation(Mutation):
    """Record dialogue spoken by a citizen."""
    citizen_name: str = ""
    dialogue: str = ""
    target_name: str = ""


@dataclass(frozen=True)
class RestMutation(Mutation):
    """Rest to recover energy."""
    citizen_name: str = ""
    energy_gained: int = 0


@dataclass(frozen=True)
class WorkMutation(Mutation):
    """Work to earn gold (costs energy)."""
    citizen_name: str = ""
    gold_earned: int = 0
    energy_cost: int = 0


@dataclass(frozen=True)
class ObserveMutation(Mutation):
    """Observe surroundings to gain information."""
    citizen_name: str = ""
    zone_observed: str = ""


@dataclass(frozen=True)
class ExploreMutation(Mutation):
    """Explore a new zone."""
    citizen_name: str = ""
    destination: str = ""
    energy_cost: int = 0


@dataclass(frozen=True)
class HelpMutation(Mutation):
    """Help another citizen (more effective than gift)."""
    citizen_name: str = ""
    target_name: str = ""
    gold_given: int = 0
    energy_given: int = 0


@dataclass(frozen=True)
class TradeMutation(Mutation):
    """Trade gold with another citizen."""
    citizen_name: str = ""
    target_name: str = ""
    gold_sent: int = 0
    gold_received: int = 0


@dataclass(frozen=True)
class RequestHelpMutation(Mutation):
    """Record a request for help."""
    citizen_name: str = ""
    target_name: str = ""
    dialogue: str = ""


@dataclass(frozen=True)
class WarnMutation(Mutation):
    """Warn another citizen about danger."""
    citizen_name: str = ""
    target_name: str = ""
    dialogue: str = ""


@dataclass(frozen=True)
class FollowMutation(Mutation):
    """Follow another citizen to their zone."""
    citizen_name: str = ""
    target_name: str = ""


@dataclass(frozen=True)
class IgnoreMutation(Mutation):
    """Intentionally ignore someone."""
    citizen_name: str = ""
    target_name: str = ""


def mutations_to_dicts(mutations: list[Mutation]) -> list[dict[str, Any]]:
    """Convert mutation objects to serializable dicts for logging."""
    return [{
        "type": type(m).__name__,
        "description": m.description,
        **{k: v for k, v in m.__dict__.items() if k != "description"},
    } for m in mutations]