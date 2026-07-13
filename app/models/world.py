"""
WorldState model representing the complete simulation state.

This is the single source of truth for the simulation. The LLM
never touches this directly; all mutations go through the validator
and mutation engine.
"""

from pydantic import BaseModel, Field

from app.models.citizen import Citizen


class WorldState(BaseModel):
    """
    Immutable-like container for the entire simulation state.

    While the fields are mutable Python objects, the architecture
    ensures that only the MutationEngine may modify WorldState
    instances, and only after validation.

    Citizens are keyed by their canonical name (Alice, Bob, Charlie).
    """

    tick: int = Field(default=0, ge=0)
    citizens: dict[str, Citizen] = Field(default_factory=dict)
    history: list[dict] = Field(default_factory=list)

    @property
    def citizen_list(self) -> list[Citizen]:
        """Return citizens sorted by name for deterministic iteration."""
        return sorted(self.citizens.values(), key=lambda c: c.name)

    def get_citizen(self, name: str) -> Citizen | None:
        """Look up a citizen by canonical name."""
        return self.citizens.get(name)

    def get_citizens_in_zone(self, zone: str, exclude_name: str | None = None) -> list[Citizen]:
        """Return all citizens in a given zone, optionally excluding one by name."""
        return [
            c for c in self.citizens.values()
            if c.current_zone == zone and c.name != exclude_name
        ]

    def model_dump_simple(self) -> dict:
        """Serialize to a JSON-safe dict for logging."""
        return self.model_dump(mode="json")