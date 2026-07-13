"""
Citizen model representing an individual in the simulation.

Each citizen has a personality tensor (Sociability, Aggression, Empathy),
resources (gold, energy), and a map of relationships to other citizens.
"""

from pydantic import BaseModel, Field

from app.models.relationship import Relationship


class PersonalityTensor(BaseModel):
    """Three-dimensional personality trait vector.

    Values are floats in [0.0, 1.0] and influence LLM prompt construction.
    They do NOT directly determine action probabilities.
    """
    sociability: float = Field(default=0.5, ge=0.0, le=1.0)
    aggression: float = Field(default=0.5, ge=0.0, le=1.0)
    empathy: float = Field(default=0.5, ge=0.0, le=1.0)


class Citizen(BaseModel):
    """A single citizen in the simulation."""

    id: str
    name: str
    current_zone: str
    gold: int = Field(default=50, ge=0)
    energy: int = Field(default=10, ge=0)
    personality: PersonalityTensor = Field(default_factory=PersonalityTensor)
    relationships: dict[str, Relationship] = Field(default_factory=dict)
    
    # Cooldowns: action -> ticks remaining
    cooldowns: dict[str, int] = Field(default_factory=dict)
    
    # Interaction history for memory
    recent_interactions: list[dict] = Field(default_factory=list, max_length=10)

    @property
    def is_desperate(self) -> bool:
        """A citizen is desperate when their energy reaches zero."""
        return self.energy <= 0

    @property
    def is_distressed(self) -> bool:
        """A citizen is distressed when their energy is critically low."""
        return self.energy <= 2

    @property
    def is_poor(self) -> bool:
        """A citizen is poor when they have less than 15 gold."""
        return self.gold < 15

    def get_relationship(self, target_name: str) -> Relationship:
        """Get the relationship to another citizen by name, creating a default if absent."""
        if target_name not in self.relationships:
            self.relationships[target_name] = Relationship(target_id=target_name)
        return self.relationships[target_name]
