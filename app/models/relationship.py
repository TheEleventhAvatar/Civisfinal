"""
Relationship model representing the dyadic bond between two citizens.

Each relationship tracks affinity (like/dislike) and trust,
both clamped to their respective ranges.
"""

from pydantic import BaseModel, Field


class Relationship(BaseModel):
    """
    A directed relationship from one citizen to another.

    Affinity ranges from -1.0 (hostile) to 1.0 (friendly).
    Trust ranges from 0.0 (none) to 1.0 (absolute).
    """

    target_id: str
    affinity: float = Field(default=0.0, ge=-1.0, le=1.0)
    trust: float = Field(default=0.3, ge=0.0, le=1.0)

    def clamp(self) -> None:
        """Ensure affinity and trust stay within valid bounds."""
        self.affinity = max(-1.0, min(1.0, self.affinity))
        self.trust = max(0.0, min(1.0, self.trust))