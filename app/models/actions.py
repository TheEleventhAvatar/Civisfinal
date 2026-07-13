"""
Action models for the simulation.

Defines the structured action schema that the LLM must respond with,
and the validated action types that the engine processes.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """All supported citizen actions in the simulation."""
    # Movement
    MOVE = "MOVE"
    EXPLORE = "EXPLORE"
    FOLLOW = "FOLLOW"
    
    # Social
    CONVERSE = "CONVERSE"
    GIFT = "GIFT"
    HELP = "HELP"
    REQUEST_HELP = "REQUEST_HELP"
    WARN = "WARN"
    TRADE = "TRADE"
    IGNORE = "IGNORE"
    
    # Economic
    WORK = "WORK"
    STEAL = "STEAL"
    
    # Survival
    REST = "REST"
    OBSERVE = "OBSERVE"


class LLMAction(BaseModel):
    """
    The raw action proposed by the LLM.

    The `intended_state_delta` is informational only and MUST NOT
    be used to mutate state. It exists to satisfy the LLM's desire
    to express intent, but the validator and mutation engine derive
    all actual state changes from the validated action type.

    Action-specific parameters:
    - MOVE/EXPLORE: target_zone (target_citizen ignored)
    - CONVERSE/GIFT/HELP/REQUEST_HELP/WARN/TRADE/IGNORE/FOLLOW: target_citizen (target_zone ignored)
    - WORK/REST/OBSERVE: no target needed
    - STEAL: target_citizen (target_zone ignored)
    """
    action: ActionType
    target_citizen: str | None = Field(
        default=None,
        description="Target citizen for social/economic actions. Null for movement/rest/work.",
    )
    target_zone: str | None = Field(
        default=None,
        description="Destination zone for MOVE/EXPLORE. Ignored for other actions.",
    )
    dialogue: str = Field(
        default="",
        description="Dialogue text for social actions (CONVERSE, GIFT, HELP, WARN, REQUEST_HELP, TRADE)",
    )
    intended_state_delta: dict[str, Any] = Field(
        default_factory=dict,
        description="LLM's intended state changes. IGNORED during execution.",
    )


class ParserError(BaseModel):
    """Represents a failure to parse the LLM response."""
    raw_text: str
    reason: str


class ValidationResult(BaseModel):
    """Result of validating an LLM-proposed action."""
    accepted: bool
    reason: str
    error_code: str = Field(
        default="OK",
        description="Machine-readable error code for the validation failure",
    )
    safe_action: LLMAction
