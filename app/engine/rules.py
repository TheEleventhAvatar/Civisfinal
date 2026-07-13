"""
Game rules and constants for the simulation.

All tunable gameplay parameters are defined here to keep them
centralized and easy to modify.
"""

# Energy
ENERGY_DECAY_PER_TICK: int = 1
INITIAL_ENERGY: int = 10
REST_ENERGY_GAIN: int = 4
WORK_ENERGY_COST: int = 2
EXPLORE_ENERGY_COST: int = 1

# Zone effects
ZONE_REST_BONUS: dict[str, int] = {
    "Residential_Sector": 2,  # +2 energy when resting
    "Market_District": 0,
    "City_Park": 1,  # +1 energy when resting
}

ZONE_WORK_BONUS: dict[str, int] = {
    "Residential_Sector": 0,
    "Market_District": 3,  # +3 gold when working
    "City_Park": 0,
}

ZONE_SOCIAL_BONUS: dict[str, float] = {
    "Residential_Sector": 0.0,
    "Market_District": 0.0,
    "City_Park": 0.05,  # +0.05 affinity when conversing
}

# Gold
INITIAL_GOLD: int = 50
GIFT_GOLD_AMOUNT: int = 5
STEAL_GOLD_AMOUNT: int = 5
HELP_GOLD_AMOUNT: int = 8
HELP_ENERGY_AMOUNT: int = 2
WORK_GOLD_EARNED: int = 8
TRADE_GOLD_AMOUNT: int = 10

# Relationship modifiers
REL_AFFINITY_GIFT: float = 0.15
REL_TRUST_GIFT: float = 0.10

REL_AFFINITY_STEAL: float = -0.30
REL_TRUST_STEAL: float = -0.40

REL_AFFINITY_CONVERSE: float = 0.05
REL_TRUST_CONVERSE: float = 0.0

REL_AFFINITY_HELP: float = 0.25
REL_TRUST_HELP: float = 0.20

REL_AFFINITY_WARN: float = 0.10
REL_TRUST_WARN: float = 0.05

REL_AFFINITY_TRADE: float = 0.20
REL_TRUST_TRADE: float = 0.15

REL_AFFINITY_REQUEST_HELP: float = -0.05
REL_TRUST_REQUEST_HELP: float = 0.0

REL_AFFINITY_IGNORE: float = -0.10
REL_TRUST_IGNORE: float = -0.05

REL_AFFINITY_MOVE: float = 0.0
REL_TRUST_MOVE: float = 0.0

# Relationship bounds
AFFINITY_MIN: float = -1.0
AFFINITY_MAX: float = 1.0
TRUST_MIN: float = 0.0
TRUST_MAX: float = 1.0

# Action constraints
MIN_ENERGY_FOR_WORK: int = 3
MIN_ENERGY_FOR_EXPLORE: int = 2
MIN_ENERGY_FOR_STEAL: int = 2
MIN_GOLD_FOR_GIFT: int = 5
MIN_GOLD_FOR_HELP: int = 8
MIN_GOLD_FOR_TRADE: int = 10

# Cooldowns (in ticks)
COOLDOWN_REST: int = 2
COOLDOWN_WORK: int = 3
COOLDOWN_STEAL: int = 5
