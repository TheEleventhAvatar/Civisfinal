# Project Civis

An AI-native city social simulator demonstrating exceptional software architecture around LLM agents, deterministic simulation, and state isolation.

## Architecture

```
CLI (scripts/simulate.py)
    │
    ▼
LLM Provider (app/llm/)
    │  Builds prompt with personality context
    ▼
Structured JSON Action (app/models/actions.py)
    │
    ▼
Parser (app/llm/parser.py)
    │  Resilient JSON extraction and repair
    ▼
Validator (app/engine/validator.py)
    │  The ONLY trusted authority for state changes
    ▼
Immutable Mutations (app/engine/mutations.py)
    │  GoldMutation, EnergyMutation, MoveMutation, RelationshipMutation
    ▼
Mutation Engine (app/engine/simulator.py)
    │  Applies mutations to WorldState
    ▼
State Store (app/models/world.py)
    │  Pydantic-validated WorldState
    ▼
Chronicle Engine (app/engine/chronicle.py)
    │  Deterministic prose (no LLM used)
```

## State Isolation

The most important architectural constraint: **The LLM NEVER modifies state directly.**

The flow is strictly:
1. The LLM **proposes** actions as structured JSON
2. The Parser extracts and validates the JSON schema
3. The Validator checks every action against game rules (resources, zones, targets)
4. Invalid actions are replaced with a safe fallback (CONVERSE or REST)
5. The Mutation Engine produces immutable `Mutation` objects
6. The Simulator applies mutations to the `WorldState`

The `intended_state_delta` field in the LLM response is **COMPLETELY IGNORED**. The simulation derives all state changes from the validated action type using deterministic rules, not from what the LLM says it wants to happen.

The simulation engine behaves exactly the same regardless of whether actions come from a real LLM or the `MockProvider`.

## Supported Actions

- **MOVE**: Travel to another zone
- **EXPLORE**: Explore a zone (costs 1 energy)
- **FOLLOW**: Follow a citizen to their zone
- **CONVERSE**: Talk to a nearby citizen
- **GIFT**: Give 5 gold to a nearby citizen
- **HELP**: Give 8 gold and 2 energy to help someone
- **REQUEST_HELP**: Ask someone for help
- **WARN**: Warn someone about danger
- **TRADE**: Trade 10 gold with someone
- **IGNORE**: Intentionally ignore someone
- **WORK**: Earn 8+ gold (costs 2 energy, 3-tick cooldown)
- **REST**: Recover 4+ energy (2-tick cooldown)
- **OBSERVE**: Observe surroundings (informational)
- **STEAL**: Take 5 gold from someone (5-tick cooldown)

### Zone Effects

Zones provide strategic bonuses:

| Zone | REST Bonus | WORK Bonus | Social Bonus |
|------|-----------|-----------|--------------|
| Residential_Sector | +2 energy | — | — |
| Market_District | — | +3 gold | — |
| City_Park | +1 energy | — | +0.05 affinity |

### Relationship Effects

| Action | Affinity Δ | Trust Δ |
|--------|-----------|---------|
| GIFT | +0.15 | +0.10 |
| HELP | +0.25 | +0.20 |
| STEAL | -0.30 | -0.40 |
| CONVERSE | +0.05 | 0.0 |
| WARN | +0.10 | +0.05 |
| TRADE | +0.20 | +0.15 |
| REQUEST_HELP | -0.05 | 0.0 |
| IGNORE | -0.10 | -0.05 |

Affinity is clamped to [-1, 1], Trust to [0, 1].

## Energy Economy

Energy decay applies only to the active citizen each tick, preventing the death spiral where all citizens become incapacitated simultaneously. REST is always allowed, even at 0 energy, as the recovery mechanic.

## Prompt Strategy

Personality traits (Sociability, Aggression, Empathy) are converted to descriptive text in prompts, which influences the LLM's behavior without hardcoding action probabilities:

- **High Sociability (≥0.7)**: "highly sociable and enjoys interacting with others"
- **Low Sociability (≤0.3)**: "reserved and prefers solitude"
- **High Aggression (≥0.7)**: "quick to aggression and willing to take what they want"
- **Low Empathy (≤0.3)**: "lacks empathy and is indifferent to others' suffering"
- **High Empathy (≥0.7)**: "deeply empathetic and considerate of others' feelings"

Distressed citizens (energy ≤ 2) and poor citizens (gold < 15) are flagged in the prompt to encourage helping behavior.

## How Validation Works

The `Validator` class checks every proposed action:

| Action | Checks |
|--------|--------|
| MOVE | Destination zone exists (with name normalization) |
| EXPLORE | Enough energy, valid destination |
| FOLLOW | Target exists, same zone |
| CONVERSE | Always accepted (safe default) |
| GIFT | Target exists, same zone, actor has ≥5 gold |
| HELP | Target exists, same zone, actor has ≥8 gold and ≥2 energy |
| REQUEST_HELP | Target exists, same zone |
| WARN | Target exists, same zone |
| TRADE | Target exists, same zone, actor has ≥10 gold |
| IGNORE | Target exists, same zone |
| WORK | Actor has ≥3 energy, not on cooldown |
| REST | Not on cooldown (always allowed, even at 0 energy) |
| OBSERVE | Always accepted |
| STEAL | Target exists, same zone, actor has ≥2 energy, not on cooldown |

If validation fails, the action is replaced with a safe fallback and a `ValidationResult` is returned with:
- `accepted`: false
- `reason`: human-readable explanation
- `error_code`: machine-readable error code (e.g., `GIFT_NO_TARGET`, `MOVE_INVALID_ZONE`, `INCAPACITATED`)

## Error Codes

Every `ValidationResult` includes a machine-readable `error_code`:

| Error Code | Meaning |
|------------|---------|
| `OK` | Action accepted |
| `INCAPACITATED` | Actor has 0 energy |
| `UNKNOWN_ACTION` | Action type not recognized |
| `MOVE_NO_DESTINATION` | MOVE missing target_zone |
| `MOVE_INVALID_ZONE` | Destination zone doesn't exist |
| `GIFT_NO_TARGET` | GIFT missing target_citizen |
| `GIFT_SELF` | Cannot gift to yourself |
| `GIFT_INSUFFICIENT_GOLD` | Not enough gold to gift |
| `GIFT_DIFFERENT_ZONE` | Target not in same zone |
| `STEAL_SELF` | Cannot steal from yourself |
| `STEAL_INSUFFICIENT_ENERGY` | Not enough energy to steal |
| `STEAL_COOLDOWN` | Steal is on cooldown |
| `WORK_INSUFFICIENT_ENERGY` | Not enough energy to work |
| `WORK_COOLDOWN` | Work is on cooldown |
| `REST_COOLDOWN` | Rest is on cooldown |
| `HELP_NO_TARGET` | HELP missing target |
| `HELP_SELF` | Cannot help yourself |
| `HELP_INSUFFICIENT_GOLD` | Not enough gold to help |
| `HELP_INSUFFICIENT_ENERGY` | Not enough energy to help |
| `TRADE_NO_TARGET` | TRADE missing target |
| `TRADE_SELF` | Cannot trade with yourself |
| `TRADE_INSUFFICIENT_GOLD` | Not enough gold to trade |
| `CONVERSE_BAD_TARGET` | Target citizen not found |
| `PARSE_ERROR` | Failed to parse LLM response |

## Comprehensive Validation Test Suite

The project includes a comprehensive validation test suite (`tests/test_validation_full.py`) covering:

- **Category A**: Schema validation (malformed JSON, missing fields, invalid enums)
- **Category B**: Entity validation (nonexistent citizens, invalid zones)
- **Category C**: World rules (cross-zone interactions)
- **Category D**: Resource rules (insufficient gold/energy)
- **Category E**: Simulation integrity (multiple actions, nested structures)
- **Category F**: Impossible state mutations (self-targeting, incapacitated actions)
- **Category G**: Adversarial LLM outputs (prose, code fences, unicode, partial JSON)
- **Category H**: Property-based random tests (random names, zones, edge cases)
- **Category I**: Regression tests (one canonical passing test per valid action)
- **Category J**: Error code coverage (every rejection includes a meaningful error_code)

## Tech Stack

- Python 3.11+
- Pydantic v2
- python-dotenv
- requests (for Ollama)
- google-generativeai (optional Gemini provider)
- openai (optional OpenAI provider)
- pytest

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure LLM Provider

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

**For OpenAI (default):**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-mini
```

**For Ollama:**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

**For Gemini:**
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-api-key-here
GEMINI_MODEL=gemini-2.0-flash
```

### 3. Run the simulation

```bash
python scripts/simulate.py
```

Or with a mock provider (no LLM needed):

```bash
python scripts/simulate.py --mock
```

Specify the number of ticks:

```bash
python scripts/simulate.py --ticks 10
```

### 4. Run tests

```bash
pytest
```

Or with verbose output:

```bash
pytest -v
```

## Project Structure

```
project/
│
├── app/
│   ├── models/           # Pydantic data models
│   │   ├── actions.py    # LLMAction, ValidationResult, ActionType, ParserError
│   │   ├── citizen.py    # Citizen, PersonalityTensor
│   │   ├── relationship.py
│   │   └── world.py      # WorldState
│   │
│   ├── llm/              # LLM integration layer
│   │   ├── provider.py   # Abstract LLMProvider
│   │   ├── ollama_provider.py
│   │   ├── gemini_provider.py
│   │   ├── openai_provider.py
│   │   ├── prompt_builder.py
│   │   └── parser.py     # Resilient JSON parser
│   │
│   ├── engine/           # Simulation engine
│   │   ├── simulator.py  # Main orchestrator + MockProvider
│   │   ├── validator.py  # Action validation with error codes
│   │   ├── mutations.py  # Immutable mutation objects
│   │   ├── chronicle.py  # Deterministic prose
│   │   └── rules.py      # Game constants
│   │
│   ├── config.py         # Environment configuration
│   └── logger.py         # JSONL + console logging
│
├── logs/
│   └── run.jsonl         # Simulation log output
│
├── scripts/
│   ├── simulate.py       # CLI entry point with injection scenarios
│   ├── replay.py         # Deterministic replay system
│   └── benchmark.py      # Benchmark mode with seed management
│
├── tests/
│   ├── test_validator.py # Validation tests
│   ├── test_parser.py    # Parser tests
│   ├── test_benchmark.py # Benchmark/integration tests
│   └── test_validation_full.py # Comprehensive validation suite (Categories A-J)
│
├── .env.example
├── requirements.txt
└── README.md
```

## Switching Between LLM Providers

Edit `.env` and change `LLM_PROVIDER`:

- `LLM_PROVIDER=openai` — Uses OpenAI API (requires API key)
- `LLM_PROVIDER=ollama` — Uses local Ollama instance
- `LLM_PROVIDER=gemini` — Uses Google's Gemini API (requires API key)

The `Simulator._create_provider()` method handles provider selection automatically.

## Production Design Principles

- **Dependency Injection**: LLM providers and loggers are injectable
- **Abstract Base Classes**: `LLMProvider` defines a clean interface
- **Immutable Mutations**: All state changes are recorded as frozen dataclass objects
- **Separation of Concerns**: Validation, mutation derivation, and state application are in separate modules
- **No Global State**: All state is contained in `WorldState` instances
- **Type Safety**: Pydantic models everywhere, no raw dicts for state
- **Resilient Parsing**: Multi-strategy JSON extraction with repair
- **Deterministic Chronicle**: No LLM used for narrative generation
- **Error Code Tracking**: Every rejection includes a machine-readable error code for debugging
- **Injection Testing**: Built-in support for injecting invalid actions to demonstrate validator behavior