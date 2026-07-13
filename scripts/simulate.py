#!/usr/bin/env python3
"""
Entry point for running the Project Civis simulation.

Usage:
    python scripts/simulate.py [--ticks N] [--mock]

Options:
    --ticks N   Number of simulation ticks to run (default: 15)
    --mock      Use the MockProvider instead of a real LLM
"""

import argparse
import sys
import os
import json

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine.simulator import Simulator, MockProvider
from app.models.actions import LLMAction, ActionType


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Project Civis - AI-native city social simulator",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="Number of simulation ticks (default: from .env or 15)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockProvider instead of a real LLM",
    )
    return parser.parse_args()


def main() -> None:
    """Run the simulation."""
    args = parse_args()

    if args.mock:
        # Use a mock provider that cycles through different actions
        actions = [
            LLMAction(action=ActionType.CONVERSE, dialogue="Hello there!", target_citizen="Bob"),
            LLMAction(action=ActionType.MOVE, target_citizen="Market_District"),
            LLMAction(action=ActionType.GIFT, dialogue="Take this gold.", target_citizen="Alice"),
            LLMAction(action=ActionType.STEAL, target_citizen="Charlie"),
            LLMAction(action=ActionType.CONVERSE, dialogue="Nice weather today."),
        ]
        import itertools
        action_cycle = itertools.cycle(actions)
        provider = MockProvider(action=next(action_cycle))
        # Override generate to cycle through actions
        class CyclingMockProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self._cycle = action_cycle

            def generate(self, prompt: str) -> str:
                action = next(self._cycle)
                return action.model_dump_json()

        provider = CyclingMockProvider()
        injection_scenarios = []
    else:
        provider = None  # Will be created from config
        # Inject invalid actions at specific ticks to demonstrate validator rejection
        injection_scenarios = [
            {
                "tick": 5,
                "description": "Invalid zone injection",
                "raw_response": '{"action":"MOVE","target_zone":"Atlantis","dialogue":""}',
            },
            {
                "tick": 10,
                "description": "Insufficient gold injection",
                "raw_response": '{"action":"GIFT","target_citizen":"Charlie","dialogue":"Here is 1000 gold.","intended_state_delta":{}}',
                "set_actor_gold": 3,
            },
            {
                "tick": 14,
                "description": "Unsupported action injection",
                "raw_response": '{"action":"TELEPORT","target_zone":"Market_District","dialogue":""}',
            },
        ]

    sim = Simulator(llm_provider=provider, injection_scenarios=injection_scenarios)
    results = sim.run(max_ticks=args.ticks)

    # Print final summary
    print("\n" + "=" * 60, file=sys.stderr)
    print("FINAL STATE SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    for citizen in sim.world.citizen_list:
        print(f"\n{citizen.name}:", file=sys.stderr)
        print(f"  Zone: {citizen.current_zone}", file=sys.stderr)
        print(f"  Gold: {citizen.gold}", file=sys.stderr)
        print(f"  Energy: {citizen.energy}", file=sys.stderr)
        print(f"  Personality: soc={citizen.personality.sociability:.1f} "
              f"agg={citizen.personality.aggression:.1f} "
              f"emp={citizen.personality.empathy:.1f}", file=sys.stderr)
        if citizen.relationships:
            print("  Relationships:", file=sys.stderr)
            for rel in citizen.relationships.values():
                target = sim.world.get_citizen(rel.target_id)
                target_name = target.name if target else rel.target_id
                print(f"    {target_name}: affinity={rel.affinity:.2f}, "
                      f"trust={rel.trust:.2f}", file=sys.stderr)

    # Count successful gifts
    successful_gifts = sum(
        1 for r in results
        if r["action"].get("action") == "GIFT" and r["accepted"]
    )

    print(f"\nTotal ticks executed: {len(results)}", file=sys.stderr)
    print(f"Successful Gifts: {successful_gifts}", file=sys.stderr)
    print(f"Log written to: {sim.logger.log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()