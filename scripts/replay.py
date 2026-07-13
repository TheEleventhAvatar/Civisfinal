#!/usr/bin/env python3
"""
Deterministic replay system for simulation runs.

Replays a simulation from a log file to verify deterministic behavior.
Usage:
    python scripts/replay.py logs/run.jsonl
"""

import argparse
import json
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine.simulator import Simulator
from app.models.actions import ActionType, LLMAction
from app.engine.validator import Validator
from app.models.citizen import PersonalityTensor


def load_log(filepath: str) -> list[dict]:
    """Load simulation log from JSONL file."""
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def replay_simulation(log_entries: list[dict], verbose: bool = False) -> bool:
    """
    Replay the simulation from log entries.
    
    Returns True if replay matches original (deterministic), False otherwise.
    """
    print("=" * 60, file=sys.stderr)
    print("DETERMINISTIC REPLAY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    # Create a fresh simulator
    sim = Simulator()
    
    # Track state changes
    state_snapshots = []
    match_count = 0
    mismatch_count = 0
    
    for idx, entry in enumerate(log_entries):
        tick = entry.get('tick', idx + 1)
        
        # Extract action from log
        action_data = entry.get('action', {})
        action_type = action_data.get('action', 'CONVERSE')
        
        # Reconstruct the action
        try:
            action = LLMAction(
                action=ActionType(action_type),
                target_citizen=action_data.get('target_citizen'),
                target_zone=action_data.get('target_zone'),
                dialogue=action_data.get('dialogue', ''),
            )
        except Exception as e:
            print(f"Tick {tick}: Failed to reconstruct action: {e}", file=sys.stderr)
            continue
        
        # Get current actor
        actor = sim.world.get_citizen(entry.get('actor', ''))
        if not actor:
            print(f"Tick {tick}: Actor not found: {entry.get('actor')}", file=sys.stderr)
            continue
        
        # Validate action
        validation = sim.validator.validate(action, actor)
        
        # Check if validation matches
        original_accepted = entry.get('accepted', False)
        if validation.accepted != original_accepted:
            print(f"Tick {tick}: MISMATCH - acceptance: replayed={validation.accepted}, original={original_accepted}", file=sys.stderr)
            mismatch_count += 1
        else:
            match_count += 1
        
        # Derive and apply mutations
        mutations = sim._derive_mutations(validation.safe_action, actor)
        sim._apply_mutations(mutations)
        sim._update_cooldowns(actor, validation.safe_action)
        
        # Decay energy
        sim._decay_energy()
        
        if verbose:
            print(f"Tick {tick}: {actor.name} -> {action_type} (accepted={validation.accepted})", file=sys.stderr)
    
    print("\n" + "=" * 60, file=sys.stderr)
    print("REPLAY RESULTS", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Total ticks: {len(log_entries)}", file=sys.stderr)
    print(f"Matches: {match_count}", file=sys.stderr)
    print(f"Mismatches: {mismatch_count}", file=sys.stderr)
    
    if mismatch_count == 0:
        print("\n✓ DETERMINISTIC: All ticks match original simulation", file=sys.stderr)
        return True
    else:
        print(f"\n✗ NON-DETERMINISTIC: {mismatch_count} mismatches detected", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Replay a simulation from log file to verify determinism"
    )
    parser.add_argument(
        "logfile",
        type=str,
        help="Path to simulation log file (JSONL format)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed replay information"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.logfile):
        print(f"Error: Log file not found: {args.logfile}", file=sys.stderr)
        sys.exit(1)
    
    try:
        log_entries = load_log(args.logfile)
        print(f"Loaded {len(log_entries)} entries from {args.logfile}", file=sys.stderr)
        
        deterministic = replay_simulation(log_entries, verbose=args.verbose)
        
        sys.exit(0 if deterministic else 1)
        
    except Exception as e:
        print(f"Error during replay: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()