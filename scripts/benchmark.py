#!/usr/bin/env python3
"""
Benchmark mode for running multiple simulations with different seeds.

Collects aggregate statistics for analysis:
- Action distribution
- Rejection counts
- Theft/gift frequency
- Average trust/affinity
- Emergent cooperation/conflict rates

Usage:
    python scripts/benchmark.py --runs 100 --ticks 20
"""

import argparse
import csv
import json
import sys
import os
import random
from typing import Any
from collections import defaultdict

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine.simulator import Simulator
from app.models.actions import ActionType


def run_simulation(seed: int, ticks: int, verbose: bool = False) -> dict[str, Any]:
    """
    Run a single simulation with a given seed.
    
    Args:
        seed: Random seed for reproducibility
        ticks: Number of ticks to run
        verbose: Whether to print progress
    
    Returns:
        Dictionary of metrics from the simulation
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create simulator (uses MockProvider by default)
    sim = Simulator()
    
    # Run simulation
    results = sim.run(max_ticks=ticks)
    
    # Extract metrics
    metrics = {
        "seed": seed,
        "ticks": ticks,
        "total_actions": sum([
            sim.metrics["moves"],
            sim.metrics["explores"],
            sim.metrics["follows"],
            sim.metrics["conversations"],
            sim.metrics["gifts"],
            sim.metrics["helps"],
            sim.metrics["requests_help"],
            sim.metrics["warns"],
            sim.metrics["trades"],
            sim.metrics["ignores"],
            sim.metrics["works"],
            sim.metrics["rests"],
            sim.metrics["observes"],
            sim.metrics["thefts"],
        ]),
        **sim.metrics,
    }
    
    # Calculate derived metrics
    if metrics["total_actions"] > 0:
        metrics["rejection_rate"] = (metrics["rejected_actions"] / metrics["total_actions"]) * 100
    else:
        metrics["rejection_rate"] = 0.0
    
    # Calculate cooperation rate (gifts + helps + trades)
    metrics["cooperation_rate"] = metrics["gifts"] + metrics["helps"] + metrics["trades"]
    if metrics["total_actions"] > 0:
        metrics["cooperation_pct"] = (metrics["cooperation_rate"] / metrics["total_actions"]) * 100
    else:
        metrics["cooperation_pct"] = 0.0
    
    # Calculate conflict rate (thefts + ignores)
    metrics["conflict_rate"] = metrics["thefts"] + metrics["ignores"]
    if metrics["total_actions"] > 0:
        metrics["conflict_pct"] = (metrics["conflict_rate"] / metrics["total_actions"]) * 100
    else:
        metrics["conflict_pct"] = 0.0
    
    # Calculate social rate (conversations + warns)
    metrics["social_rate"] = metrics["conversations"] + metrics["warns"]
    if metrics["total_actions"] > 0:
        metrics["social_pct"] = (metrics["social_rate"] / metrics["total_actions"]) * 100
    else:
        metrics["social_pct"] = 0.0
    
    # Calculate economic rate (works + trades)
    metrics["economic_rate"] = metrics["works"] + metrics["trades"]
    if metrics["total_actions"] > 0:
        metrics["economic_pct"] = (metrics["economic_rate"] / metrics["total_actions"]) * 100
    else:
        metrics["economic_pct"] = 0.0
    
    # Calculate final social metrics
    total_trust = 0.0
    total_affinity = 0.0
    relationship_count = 0
    
    for citizen in sim.world.citizens.values():
        for rel in citizen.relationships.values():
            total_trust += rel.trust
            total_affinity += rel.affinity
            relationship_count += 1
    
    if relationship_count > 0:
        metrics["avg_trust"] = total_trust / relationship_count
        metrics["avg_affinity"] = total_affinity / relationship_count
    else:
        metrics["avg_trust"] = 0.0
        metrics["avg_affinity"] = 0.0
    
    # Calculate final gold distribution
    gold_values = [c.gold for c in sim.world.citizens.values()]
    if gold_values:
        metrics["avg_gold"] = sum(gold_values) / len(gold_values)
        metrics["min_gold"] = min(gold_values)
        metrics["max_gold"] = max(gold_values)
        
        # Calculate Gini coefficient
        gold_values_sorted = sorted(gold_values)
        n = len(gold_values_sorted)
        if n > 0 and sum(gold_values_sorted) > 0:
            cumsum = 0
            gini_numerator = 0
            for i, g in enumerate(gold_values_sorted):
                cumsum += g
                gini_numerator += (2 * (i + 1) - n - 1) * g
            gini = gini_numerator / (n * cumsum)
            metrics["gini_coefficient"] = gini
        else:
            metrics["gini_coefficient"] = 0.0
    else:
        metrics["avg_gold"] = 0.0
        metrics["min_gold"] = 0
        metrics["max_gold"] = 0
        metrics["gini_coefficient"] = 0.0
    
    if verbose:
        print(f"\nSeed {seed}: {metrics['total_actions']} actions, "
              f"{metrics['cooperation_pct']:.1f}% cooperation, "
              f"{metrics['conflict_pct']:.1f}% conflict", file=sys.stderr)
    
    return metrics


def run_benchmark(runs: int, ticks: int, output_csv: str | None = None, verbose: bool = False) -> None:
    """
    Run benchmark with multiple simulations.
    
    Args:
        runs: Number of simulation runs
        ticks: Ticks per simulation
        output_csv: Optional path to save CSV results
        verbose: Whether to print progress
    """
    print("=" * 60, file=sys.stderr)
    print("BENCHMARK MODE", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Runs: {runs}", file=sys.stderr)
    print(f"Ticks per run: {ticks}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    all_metrics = []
    
    for i in range(runs):
        seed = i + 1  # Deterministic seeds
        if verbose:
            print(f"\nRun {i+1}/{runs} (seed={seed})...", file=sys.stderr)
        
        metrics = run_simulation(seed, ticks, verbose=verbose)
        all_metrics.append(metrics)
    
    # Calculate aggregate statistics
    print("\n" + "=" * 60, file=sys.stderr)
    print("AGGREGATE STATISTICS", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    # Action distribution
    action_keys = [
        "moves", "explores", "follows", "conversations", "gifts",
        "helps", "requests_help", "warns", "trades", "ignores",
        "works", "rests", "observes", "thefts"
    ]
    
    print("\nAction Distribution (averages per simulation):", file=sys.stderr)
    for key in action_keys:
        avg = sum(m[key] for m in all_metrics) / len(all_metrics)
        print(f"  {key.replace('_', ' ').title()}: {avg:.2f}", file=sys.stderr)
    
    print(f"\nValidation:", file=sys.stderr)
    avg_rejection = sum(m["rejection_rate"] for m in all_metrics) / len(all_metrics)
    print(f"  Average Rejection Rate: {avg_rejection:.1f}%", file=sys.stderr)
    
    print(f"\nEconomic Activity:", file=sys.stderr)
    avg_gifted = sum(m["total_gold_gifted"] for m in all_metrics) / len(all_metrics)
    avg_traded = sum(m["total_gold_traded"] for m in all_metrics) / len(all_metrics)
    avg_earned = sum(m["total_gold_earned"] for m in all_metrics) / len(all_metrics)
    avg_stolen = sum(m["total_gold_stolen"] for m in all_metrics) / len(all_metrics)
    print(f"  Avg Gold Gifted: {avg_gifted:.2f}", file=sys.stderr)
    print(f"  Avg Gold Traded: {avg_traded:.2f}", file=sys.stderr)
    print(f"  Avg Gold Earned: {avg_earned:.2f}", file=sys.stderr)
    print(f"  Avg Gold Stolen: {avg_stolen:.2f}", file=sys.stderr)
    
    print(f"\nSocial Metrics:", file=sys.stderr)
    avg_trust = sum(m["avg_trust"] for m in all_metrics) / len(all_metrics)
    avg_affinity = sum(m["avg_affinity"] for m in all_metrics) / len(all_metrics)
    avg_gini = sum(m["gini_coefficient"] for m in all_metrics) / len(all_metrics)
    print(f"  Average Trust: {avg_trust:.3f}", file=sys.stderr)
    print(f"  Average Affinity: {avg_affinity:.3f}", file=sys.stderr)
    print(f"  Average Gini Coefficient: {avg_gini:.3f}", file=sys.stderr)
    
    print(f"\nEmergent Behavior:", file=sys.stderr)
    avg_coop = sum(m["cooperation_pct"] for m in all_metrics) / len(all_metrics)
    avg_conflict = sum(m["conflict_pct"] for m in all_metrics) / len(all_metrics)
    avg_social = sum(m["social_pct"] for m in all_metrics) / len(all_metrics)
    avg_economic = sum(m["economic_pct"] for m in all_metrics) / len(all_metrics)
    print(f"  Cooperation Rate: {avg_coop:.1f}%", file=sys.stderr)
    print(f"  Conflict Rate: {avg_conflict:.1f}%", file=sys.stderr)
    print(f"  Social Rate: {avg_social:.1f}%", file=sys.stderr)
    print(f"  Economic Rate: {avg_economic:.1f}%", file=sys.stderr)
    
    print(f"\nGold Distribution:", file=sys.stderr)
    avg_gold = sum(m["avg_gold"] for m in all_metrics) / len(all_metrics)
    min_gold = min(m["min_gold"] for m in all_metrics)
    max_gold = max(m["max_gold"] for m in all_metrics)
    print(f"  Average Gold: {avg_gold:.2f}", file=sys.stderr)
    print(f"  Min Gold: {min_gold}", file=sys.stderr)
    print(f"  Max Gold: {max_gold}", file=sys.stderr)
    
    print("=" * 60, file=sys.stderr)
    
    # Save to CSV if requested
    if output_csv:
        save_to_csv(all_metrics, output_csv)
        print(f"\nResults saved to: {output_csv}", file=sys.stderr)


def save_to_csv(metrics_list: list[dict], filepath: str) -> None:
    """Save benchmark results to CSV file."""
    if not metrics_list:
        return
    
    # Get all keys from first metric
    fieldnames = list(metrics_list[0].keys())
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark simulations with aggregate statistics"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of simulation runs (default: 10)"
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=20,
        help="Number of ticks per simulation (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: None)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    args = parser.parse_args()
    
    try:
        run_benchmark(
            runs=args.runs,
            ticks=args.ticks,
            output_csv=args.output,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Error during benchmark: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()