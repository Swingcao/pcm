#!/usr/bin/env python
# coding=utf-8
"""
Advanced Results Analysis
=========================
Additional analysis tools including:
- Multi-experiment comparison
- Surprisal distribution analysis
- Agent activation analysis
- Visualization support

Usage:
    python advanced_analysis.py --compare exp1 exp2
    python advanced_analysis.py --surprisal-analysis
    python advanced_analysis.py --visualize
"""

import json
import os
import re
import argparse
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
import sys

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.evaluation.analyze_results import (
    load_experiment_results,
    load_all_qa_pairs,
    compute_statistics,
    print_summary_table,
    aggregate_metrics
)


# =============================================================================
# Intermediate Results Loading (Interaction-level)
# =============================================================================

def load_intermediate_results(sample_dir: str) -> List[Dict[str, Any]]:
    """
    Load all intermediate interaction results from a sample directory.

    Args:
        sample_dir: Path to sample directory

    Returns:
        List of interaction results sorted by interaction_id
    """
    intermediate_dir = os.path.join(sample_dir, "intermediate")

    if not os.path.exists(intermediate_dir):
        return []

    results = []

    for filename in os.listdir(intermediate_dir):
        if filename.startswith("interaction_") and filename.endswith(".json"):
            filepath = os.path.join(intermediate_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")

    # Sort by interaction_id
    results.sort(key=lambda x: x.get("interaction_id", 0))

    return results


def load_all_intermediate(
    experiments_dir: str,
    experiment_name: str = "locomo_experiment"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all intermediate results from an experiment.

    Args:
        experiments_dir: Base experiments directory
        experiment_name: Name of the experiment

    Returns:
        Dict mapping sample_id to list of intermediate results
    """
    exp_path = os.path.join(experiments_dir, experiment_name)

    if not os.path.exists(exp_path):
        return {}

    all_intermediate = {}

    for item in sorted(os.listdir(exp_path)):
        if item.startswith("sample_"):
            sample_dir = os.path.join(exp_path, item)
            if os.path.isdir(sample_dir):
                results = load_intermediate_results(sample_dir)
                if results:
                    all_intermediate[item] = results

    return all_intermediate


# =============================================================================
# Surprisal Analysis
# =============================================================================

def analyze_surprisal_distribution(
    intermediate_results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Analyze surprisal distribution across all interactions.

    Args:
        intermediate_results: Dict of sample_id -> interactions

    Returns:
        Dict with surprisal statistics
    """
    all_surprisals = {
        "raw": [],
        "effective": [],
        "entropy": []
    }

    level_counts = defaultdict(int)
    agent_counts = defaultdict(int)

    for sample_id, interactions in intermediate_results.items():
        for interaction in interactions:
            surprisal = interaction.get("surprisal", {})

            if "raw" in surprisal:
                all_surprisals["raw"].append(surprisal["raw"])
            if "effective" in surprisal:
                all_surprisals["effective"].append(surprisal["effective"])
            if "entropy" in surprisal:
                all_surprisals["entropy"].append(surprisal["entropy"])

            level = surprisal.get("level", "unknown")
            level_counts[level] += 1

            evolution = interaction.get("evolution", {})
            agent = evolution.get("agent", "unknown")
            agent_counts[agent] += 1

    # Compute statistics
    def compute_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "std": 0}

        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        std = variance ** 0.5

        return {
            "count": n,
            "mean": mean,
            "min": min(values),
            "max": max(values),
            "std": std
        }

    return {
        "surprisal_stats": {
            "raw": compute_stats(all_surprisals["raw"]),
            "effective": compute_stats(all_surprisals["effective"]),
            "entropy": compute_stats(all_surprisals["entropy"])
        },
        "level_distribution": dict(level_counts),
        "agent_activation": dict(agent_counts),
        "total_interactions": sum(level_counts.values())
    }


def print_surprisal_report(analysis: Dict[str, Any]) -> None:
    """
    Print surprisal analysis report.

    Args:
        analysis: Surprisal analysis results
    """
    print("\n" + "=" * 70)
    print("Surprisal Distribution Analysis")
    print("=" * 70)

    print(f"\nTotal Interactions: {analysis['total_interactions']}")

    # Surprisal statistics
    print("\n" + "-" * 70)
    print("Surprisal Statistics")
    print("-" * 70)
    print(f"{'Metric':<15} {'Count':>8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for metric, stats in analysis["surprisal_stats"].items():
        print(f"{metric:<15} {stats['count']:>8} {stats['mean']:>10.4f} "
              f"{stats['std']:>10.4f} {stats['min']:>10.4f} {stats['max']:>10.4f}")

    # Level distribution
    print("\n" + "-" * 70)
    print("Surprisal Level Distribution")
    print("-" * 70)

    total = sum(analysis["level_distribution"].values())
    for level, count in sorted(analysis["level_distribution"].items()):
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{level:<12} {count:>6} ({pct:>5.1f}%) {bar}")

    # Agent activation
    print("\n" + "-" * 70)
    print("Cognitive Agent Activation")
    print("-" * 70)

    total = sum(analysis["agent_activation"].values())
    for agent, count in sorted(analysis["agent_activation"].items()):
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{agent:<15} {count:>6} ({pct:>5.1f}%) {bar}")

    print("=" * 70)


# =============================================================================
# Experiment Comparison
# =============================================================================

def compare_experiments(
    experiments_dir: str,
    experiment_names: List[str]
) -> Dict[str, Any]:
    """
    Compare results across multiple experiments.

    Args:
        experiments_dir: Base experiments directory
        experiment_names: List of experiment names to compare

    Returns:
        Comparison results
    """
    comparison = {}

    for exp_name in experiment_names:
        results = load_experiment_results(experiments_dir, exp_name)

        if not results:
            print(f"Warning: No results found for {exp_name}")
            continue

        qa_pairs = load_all_qa_pairs(results)
        stats = compute_statistics(qa_pairs)

        comparison[exp_name] = {
            "overall": stats["overall"],
            "by_category": stats["by_category"],
            "num_samples": len(stats["by_sample"]),
            "total_qa": stats["overall"]["count"]
        }

    return comparison


def print_comparison_table(comparison: Dict[str, Any]) -> None:
    """
    Print experiment comparison table.

    Args:
        comparison: Comparison results
    """
    if not comparison:
        print("No experiments to compare.")
        return

    experiments = list(comparison.keys())

    print("\n" + "=" * 90)
    print("Experiment Comparison")
    print("=" * 90)

    # Header
    header = f"{'Metric':<20}"
    for exp in experiments:
        header += f" {exp:<15}"
    print(header)
    print("-" * 90)

    # Basic stats
    for metric in ["total_qa", "num_samples"]:
        row = f"{metric:<20}"
        for exp in experiments:
            value = comparison[exp].get(metric, 0)
            row += f" {value:<15}"
        print(row)

    print("-" * 90)

    # Main metrics
    metrics = ["exact_match", "f1", "bleu1", "contains_match"]
    for metric in metrics:
        row = f"{metric:<20}"
        for exp in experiments:
            value = comparison[exp]["overall"].get(metric, 0)
            row += f" {value:<15.4f}"
        print(row)

    print("=" * 90)

    # Per-category comparison
    print("\n" + "-" * 90)
    print("F1 Score by Category")
    print("-" * 90)

    all_categories = set()
    for exp in experiments:
        all_categories.update(comparison[exp]["by_category"].keys())

    header = f"{'Category':<20}"
    for exp in experiments:
        header += f" {exp:<15}"
    print(header)
    print("-" * 90)

    for cat in sorted(all_categories):
        row = f"{cat:<20}"
        for exp in experiments:
            cat_stats = comparison[exp]["by_category"].get(cat, {})
            f1 = cat_stats.get("f1", 0)
            row += f" {f1:<15.4f}"
        print(row)

    print("=" * 90)


# =============================================================================
# Error Analysis
# =============================================================================

def analyze_errors(stats: Dict[str, Any], threshold: float = 0.3) -> Dict[str, Any]:
    """
    Detailed error analysis.

    Args:
        stats: Statistics from compute_statistics
        threshold: F1 threshold for "error"

    Returns:
        Error analysis results
    """
    errors = [qa for qa in stats["detailed"]
              if qa["metrics"]["f1"] < threshold]

    # Group errors by category
    by_category = defaultdict(list)
    for error in errors:
        by_category[error["category_name"]].append(error)

    # Analyze error patterns
    error_patterns = {
        "empty_prediction": [],
        "numeric_mismatch": [],
        "partial_match": [],
        "completely_wrong": []
    }

    for error in errors:
        pred = error["predicted_answer"].strip()
        ref = error["reference_answer"].strip()
        metrics = error["metrics"]

        if not pred or pred.lower() in ["n/a", "unknown", "error"]:
            error_patterns["empty_prediction"].append(error)
        elif metrics["contains_match"] > 0:
            error_patterns["partial_match"].append(error)
        elif metrics["numeric_match"] == 0 and re.search(r'\d', ref):
            error_patterns["numeric_mismatch"].append(error)
        else:
            error_patterns["completely_wrong"].append(error)

    return {
        "total_errors": len(errors),
        "error_rate": len(errors) / len(stats["detailed"]) if stats["detailed"] else 0,
        "by_category": {cat: len(items) for cat, items in by_category.items()},
        "error_patterns": {pat: len(items) for pat, items in error_patterns.items()},
        "errors": errors
    }


def print_error_analysis(error_analysis: Dict[str, Any]) -> None:
    """
    Print error analysis report.

    Args:
        error_analysis: Error analysis results
    """
    print("\n" + "=" * 70)
    print("Error Analysis")
    print("=" * 70)

    print(f"\nTotal Errors: {error_analysis['total_errors']}")
    print(f"Error Rate: {error_analysis['error_rate']:.2%}")

    print("\n" + "-" * 70)
    print("Errors by Category")
    print("-" * 70)

    for cat, count in sorted(error_analysis["by_category"].items()):
        print(f"  {cat}: {count}")

    print("\n" + "-" * 70)
    print("Error Patterns")
    print("-" * 70)

    for pattern, count in sorted(error_analysis["error_patterns"].items(),
                                  key=lambda x: -x[1]):
        print(f"  {pattern}: {count}")

    print("=" * 70)


# =============================================================================
# Visualization Support
# =============================================================================

def generate_visualization_data(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate data for visualization (compatible with common charting libraries).

    Args:
        stats: Statistics from compute_statistics

    Returns:
        Visualization-ready data
    """
    # Metrics over categories
    categories = []
    metrics_data = defaultdict(list)

    for cat_name, cat_stats in sorted(stats["by_category"].items()):
        categories.append(cat_name)
        for metric in ["exact_match", "f1", "bleu1", "contains_match"]:
            metrics_data[metric].append(cat_stats.get(metric, 0))

    # Sample performance
    samples = []
    sample_f1 = []
    for sample_id, sample_stats in sorted(stats["by_sample"].items()):
        samples.append(sample_id)
        sample_f1.append(sample_stats.get("f1", 0))

    # Score distribution
    f1_scores = [qa["metrics"]["f1"] for qa in stats["detailed"]]
    f1_buckets = defaultdict(int)
    for score in f1_scores:
        bucket = int(score * 10) / 10  # 0.0, 0.1, 0.2, ...
        f1_buckets[bucket] += 1

    return {
        "category_chart": {
            "labels": categories,
            "datasets": [
                {"label": metric, "data": values}
                for metric, values in metrics_data.items()
            ]
        },
        "sample_chart": {
            "labels": samples,
            "data": sample_f1
        },
        "f1_histogram": {
            "buckets": sorted(f1_buckets.keys()),
            "counts": [f1_buckets[b] for b in sorted(f1_buckets.keys())]
        }
    }


def save_visualization_data(viz_data: Dict[str, Any], output_path: str) -> None:
    """
    Save visualization data to JSON.

    Args:
        viz_data: Visualization data
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, ensure_ascii=False, indent=2)

    print(f"Visualization data saved to: {output_path}")


def print_ascii_chart(data: List[Tuple[str, float]], title: str, max_width: int = 40) -> None:
    """
    Print a simple ASCII bar chart.

    Args:
        data: List of (label, value) tuples
        title: Chart title
        max_width: Maximum bar width
    """
    if not data:
        return

    print(f"\n{title}")
    print("-" * (max_width + 25))

    max_val = max(v for _, v in data)

    for label, value in data:
        bar_len = int(value / max_val * max_width) if max_val > 0 else 0
        bar = "█" * bar_len
        print(f"{label:<15} {value:>6.4f} {bar}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Advanced experiment analysis tools",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default=None,
        help="Base results directory"
    )

    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default="locomo_experiment",
        help="Primary experiment name"
    )

    parser.add_argument(
        "--compare", "-c",
        nargs="+",
        type=str,
        help="Experiments to compare"
    )

    parser.add_argument(
        "--surprisal",
        action="store_true",
        help="Analyze surprisal distribution"
    )

    parser.add_argument(
        "--errors",
        action="store_true",
        help="Detailed error analysis"
    )

    parser.add_argument(
        "--error-threshold",
        type=float,
        default=0.3,
        help="F1 threshold for errors"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization data"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path"
    )

    args = parser.parse_args()

    # Setup paths
    results_dir = args.results_dir or config.RESULTS_DIR
    experiments_dir = os.path.join(results_dir, "experiments")

    # Comparison mode
    if args.compare:
        comparison = compare_experiments(experiments_dir, args.compare)
        print_comparison_table(comparison)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
            print(f"\nComparison saved to: {args.output}")

        return

    # Surprisal analysis
    if args.surprisal:
        intermediate = load_all_intermediate(experiments_dir, args.experiment)

        if not intermediate:
            print("No intermediate results found for surprisal analysis.")
            return

        analysis = analyze_surprisal_distribution(intermediate)
        print_surprisal_report(analysis)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            print(f"\nSurprisal analysis saved to: {args.output}")

        return

    # Error analysis
    if args.errors:
        results = load_experiment_results(experiments_dir, args.experiment)

        if not results:
            print("No results found for error analysis.")
            return

        qa_pairs = load_all_qa_pairs(results)
        stats = compute_statistics(qa_pairs)
        error_analysis = analyze_errors(stats, args.error_threshold)
        print_error_analysis(error_analysis)

        if args.output:
            # Don't include full error details for JSON
            output_data = {k: v for k, v in error_analysis.items() if k != "errors"}
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\nError analysis saved to: {args.output}")

        return

    # Visualization mode
    if args.visualize:
        results = load_experiment_results(experiments_dir, args.experiment)

        if not results:
            print("No results found for visualization.")
            return

        qa_pairs = load_all_qa_pairs(results)
        stats = compute_statistics(qa_pairs)
        viz_data = generate_visualization_data(stats)

        output_path = args.output or f"./viz_data_{args.experiment}.json"
        save_visualization_data(viz_data, output_path)

        # Print ASCII preview
        category_data = [
            (label, stats["by_category"].get(label, {}).get("f1", 0))
            for label in ["single-hop", "temporal", "multi-hop", "adversarial"]
        ]
        print_ascii_chart(category_data, "F1 by Category")

        return

    # Default: Print basic summary
    results = load_experiment_results(experiments_dir, args.experiment)

    if not results:
        print(f"No results found for experiment: {args.experiment}")
        return

    qa_pairs = load_all_qa_pairs(results)
    stats = compute_statistics(qa_pairs)
    print_summary_table(stats)


if __name__ == "__main__":
    main()
