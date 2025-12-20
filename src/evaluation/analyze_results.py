#!/usr/bin/env python
# coding=utf-8
"""
ProCoMemory Results Analysis Script
===================================
Comprehensive analysis and statistics for experiment results.

Features:
- Load and parse experiment results from JSON files
- Compute F1, BLEU-1, Exact Match, and other metrics
- Statistics by sample, category, and overall
- Detailed reports and visualizations
- Export to JSON/CSV formats

Usage:
    python analyze_results.py                           # Analyze default experiment
    python analyze_results.py --experiment my_exp       # Analyze specific experiment
    python analyze_results.py --results-dir ./results   # Custom results directory
    python analyze_results.py --output ./report.json    # Custom output path
    python analyze_results.py --format csv              # Output as CSV
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


# =============================================================================
# Tokenization and Basic Metrics
# =============================================================================

def tokenize(text: str) -> List[str]:
    """
    Simple tokenization: lowercase, split by whitespace and punctuation.

    Args:
        text: Input text string

    Returns:
        List of tokens
    """
    if text is None:
        return []
    text = str(text).lower()
    # Remove punctuation, keep alphanumeric and whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return tokens


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.

    Operations:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove articles
    articles = ['a', 'an', 'the']
    words = text.split()
    words = [w for w in words if w not in articles]
    text = ' '.join(words).strip()

    return text


# =============================================================================
# Metric Computation Functions
# =============================================================================

def compute_f1(prediction: str, reference: str) -> Tuple[float, float, float]:
    """
    Compute F1 score at token level.

    Args:
        prediction: Predicted answer
        reference: Reference answer

    Returns:
        Tuple of (precision, recall, f1)
    """
    pred_tokens = set(tokenize(prediction))
    ref_tokens = set(tokenize(reference))

    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0, 0.0, 0.0

    common = pred_tokens & ref_tokens

    precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = len(common) / len(ref_tokens) if len(ref_tokens) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def compute_bleu1(prediction: str, reference: str) -> float:
    """
    Compute BLEU-1 score (unigram precision with brevity penalty).

    Args:
        prediction: Predicted answer
        reference: Reference answer

    Returns:
        BLEU-1 score between 0 and 1
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if len(pred_tokens) == 0:
        return 0.0
    if len(ref_tokens) == 0:
        return 0.0

    # Count reference tokens
    ref_token_counts = defaultdict(int)
    for token in ref_tokens:
        ref_token_counts[token] += 1

    # Count matches with clipping
    match_count = 0
    pred_token_counts = defaultdict(int)
    for token in pred_tokens:
        pred_token_counts[token] += 1

    for token, count in pred_token_counts.items():
        match_count += min(count, ref_token_counts.get(token, 0))

    precision = match_count / len(pred_tokens)

    # Brevity penalty
    if len(pred_tokens) >= len(ref_tokens):
        bp = 1.0
    else:
        bp = len(pred_tokens) / len(ref_tokens)

    bleu1 = bp * precision
    return bleu1


def exact_match(prediction: str, reference: str) -> float:
    """
    Exact match after normalization.

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    return 1.0 if pred == ref else 0.0


def contains_match(prediction: str, reference: str) -> float:
    """
    Check if reference is contained in prediction or vice versa.

    Returns:
        1.0 if contained, 0.0 otherwise
    """
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)

    if ref in pred or pred in ref:
        return 1.0
    return 0.0


def numeric_match(prediction: str, reference: str) -> float:
    """
    Extract and compare numeric values.

    Returns:
        1.0 if any numeric value matches, 0.0 otherwise
    """
    pred_numbers = re.findall(r'\d+', prediction)
    ref_numbers = re.findall(r'\d+', reference)

    if not pred_numbers or not ref_numbers:
        return 0.0

    for ref_num in ref_numbers:
        if ref_num in pred_numbers:
            return 1.0

    return 0.0


def word_overlap(prediction: str, reference: str) -> float:
    """
    Compute word overlap (Jaccard similarity).

    Returns:
        Overlap ratio between 0 and 1
    """
    pred_tokens = set(tokenize(prediction))
    ref_tokens = set(tokenize(reference))

    if not pred_tokens or not ref_tokens:
        return 0.0

    intersection = pred_tokens & ref_tokens
    union = pred_tokens | ref_tokens

    return len(intersection) / len(union)


# =============================================================================
# Result Loading
# =============================================================================

def load_sample_results(sample_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load results from a sample directory.

    Args:
        sample_dir: Path to sample directory

    Returns:
        Dict with sample results or None if not found
    """
    results_path = os.path.join(sample_dir, "results.json")

    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    return None


def load_experiment_results(
    experiment_dir: str,
    experiment_name: str = "locomo_experiment"
) -> List[Dict[str, Any]]:
    """
    Load all results from an experiment.

    Args:
        experiment_dir: Base experiments directory
        experiment_name: Name of the experiment

    Returns:
        List of sample results
    """
    exp_path = os.path.join(experiment_dir, experiment_name)

    if not os.path.exists(exp_path):
        print(f"Experiment directory not found: {exp_path}")
        return []

    all_results = []

    # Find all sample directories
    for item in sorted(os.listdir(exp_path)):
        if item.startswith("sample_"):
            sample_dir = os.path.join(exp_path, item)
            if os.path.isdir(sample_dir):
                result = load_sample_results(sample_dir)
                if result:
                    all_results.append(result)

    return all_results


def load_all_qa_pairs(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract all QA pairs from results.

    Args:
        results: List of sample results

    Returns:
        Flat list of all QA results with sample info
    """
    qa_pairs = []

    for sample in results:
        sample_id = sample.get("sample_id", "unknown")

        for qa in sample.get("qa_results", []):
            qa_item = {
                "sample_id": sample_id,
                "question": qa.get("question", ""),
                "reference_answer": qa.get("reference_answer", ""),
                "predicted_answer": qa.get("predicted_answer", ""),
                "category": qa.get("category", 0),
                "category_name": qa.get("category_name", "unknown"),
                "evidence": qa.get("evidence", []),
                "original_metrics": qa.get("metrics", {}),
                "retrieved_context": qa.get("retrieved_context", [])
            }
            qa_pairs.append(qa_item)

    return qa_pairs


# =============================================================================
# Statistics Computation
# =============================================================================

def compute_qa_metrics(qa_item: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute all metrics for a single QA pair.

    Args:
        qa_item: QA pair with prediction and reference

    Returns:
        Dict with all computed metrics
    """
    pred = qa_item.get("predicted_answer", "")
    ref = qa_item.get("reference_answer", "")

    precision, recall, f1 = compute_f1(pred, ref)

    return {
        "exact_match": exact_match(pred, ref),
        "contains_match": contains_match(pred, ref),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "bleu1": compute_bleu1(pred, ref),
        "word_overlap": word_overlap(pred, ref),
        "numeric_match": numeric_match(pred, ref)
    }


def aggregate_metrics(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate metrics from a list of items.

    Args:
        items: List of items with computed metrics

    Returns:
        Dict with average metrics
    """
    if not items:
        return {
            "count": 0,
            "exact_match": 0.0,
            "contains_match": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "bleu1": 0.0,
            "word_overlap": 0.0,
            "numeric_match": 0.0
        }

    metric_keys = ["exact_match", "contains_match", "f1", "precision",
                   "recall", "bleu1", "word_overlap", "numeric_match"]

    aggregated = {"count": len(items)}

    for key in metric_keys:
        values = [item["metrics"].get(key, 0.0) for item in items]
        aggregated[key] = sum(values) / len(values)

    return aggregated


def compute_statistics(qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for all QA pairs.

    Args:
        qa_pairs: List of QA pairs

    Returns:
        Dict with statistics by sample, category, and overall
    """
    # Compute metrics for each QA pair
    for qa in qa_pairs:
        qa["metrics"] = compute_qa_metrics(qa)

    # Group by sample
    by_sample = defaultdict(list)
    for qa in qa_pairs:
        by_sample[qa["sample_id"]].append(qa)

    # Group by category
    by_category = defaultdict(list)
    for qa in qa_pairs:
        by_category[qa["category"]].append(qa)

    # Category names mapping
    category_names = {
        1: "single-hop",
        2: "temporal",
        3: "multi-hop",
        4: "adversarial",
        5: "open-domain"
    }

    # Compute sample-level statistics
    sample_stats = {}
    for sample_id, items in sorted(by_sample.items()):
        sample_stats[sample_id] = aggregate_metrics(items)

    # Compute category-level statistics
    category_stats = {}
    for cat_id, items in sorted(by_category.items()):
        cat_name = category_names.get(cat_id, f"category_{cat_id}")
        category_stats[cat_name] = aggregate_metrics(items)
        category_stats[cat_name]["category_id"] = cat_id

    # Compute overall statistics
    overall_stats = aggregate_metrics(qa_pairs)

    return {
        "overall": overall_stats,
        "by_sample": sample_stats,
        "by_category": category_stats,
        "detailed": qa_pairs
    }


# =============================================================================
# Report Generation
# =============================================================================

def generate_text_report(
    stats: Dict[str, Any],
    experiment_name: str = "",
    error_threshold: float = 0.3,
    max_errors: int = 20
) -> str:
    """
    Generate a formatted text report.

    Args:
        stats: Statistics dict from compute_statistics
        experiment_name: Name of the experiment
        error_threshold: F1 threshold for errors
        max_errors: Maximum number of errors to show

    Returns:
        Formatted text report string
    """
    lines = []

    # Header
    lines.append("=" * 85)
    lines.append("ProCoMemory Experiment Results Analysis")
    if experiment_name:
        lines.append(f"Experiment: {experiment_name}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 85)

    # Overall summary
    overall = stats["overall"]
    lines.append(f"\nTotal QA Pairs: {overall['count']}")
    lines.append(f"Total Samples: {len(stats['by_sample'])}")

    # Category name mappings for display
    category_display = {
        "single-hop": "Single-hop (1)",
        "temporal": "Temporal (2)",
        "multi-hop": "Multi-hop (3)",
        "adversarial": "Adversarial (4)",
        "open-domain": "Open-domain (5)"
    }

    # Metrics table header
    lines.append("\n" + "-" * 85)
    lines.append(f"{'Category':<25} {'Count':>6} {'EM':>8} {'F1':>8} {'BLEU-1':>8} {'Contains':>9} {'NumMatch':>9}")
    lines.append("-" * 85)

    # Category rows
    for cat_name in ["single-hop", "temporal", "multi-hop", "adversarial", "open-domain"]:
        if cat_name in stats["by_category"]:
            cat_stats = stats["by_category"][cat_name]
            display_name = category_display.get(cat_name, cat_name)
            lines.append(f"{display_name:<25} {cat_stats['count']:>6} "
                        f"{cat_stats['exact_match']:>8.4f} "
                        f"{cat_stats['f1']:>8.4f} "
                        f"{cat_stats['bleu1']:>8.4f} "
                        f"{cat_stats['contains_match']:>9.4f} "
                        f"{cat_stats['numeric_match']:>9.4f}")

    # Overall row
    lines.append("-" * 85)
    lines.append(f"{'Overall':<25} {overall['count']:>6} "
                f"{overall['exact_match']:>8.4f} "
                f"{overall['f1']:>8.4f} "
                f"{overall['bleu1']:>8.4f} "
                f"{overall['contains_match']:>9.4f} "
                f"{overall['numeric_match']:>9.4f}")
    lines.append("=" * 85)

    # Sample-level summary
    lines.append("\n" + "-" * 85)
    lines.append("Per-Sample Summary")
    lines.append("-" * 85)
    lines.append(f"{'Sample ID':<20} {'Count':>6} {'EM':>8} {'F1':>8} {'BLEU-1':>8} {'Precision':>10} {'Recall':>8}")
    lines.append("-" * 85)

    for sample_id, sample_stats in sorted(stats["by_sample"].items()):
        lines.append(f"{sample_id:<20} {sample_stats['count']:>6} "
                    f"{sample_stats['exact_match']:>8.4f} "
                    f"{sample_stats['f1']:>8.4f} "
                    f"{sample_stats['bleu1']:>8.4f} "
                    f"{sample_stats['precision']:>10.4f} "
                    f"{sample_stats['recall']:>8.4f}")

    lines.append("=" * 85)

    # Score distribution summary
    lines.append("\n" + "-" * 85)
    lines.append("Score Distribution Summary")
    lines.append("-" * 85)

    f1_scores = [qa["metrics"]["f1"] for qa in stats["detailed"]]
    bleu_scores = [qa["metrics"]["bleu1"] for qa in stats["detailed"]]

    if f1_scores:
        lines.append(f"F1 Score:    Mean={sum(f1_scores)/len(f1_scores):.4f}, "
                    f"Min={min(f1_scores):.4f}, Max={max(f1_scores):.4f}, "
                    f"Median={sorted(f1_scores)[len(f1_scores)//2]:.4f}")
    if bleu_scores:
        lines.append(f"BLEU-1:      Mean={sum(bleu_scores)/len(bleu_scores):.4f}, "
                    f"Min={min(bleu_scores):.4f}, Max={max(bleu_scores):.4f}, "
                    f"Median={sorted(bleu_scores)[len(bleu_scores)//2]:.4f}")

    lines.append("=" * 85)

    # Error analysis
    errors = [qa for qa in stats["detailed"] if qa["metrics"]["f1"] < error_threshold]

    lines.append(f"\n" + "=" * 85)
    lines.append(f"Error Analysis (F1 < {error_threshold})")
    lines.append("=" * 85)
    lines.append(f"Total Errors: {len(errors)} / {len(stats['detailed'])} ({len(errors)/len(stats['detailed'])*100:.1f}%)")

    # Error by category
    lines.append("\nErrors by Category:")
    for cat_name in ["single-hop", "temporal", "multi-hop", "adversarial", "open-domain"]:
        cat_errors = [e for e in errors if e["category_name"] == cat_name]
        cat_total = len([q for q in stats["detailed"] if q["category_name"] == cat_name])
        if cat_total > 0:
            lines.append(f"  {category_display.get(cat_name, cat_name)}: {len(cat_errors)}/{cat_total} "
                        f"({len(cat_errors)/cat_total*100:.1f}%)")

    # Low scoring examples
    if errors:
        errors.sort(key=lambda x: x["metrics"]["f1"])
        lines.append(f"\nLowest Scoring QA Pairs (showing {min(len(errors), max_errors)}):")
        lines.append("-" * 85)

        for i, qa in enumerate(errors[:max_errors]):
            lines.append(f"\n[{i+1}] Sample: {qa['sample_id']} | Category: {qa['category_name']}")
            lines.append(f"    Question: {qa['question']}")
            lines.append(f"    Reference: {qa['reference_answer']}")
            lines.append(f"    Predicted: {qa['predicted_answer']}")
            lines.append(f"    Metrics: EM={qa['metrics']['exact_match']:.2f}, "
                        f"F1={qa['metrics']['f1']:.2f}, "
                        f"BLEU-1={qa['metrics']['bleu1']:.2f}, "
                        f"Contains={qa['metrics']['contains_match']:.2f}")

    lines.append("\n" + "=" * 85)
    lines.append("End of Report")
    lines.append("=" * 85)

    return "\n".join(lines)


def save_text_report(
    stats: Dict[str, Any],
    output_path: str,
    experiment_name: str = "",
    error_threshold: float = 0.3
) -> None:
    """
    Save text report to file.

    Args:
        stats: Statistics dict
        output_path: Output file path
        experiment_name: Name of the experiment
        error_threshold: F1 threshold for errors
    """
    report = generate_text_report(stats, experiment_name, error_threshold)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Text report saved to: {output_path}")


def print_summary_table(stats: Dict[str, Any]) -> None:
    """
    Print a formatted summary table.

    Args:
        stats: Statistics dict from compute_statistics
    """
    # Category name mappings for display
    category_display = {
        "single-hop": "Single-hop (1)",
        "temporal": "Temporal (2)",
        "multi-hop": "Multi-hop (3)",
        "adversarial": "Adversarial (4)",
        "open-domain": "Open-domain (5)"
    }

    print("\n" + "=" * 85)
    print("ProCoMemory Experiment Results Analysis")
    print("=" * 85)

    # Overall summary
    overall = stats["overall"]
    print(f"\nTotal QA Pairs: {overall['count']}")
    print(f"Total Samples: {len(stats['by_sample'])}")

    # Metrics table header
    print("\n" + "-" * 85)
    print(f"{'Category':<25} {'Count':>6} {'EM':>8} {'F1':>8} {'BLEU-1':>8} {'Contains':>9} {'NumMatch':>9}")
    print("-" * 85)

    # Category rows
    for cat_name in ["single-hop", "temporal", "multi-hop", "adversarial", "open-domain"]:
        if cat_name in stats["by_category"]:
            cat_stats = stats["by_category"][cat_name]
            display_name = category_display.get(cat_name, cat_name)
            print(f"{display_name:<25} {cat_stats['count']:>6} "
                  f"{cat_stats['exact_match']:>8.4f} "
                  f"{cat_stats['f1']:>8.4f} "
                  f"{cat_stats['bleu1']:>8.4f} "
                  f"{cat_stats['contains_match']:>9.4f} "
                  f"{cat_stats['numeric_match']:>9.4f}")

    # Overall row
    print("-" * 85)
    print(f"{'Overall':<25} {overall['count']:>6} "
          f"{overall['exact_match']:>8.4f} "
          f"{overall['f1']:>8.4f} "
          f"{overall['bleu1']:>8.4f} "
          f"{overall['contains_match']:>9.4f} "
          f"{overall['numeric_match']:>9.4f}")
    print("=" * 85)

    # Sample-level summary
    print("\n" + "-" * 85)
    print("Per-Sample Summary")
    print("-" * 85)
    print(f"{'Sample ID':<20} {'Count':>6} {'EM':>8} {'F1':>8} {'BLEU-1':>8}")
    print("-" * 85)

    for sample_id, sample_stats in sorted(stats["by_sample"].items()):
        print(f"{sample_id:<20} {sample_stats['count']:>6} "
              f"{sample_stats['exact_match']:>8.4f} "
              f"{sample_stats['f1']:>8.4f} "
              f"{sample_stats['bleu1']:>8.4f}")

    print("=" * 85)


def print_detailed_errors(
    stats: Dict[str, Any],
    threshold: float = 0.3,
    max_items: int = 10
) -> None:
    """
    Print detailed information about low-scoring QA pairs.

    Args:
        stats: Statistics dict
        threshold: F1 threshold for "error"
        max_items: Maximum items to show
    """
    errors = [qa for qa in stats["detailed"] if qa["metrics"]["f1"] < threshold]

    if not errors:
        print("\nNo significant errors found (all F1 >= {:.2f})".format(threshold))
        return

    # Sort by F1 score
    errors.sort(key=lambda x: x["metrics"]["f1"])

    print(f"\n{'=' * 85}")
    print(f"Low-Scoring QA Pairs (F1 < {threshold}) - Showing {min(len(errors), max_items)} of {len(errors)}")
    print("=" * 85)

    for i, qa in enumerate(errors[:max_items]):
        print(f"\n[{i+1}] Sample: {qa['sample_id']} | Category: {qa['category_name']}")
        print(f"    Q: {qa['question'][:80]}...")
        print(f"    Reference: {qa['reference_answer']}")
        print(f"    Predicted: {qa['predicted_answer'][:80]}...")
        print(f"    Metrics: EM={qa['metrics']['exact_match']:.2f}, "
              f"F1={qa['metrics']['f1']:.2f}, "
              f"BLEU1={qa['metrics']['bleu1']:.2f}")


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

    for cat_name in ["single-hop", "temporal", "multi-hop", "adversarial", "open-domain"]:
        if cat_name in stats["by_category"]:
            cat_stats = stats["by_category"][cat_name]
            categories.append(cat_name)
            for metric in ["exact_match", "f1", "bleu1", "contains_match", "numeric_match"]:
                metrics_data[metric].append(cat_stats.get(metric, 0))

    # Sample performance
    samples = []
    sample_metrics = defaultdict(list)
    for sample_id, sample_stats in sorted(stats["by_sample"].items()):
        samples.append(sample_id)
        for metric in ["exact_match", "f1", "bleu1"]:
            sample_metrics[metric].append(sample_stats.get(metric, 0))

    # Score distribution (histogram data)
    f1_scores = [qa["metrics"]["f1"] for qa in stats["detailed"]]
    em_scores = [qa["metrics"]["exact_match"] for qa in stats["detailed"]]
    bleu_scores = [qa["metrics"]["bleu1"] for qa in stats["detailed"]]

    def compute_histogram(scores: List[float], bins: int = 10) -> Dict[str, Any]:
        """Compute histogram data for scores."""
        buckets = defaultdict(int)
        for score in scores:
            bucket = min(int(score * bins) / bins, (bins - 1) / bins)
            buckets[bucket] += 1

        sorted_buckets = sorted(buckets.keys())
        return {
            "buckets": [f"{b:.1f}-{b + 1/bins:.1f}" for b in sorted_buckets],
            "counts": [buckets[b] for b in sorted_buckets]
        }

    # Error analysis data
    error_by_category = {}
    for cat_name, cat_stats in stats["by_category"].items():
        cat_items = [qa for qa in stats["detailed"] if qa["category_name"] == cat_name]
        errors = [qa for qa in cat_items if qa["metrics"]["f1"] < 0.3]
        error_by_category[cat_name] = {
            "total": len(cat_items),
            "errors": len(errors),
            "error_rate": len(errors) / len(cat_items) if cat_items else 0
        }

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
            "datasets": [
                {"label": metric, "data": values}
                for metric, values in sample_metrics.items()
            ]
        },
        "histograms": {
            "f1": compute_histogram(f1_scores),
            "exact_match": compute_histogram(em_scores),
            "bleu1": compute_histogram(bleu_scores)
        },
        "error_analysis": error_by_category,
        "score_distributions": {
            "f1": {
                "mean": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
                "min": min(f1_scores) if f1_scores else 0,
                "max": max(f1_scores) if f1_scores else 0,
                "median": sorted(f1_scores)[len(f1_scores) // 2] if f1_scores else 0
            },
            "bleu1": {
                "mean": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
                "min": min(bleu_scores) if bleu_scores else 0,
                "max": max(bleu_scores) if bleu_scores else 0,
                "median": sorted(bleu_scores)[len(bleu_scores) // 2] if bleu_scores else 0
            }
        }
    }


def save_results_json(
    stats: Dict[str, Any],
    output_path: str,
    include_detailed: bool = True,
    include_visualization: bool = True
) -> None:
    """
    Save results to JSON file.

    Args:
        stats: Statistics dict
        output_path: Output file path
        include_detailed: Whether to include detailed QA results
        include_visualization: Whether to include visualization data
    """
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "overall": stats["overall"],
            "by_category": stats["by_category"],
            "by_sample": stats["by_sample"]
        }
    }

    if include_detailed:
        output["detailed"] = stats["detailed"]

    if include_visualization:
        output["visualization"] = generate_visualization_data(stats)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")


def save_results_csv(stats: Dict[str, Any], output_path: str) -> None:
    """
    Save summary results to CSV file.

    Args:
        stats: Statistics dict
        output_path: Output file path
    """
    import csv

    rows = []

    # Header
    header = ["Category", "Count", "Exact_Match", "F1", "BLEU-1",
              "Precision", "Recall", "Contains_Match", "Word_Overlap", "Numeric_Match"]

    # Category rows
    for cat_name, cat_stats in stats["by_category"].items():
        rows.append([
            cat_name,
            cat_stats["count"],
            f"{cat_stats['exact_match']:.4f}",
            f"{cat_stats['f1']:.4f}",
            f"{cat_stats['bleu1']:.4f}",
            f"{cat_stats['precision']:.4f}",
            f"{cat_stats['recall']:.4f}",
            f"{cat_stats['contains_match']:.4f}",
            f"{cat_stats['word_overlap']:.4f}",
            f"{cat_stats['numeric_match']:.4f}"
        ])

    # Overall row
    overall = stats["overall"]
    rows.append([
        "Overall",
        overall["count"],
        f"{overall['exact_match']:.4f}",
        f"{overall['f1']:.4f}",
        f"{overall['bleu1']:.4f}",
        f"{overall['precision']:.4f}",
        f"{overall['recall']:.4f}",
        f"{overall['contains_match']:.4f}",
        f"{overall['word_overlap']:.4f}",
        f"{overall['numeric_match']:.4f}"
    ])

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nCSV results saved to: {output_path}")


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_experiment(
    results_dir: str = None,
    experiment_name: str = "locomo_experiment",
    output_path: str = None,
    output_format: str = "json",
    show_errors: bool = True,
    error_threshold: float = 0.3,
    auto_save: bool = True,
    include_visualization: bool = True
) -> Dict[str, Any]:
    """
    Main analysis function.

    Args:
        results_dir: Base results directory
        experiment_name: Name of experiment to analyze
        output_path: Output file path (optional, auto-generated if auto_save=True)
        output_format: Output format ("json" or "csv")
        show_errors: Whether to print low-scoring items
        error_threshold: F1 threshold for errors
        auto_save: Whether to automatically save results
        include_visualization: Whether to include visualization data in output

    Returns:
        Statistics dict
    """
    # Default paths
    if results_dir is None:
        results_dir = config.RESULTS_DIR

    experiments_dir = os.path.join(results_dir, "experiments")

    print(f"Loading results from: {experiments_dir}/{experiment_name}")

    # Load results
    results = load_experiment_results(experiments_dir, experiment_name)

    if not results:
        print("No results found!")
        return {}

    print(f"Loaded {len(results)} sample(s)")

    # Extract QA pairs
    qa_pairs = load_all_qa_pairs(results)
    print(f"Total QA pairs: {len(qa_pairs)}")

    # Compute statistics
    stats = compute_statistics(qa_pairs)

    # Print summary
    print_summary_table(stats)

    # Print errors if requested
    if show_errors:
        print_detailed_errors(stats, threshold=error_threshold)

    # Determine output path
    if output_path is None and auto_save:
        # Auto-generate output path in experiment directory
        exp_dir = os.path.join(experiments_dir, experiment_name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(exp_dir, f"analysis_report_{timestamp}.json")

    # Save results if output path specified or auto_save enabled
    if output_path:
        # Get base path for txt report
        if output_path.endswith('.json'):
            txt_path = output_path.replace('.json', '.txt')
        elif output_path.endswith('.csv'):
            txt_path = output_path.replace('.csv', '.txt')
        else:
            txt_path = output_path + '.txt'

        # Save text report
        save_text_report(stats, txt_path, experiment_name, error_threshold)

        # Save JSON/CSV
        if output_format == "csv":
            save_results_csv(stats, output_path)
            # Also save JSON with visualization if requested
            if include_visualization:
                json_path = output_path.replace('.csv', '_visualization.json')
                save_results_json(stats, json_path, include_detailed=False, include_visualization=True)
        else:
            save_results_json(stats, output_path, include_detailed=True, include_visualization=include_visualization)

    return stats


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze ProCoMemory experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_results.py
  python analyze_results.py --experiment my_experiment
  python analyze_results.py --output ./analysis_report.json
  python analyze_results.py --format csv --output ./results.csv
  python analyze_results.py --no-errors
        """
    )

    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default=None,
        help="Base results directory (default: config.RESULTS_DIR)"
    )

    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default="locomo_experiment",
        help="Experiment name to analyze (default: locomo_experiment)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path for results"
    )

    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)"
    )

    parser.add_argument(
        "--no-errors",
        action="store_true",
        help="Don't show low-scoring items"
    )

    parser.add_argument(
        "--error-threshold",
        type=float,
        default=0.3,
        help="F1 threshold for 'error' items (default: 0.3)"
    )

    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="Include detailed QA results in output"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't auto-save results to file"
    )

    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Don't include visualization data in output"
    )

    args = parser.parse_args()

    # Generate default output path if not specified but format is given
    if args.output is None and args.save_detailed:
        args.output = f"./analysis_{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.format}"

    # Run analysis
    stats = analyze_experiment(
        results_dir=args.results_dir,
        experiment_name=args.experiment,
        output_path=args.output,
        output_format=args.format,
        show_errors=not args.no_errors,
        error_threshold=args.error_threshold,
        auto_save=not args.no_save,
        include_visualization=not args.no_visualization
    )

    return stats


if __name__ == "__main__":
    main()
