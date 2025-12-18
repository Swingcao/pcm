"""
Statistics Module
=================
Comprehensive statistics calculation for experiment results.

This module provides:
1. Per-sample statistics calculation
2. Category-wise breakdown
3. Overall experiment aggregation
4. Failure analysis
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict


@dataclass
class CategoryStats:
    """Statistics for a specific question category."""
    category: str
    total: int = 0
    exact_match: float = 0.0
    contains_match: float = 0.0
    token_f1: float = 0.0
    bleu: float = 0.0
    no_info_count: int = 0  # Number of "no information" responses
    correct_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SampleStats:
    """Statistics for a single sample."""
    sample_id: str
    total_questions: int = 0

    # Overall metrics
    exact_match: float = 0.0
    contains_match: float = 0.0
    token_f1: float = 0.0
    bleu: float = 0.0

    # Category breakdown
    category_stats: Dict[str, CategoryStats] = field(default_factory=dict)

    # Failure analysis
    no_info_count: int = 0
    wrong_info_count: int = 0
    format_diff_count: int = 0

    # Retrieval stats
    avg_kg_retrieved: float = 0.0
    avg_raw_retrieved: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['category_stats'] = {k: v.to_dict() for k, v in self.category_stats.items()}
        return result


@dataclass
class OverallStats:
    """Overall statistics across all samples."""
    total_samples: int = 0
    total_questions: int = 0

    # Aggregate metrics
    exact_match: float = 0.0
    contains_match: float = 0.0
    token_f1: float = 0.0
    bleu: float = 0.0

    # Category breakdown
    category_stats: Dict[str, CategoryStats] = field(default_factory=dict)

    # Failure analysis
    no_info_rate: float = 0.0
    wrong_info_rate: float = 0.0
    format_diff_rate: float = 0.0

    # Per-sample stats
    sample_stats: List[SampleStats] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['category_stats'] = {k: v.to_dict() for k, v in self.category_stats.items()}
        result['sample_stats'] = [s.to_dict() for s in self.sample_stats]
        return result


class ExperimentStatistics:
    """
    Comprehensive statistics calculator for experiment results.

    Analyzes results from PCM experiments and generates detailed statistics
    including per-sample metrics, category breakdowns, and failure analysis.
    """

    def __init__(self):
        self.sample_stats: List[SampleStats] = []
        self.overall_stats: Optional[OverallStats] = None

    def analyze_sample(self, sample_results: Dict[str, Any]) -> SampleStats:
        """
        Analyze results for a single sample.

        Args:
            sample_results: Results dict from a single sample run

        Returns:
            SampleStats object with computed statistics
        """
        sample_id = sample_results.get('sample_id', 'unknown')
        qa_results = sample_results.get('qa_results', [])

        stats = SampleStats(
            sample_id=sample_id,
            total_questions=len(qa_results)
        )

        if not qa_results:
            return stats

        # Initialize category stats
        category_metrics = defaultdict(lambda: {
            'exact_match': [], 'contains_match': [],
            'token_f1': [], 'bleu': [],
            'no_info': 0, 'total': 0
        })

        # Collect metrics
        all_metrics = {
            'exact_match': [],
            'contains_match': [],
            'token_f1': [],
            'bleu': []
        }

        kg_counts = []
        raw_counts = []

        for qa in qa_results:
            metrics = qa.get('metrics', {})
            category = qa.get('category_name', 'unknown')
            prediction = qa.get('predicted_answer', '').lower()

            # Overall metrics
            all_metrics['exact_match'].append(metrics.get('exact_match', 0))
            all_metrics['contains_match'].append(metrics.get('contains_match', 0))
            all_metrics['token_f1'].append(metrics.get('token_f1', 0))
            all_metrics['bleu'].append(metrics.get('bleu', 0))

            # Category metrics
            category_metrics[category]['exact_match'].append(metrics.get('exact_match', 0))
            category_metrics[category]['contains_match'].append(metrics.get('contains_match', 0))
            category_metrics[category]['token_f1'].append(metrics.get('token_f1', 0))
            category_metrics[category]['bleu'].append(metrics.get('bleu', 0))
            category_metrics[category]['total'] += 1

            # Failure analysis
            is_no_info = any(phrase in prediction for phrase in [
                'does not provide', 'not specified', 'no information',
                'not mentioned', 'cannot find', 'no details'
            ])

            if is_no_info:
                stats.no_info_count += 1
                category_metrics[category]['no_info'] += 1
            elif metrics.get('exact_match', 0) == 0:
                if metrics.get('contains_match', 0) == 1 or metrics.get('token_f1', 0) > 0.5:
                    stats.format_diff_count += 1
                else:
                    stats.wrong_info_count += 1

            # Retrieval stats
            retrieval_stats = qa.get('retrieval_stats', {})
            if retrieval_stats:
                kg_counts.append(retrieval_stats.get('kg_count', 0))
                raw_counts.append(retrieval_stats.get('raw_count', 0))

        # Compute averages
        stats.exact_match = np.mean(all_metrics['exact_match'])
        stats.contains_match = np.mean(all_metrics['contains_match'])
        stats.token_f1 = np.mean(all_metrics['token_f1'])
        stats.bleu = np.mean(all_metrics['bleu'])

        if kg_counts:
            stats.avg_kg_retrieved = np.mean(kg_counts)
        if raw_counts:
            stats.avg_raw_retrieved = np.mean(raw_counts)

        # Category stats
        for category, cat_metrics in category_metrics.items():
            if cat_metrics['total'] > 0:
                cat_stats = CategoryStats(
                    category=category,
                    total=cat_metrics['total'],
                    exact_match=np.mean(cat_metrics['exact_match']),
                    contains_match=np.mean(cat_metrics['contains_match']),
                    token_f1=np.mean(cat_metrics['token_f1']),
                    bleu=np.mean(cat_metrics['bleu']),
                    no_info_count=cat_metrics['no_info'],
                    correct_count=sum(1 for em in cat_metrics['exact_match'] if em == 1)
                )
                stats.category_stats[category] = cat_stats

        self.sample_stats.append(stats)
        return stats

    def compute_overall_stats(self) -> OverallStats:
        """
        Compute overall statistics across all analyzed samples.

        Returns:
            OverallStats object with aggregated statistics
        """
        if not self.sample_stats:
            return OverallStats()

        overall = OverallStats(
            total_samples=len(self.sample_stats),
            total_questions=sum(s.total_questions for s in self.sample_stats),
            sample_stats=self.sample_stats
        )

        # Weighted average by number of questions
        total_q = overall.total_questions
        if total_q > 0:
            overall.exact_match = sum(
                s.exact_match * s.total_questions for s in self.sample_stats
            ) / total_q
            overall.contains_match = sum(
                s.contains_match * s.total_questions for s in self.sample_stats
            ) / total_q
            overall.token_f1 = sum(
                s.token_f1 * s.total_questions for s in self.sample_stats
            ) / total_q
            overall.bleu = sum(
                s.bleu * s.total_questions for s in self.sample_stats
            ) / total_q

            # Failure rates
            total_no_info = sum(s.no_info_count for s in self.sample_stats)
            total_wrong = sum(s.wrong_info_count for s in self.sample_stats)
            total_format = sum(s.format_diff_count for s in self.sample_stats)

            overall.no_info_rate = total_no_info / total_q
            overall.wrong_info_rate = total_wrong / total_q
            overall.format_diff_rate = total_format / total_q

        # Aggregate category stats
        category_totals = defaultdict(lambda: {
            'exact_match': [], 'contains_match': [],
            'token_f1': [], 'bleu': [],
            'total': 0, 'no_info': 0, 'correct': 0
        })

        for sample in self.sample_stats:
            for cat_name, cat_stats in sample.category_stats.items():
                ct = category_totals[cat_name]
                ct['exact_match'].extend([cat_stats.exact_match] * cat_stats.total)
                ct['contains_match'].extend([cat_stats.contains_match] * cat_stats.total)
                ct['token_f1'].extend([cat_stats.token_f1] * cat_stats.total)
                ct['bleu'].extend([cat_stats.bleu] * cat_stats.total)
                ct['total'] += cat_stats.total
                ct['no_info'] += cat_stats.no_info_count
                ct['correct'] += cat_stats.correct_count

        for cat_name, ct in category_totals.items():
            if ct['total'] > 0:
                overall.category_stats[cat_name] = CategoryStats(
                    category=cat_name,
                    total=ct['total'],
                    exact_match=np.mean(ct['exact_match']) if ct['exact_match'] else 0,
                    contains_match=np.mean(ct['contains_match']) if ct['contains_match'] else 0,
                    token_f1=np.mean(ct['token_f1']) if ct['token_f1'] else 0,
                    bleu=np.mean(ct['bleu']) if ct['bleu'] else 0,
                    no_info_count=ct['no_info'],
                    correct_count=ct['correct']
                )

        self.overall_stats = overall
        return overall

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a human-readable statistics report.

        Args:
            output_path: Optional path to save the report

        Returns:
            Report string
        """
        if self.overall_stats is None:
            self.compute_overall_stats()

        stats = self.overall_stats
        lines = []

        lines.append("=" * 70)
        lines.append("EXPERIMENT STATISTICS REPORT")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("=" * 70)

        lines.append("\n## OVERALL METRICS")
        lines.append(f"Total Samples: {stats.total_samples}")
        lines.append(f"Total Questions: {stats.total_questions}")
        lines.append(f"")
        lines.append(f"Exact Match:    {stats.exact_match:.4f} ({stats.exact_match*100:.2f}%)")
        lines.append(f"Contains Match: {stats.contains_match:.4f} ({stats.contains_match*100:.2f}%)")
        lines.append(f"Token F1:       {stats.token_f1:.4f} ({stats.token_f1*100:.2f}%)")
        lines.append(f"BLEU:           {stats.bleu:.4f} ({stats.bleu*100:.2f}%)")

        lines.append("\n## FAILURE ANALYSIS")
        lines.append(f"No Information Rate: {stats.no_info_rate:.4f} ({stats.no_info_rate*100:.2f}%)")
        lines.append(f"Wrong Info Rate:     {stats.wrong_info_rate:.4f} ({stats.wrong_info_rate*100:.2f}%)")
        lines.append(f"Format Diff Rate:    {stats.format_diff_rate:.4f} ({stats.format_diff_rate*100:.2f}%)")

        lines.append("\n## CATEGORY BREAKDOWN")
        lines.append("-" * 70)
        lines.append(f"{'Category':<15} {'Total':>8} {'EM':>10} {'CM':>10} {'F1':>10} {'NoInfo':>10}")
        lines.append("-" * 70)

        for cat_name, cat_stats in sorted(stats.category_stats.items()):
            no_info_pct = cat_stats.no_info_count / cat_stats.total * 100 if cat_stats.total > 0 else 0
            lines.append(
                f"{cat_name:<15} {cat_stats.total:>8} "
                f"{cat_stats.exact_match*100:>9.2f}% "
                f"{cat_stats.contains_match*100:>9.2f}% "
                f"{cat_stats.token_f1*100:>9.2f}% "
                f"{no_info_pct:>9.2f}%"
            )

        lines.append("\n## PER-SAMPLE SUMMARY")
        lines.append("-" * 70)
        lines.append(f"{'Sample':<15} {'Questions':>10} {'EM':>10} {'CM':>10} {'F1':>10}")
        lines.append("-" * 70)

        for sample in stats.sample_stats:
            lines.append(
                f"{sample.sample_id[:15]:<15} {sample.total_questions:>10} "
                f"{sample.exact_match*100:>9.2f}% "
                f"{sample.contains_match*100:>9.2f}% "
                f"{sample.token_f1*100:>9.2f}%"
            )

        lines.append("=" * 70)

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

        return report

    def save_to_json(self, output_path: str) -> None:
        """
        Save statistics to JSON file.

        Args:
            output_path: Path to save JSON file
        """
        if self.overall_stats is None:
            self.compute_overall_stats()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.overall_stats.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_results_directory(cls, results_dir: str) -> 'ExperimentStatistics':
        """
        Create statistics from a results directory.

        Args:
            results_dir: Path to experiment results directory

        Returns:
            ExperimentStatistics instance with analyzed results
        """
        stats = cls()

        # Find all sample result files
        for item in os.listdir(results_dir):
            sample_dir = os.path.join(results_dir, item)
            if os.path.isdir(sample_dir) and item.startswith('sample_'):
                results_file = os.path.join(sample_dir, 'results.json')
                if os.path.exists(results_file):
                    with open(results_file, 'r', encoding='utf-8') as f:
                        sample_results = json.load(f)
                    stats.analyze_sample(sample_results)

        stats.compute_overall_stats()
        return stats


def analyze_experiment_results(experiment_dir: str, save_report: bool = True) -> Dict[str, Any]:
    """
    Convenience function to analyze experiment results and generate reports.

    Args:
        experiment_dir: Path to experiment directory
        save_report: Whether to save the report to files

    Returns:
        Overall statistics dictionary
    """
    stats = ExperimentStatistics.from_results_directory(experiment_dir)

    if save_report:
        # Save JSON statistics
        json_path = os.path.join(experiment_dir, 'statistics.json')
        stats.save_to_json(json_path)

        # Save text report
        report_path = os.path.join(experiment_dir, 'statistics_report.txt')
        report = stats.generate_report(report_path)

        print(report)
        print(f"\nStatistics saved to:")
        print(f"  - {json_path}")
        print(f"  - {report_path}")

    return stats.overall_stats.to_dict() if stats.overall_stats else {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze experiment statistics")
    parser.add_argument("--dir", type=str, required=True, help="Experiment results directory")
    args = parser.parse_args()

    analyze_experiment_results(args.dir)
