"""
Evaluation module for PCM system.
Contains LoComo benchmark dataset loader, metrics, experiment runner, and analysis tools.
"""

from .dataset import (
    LoCoMoDataset,
    LoCoMoSample,
    QAPair,
    DialogueTurn,
    ConversationSession,
    load_locomo,
    process_dialogues_for_memory
)

from .metrics import (
    LoCoMoEvaluator,
    exact_match,
    contains_match,
    token_f1,
    word_overlap_score,
    numeric_match,
    normalize_answer,
    compute_accuracy
)

from .run_experiment import LoCoMoExperiment

from .analyze_results import (
    analyze_experiment,
    compute_f1,
    compute_bleu1,
    compute_statistics,
    load_experiment_results,
    load_all_qa_pairs,
    generate_visualization_data,
    generate_text_report,
    save_results_json,
    save_text_report
)

from .advanced_analysis import (
    compare_experiments,
    analyze_surprisal_distribution,
    analyze_errors,
    load_all_intermediate
)

__all__ = [
    # Dataset
    "LoCoMoDataset",
    "LoCoMoSample",
    "QAPair",
    "DialogueTurn",
    "ConversationSession",
    "load_locomo",
    "process_dialogues_for_memory",
    # Metrics
    "LoCoMoEvaluator",
    "exact_match",
    "contains_match",
    "token_f1",
    "word_overlap_score",
    "numeric_match",
    "normalize_answer",
    "compute_accuracy",
    # Experiment
    "LoCoMoExperiment",
    # Analysis
    "analyze_experiment",
    "compute_f1",
    "compute_bleu1",
    "compute_statistics",
    "load_experiment_results",
    "load_all_qa_pairs",
    "generate_visualization_data",
    "generate_text_report",
    "save_results_json",
    "save_text_report",
    # Advanced Analysis
    "compare_experiments",
    "analyze_surprisal_distribution",
    "analyze_errors",
    "load_all_intermediate"
]
