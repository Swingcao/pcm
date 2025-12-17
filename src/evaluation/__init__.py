"""
Evaluation module for PCM system.
Contains LoComo benchmark dataset loader, metrics, and experiment runner.
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
    "LoCoMoExperiment"
]
