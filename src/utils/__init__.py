"""
Utilities module for PCM system.
Contains helper functions and tools.
"""

from .llm_client import LLMClient, get_llm_client, chat
from .math_utils import (
    sigmoid,
    reinforcement_update,
    decay_update,
    time_decay,
    compute_retrieval_entropy,
    compute_effective_surprisal,
    compute_retrieval_score,
    cosine_similarity,
    softmax
)
from .metrics import SurprisalCalculator, get_surprisal_calculator

__all__ = [
    # LLM Client
    "LLMClient",
    "get_llm_client",
    "chat",
    # Math Utils
    "sigmoid",
    "reinforcement_update",
    "decay_update",
    "time_decay",
    "compute_retrieval_entropy",
    "compute_effective_surprisal",
    "compute_retrieval_score",
    "cosine_similarity",
    "softmax",
    # Metrics
    "SurprisalCalculator",
    "get_surprisal_calculator"
]
