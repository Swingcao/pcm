"""
Mathematical Utilities for PCM System
=====================================
Implementation of Bayesian update formulas, entropy calculations, and weight updates.
"""

import math
from typing import List, Dict, Optional
import numpy as np


def sigmoid(x: float) -> float:
    """
    Compute sigmoid function.

    σ(x) = 1 / (1 + e^(-x))
    """
    # Clip to avoid overflow
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + math.exp(-x))


def map_to_range(value: float, min_val: float, max_val: float) -> float:
    """
    Map a value in [0, 1] to [min_val, max_val].
    """
    return min_val + value * (max_val - min_val)


def initialize_hypothesis_weight(
    surprisal_score: float,
    weight_min: float = 0.3,
    weight_max: float = 0.5
) -> float:
    """
    Initialize hypothesis weight based on surprisal score.

    w_init = σ(S_eff) mapped to [weight_min, weight_max]

    Higher surprisal (within medium range) -> higher initial attention.
    """
    sig_val = sigmoid(surprisal_score)
    return map_to_range(sig_val, weight_min, weight_max)


def reinforcement_update(
    current_weight: float,
    learning_rate: float = 0.05
) -> float:
    """
    Maintenance Agent: Reinforcement weight update for low surprise.

    w_{t+1} = w_t + η * (1 - w_t)

    This makes weight asymptotically approach 1.0 but never exceed it.

    Args:
        current_weight: Current confidence weight w_t
        learning_rate: η (default 0.05)

    Returns:
        Updated weight w_{t+1}
    """
    new_weight = current_weight + learning_rate * (1.0 - current_weight)
    return min(1.0, new_weight)


def decay_update(
    current_weight: float,
    surprisal_score: float,
    decay_factor: float = 0.3
) -> float:
    """
    Correction Agent: Exponential decay for high surprise conflicts.

    w_{t+1} = w_t * exp(-β * S_eff)

    Higher surprisal -> stronger penalty.

    Args:
        current_weight: Current confidence weight w_t
        surprisal_score: Effective surprisal S_eff
        decay_factor: β (default 0.3)

    Returns:
        Updated weight w_{t+1} (decayed)
    """
    decay = math.exp(-decay_factor * surprisal_score)
    new_weight = current_weight * decay
    return max(0.0, min(1.0, new_weight))


def time_decay(
    current_weight: float,
    time_delta: float,
    decay_rate: float = 0.01
) -> float:
    """
    Time-based weight decay for hypothesis nodes without reinforcement.

    w_{t+k} = w_t * e^(-γ * Δt)

    Args:
        current_weight: Current weight
        time_delta: Time elapsed (in appropriate units)
        decay_rate: γ (default 0.01)

    Returns:
        Decayed weight
    """
    decay = math.exp(-decay_rate * time_delta)
    return current_weight * decay


def compute_retrieval_entropy(scores: List[float]) -> float:
    """
    Compute retrieval entropy H(C_t) to measure retrieval quality.

    H(C_t) = -Σ ŝ(ε) * log(ŝ(ε))

    where ŝ is the normalized retrieval score.

    High entropy indicates uncertain/noisy retrieval.
    Low entropy indicates confident/clear retrieval.

    Args:
        scores: List of retrieval scores (similarities)

    Returns:
        Entropy value H(C_t)
    """
    if not scores:
        return 1.0  # Maximum uncertainty when no context

    # Ensure all scores are positive
    scores = [max(s, 1e-10) for s in scores]

    # Normalize scores to form a probability distribution
    total = sum(scores)
    if total <= 0:
        return 1.0

    probs = [s / total for s in scores]

    # Compute entropy
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)

    # Normalize to [0, 1] by dividing by max entropy log(n)
    max_entropy = math.log(len(scores)) if len(scores) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return normalized_entropy


def compute_effective_surprisal(
    raw_surprisal: float,
    retrieval_entropy: float,
    lambda_factor: float = 0.5
) -> float:
    """
    Compute effective surprisal considering retrieval uncertainty.

    S_eff(u_t) = S_raw(u_t) * (1 - λ * H(C_t))

    If retrieval is uncertain (high entropy), we discount the surprisal
    because high NLL might indicate "ignorance" not "conflict".

    Args:
        raw_surprisal: S_raw from NLL calculation
        retrieval_entropy: H(C_t) from retrieval scores
        lambda_factor: λ weight for entropy (default 0.5)

    Returns:
        Effective surprisal S_eff
    """
    confidence_gate = 1.0 - lambda_factor * retrieval_entropy
    return raw_surprisal * max(0.0, confidence_gate)


def bayesian_weight_update(
    prior_weight: float,
    likelihood: float,
    marginal: float = 1.0
) -> float:
    """
    General Bayesian weight update.

    w_{t+1} = P(u_t | ε) * w_t / P(u_t)

    In our framework, likelihood is approximated based on surprisal.

    Args:
        prior_weight: w_t (prior belief)
        likelihood: P(u_t | ε)
        marginal: P(u_t) (normalization factor)

    Returns:
        Posterior weight
    """
    if marginal <= 0:
        return prior_weight

    posterior = (likelihood * prior_weight) / marginal
    return max(0.0, min(1.0, posterior))


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    sim(a, b) = a · b / (||a|| * ||b||)
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def compute_retrieval_score(
    semantic_similarity: float,
    intent_relevance: float,
    confidence_weight: float
) -> float:
    """
    Compute the combined retrieval score for an edge.

    Score(ε_k) = sim(emb(u_t), emb(ε_k)) * P(d(ε_k) | u_t) * w_k

    Args:
        semantic_similarity: Embedding cosine similarity
        intent_relevance: Intent probability P(domain | query)
        confidence_weight: Edge weight w_k

    Returns:
        Combined retrieval score
    """
    return semantic_similarity * intent_relevance * confidence_weight


def softmax(logits: List[float], temperature: float = 1.0) -> List[float]:
    """
    Compute softmax probabilities with optional temperature.

    P(i) = exp(logit_i / T) / Σ exp(logit_j / T)
    """
    if not logits:
        return []

    # Apply temperature and shift for numerical stability
    scaled = [l / temperature for l in logits]
    max_val = max(scaled)
    exp_vals = [math.exp(l - max_val) for l in scaled]
    total = sum(exp_vals)

    return [e / total for e in exp_vals]
