"""
Surprisal Metrics Calculator
============================
Computes surprisal scores using semantic similarity + LLM conflict detection.

Formula: S_eff = α × semantic_distance + (1-α) × conflict_score
"""

import math
import json
from typing import Optional, List, Dict, Any
import numpy as np
from pydantic import BaseModel

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config


class ConflictAnalysisResult(BaseModel):
    """Schema for LLM conflict analysis response."""
    relation: str  # "consistent", "novel", "contradictory"
    conflict_score: float  # 0.0-1.0
    reasoning: str
    conflicting_memory_ids: List[str] = []


class SurprisalCalculator:
    """
    Semantic Surprisal Calculator using embedding similarity + LLM conflict detection.

    Combines:
    1. Embedding-based semantic distance (fast, local)
    2. LLM semantic conflict analysis (accurate, interpretable)

    S_eff = α × semantic_distance + (1-α) × conflict_score

    Advantages:
    - Accurate conflict detection using modern LLM
    - Better handling of paraphrased/semantically similar content
    - Interpretable reasoning for why something is surprising
    """

    CONFLICT_ANALYSIS_PROMPT = """You are analyzing whether a user's statement conflicts with, is consistent with, or adds new information to existing memories.

## User Statement
{user_input}

## Retrieved Memories
{memories}

## Task
Analyze the semantic relationship between the user's statement and the retrieved memories.

## Classification
- "consistent": The user's statement agrees with or is supported by the memories
- "novel": The user's statement contains new information not covered by memories (neither confirms nor contradicts)
- "contradictory": The user's statement conflicts with one or more memories

## Response Format (JSON)
{{
    "relation": "consistent" | "novel" | "contradictory",
    "conflict_score": 0.0-1.0,
    "reasoning": "Brief explanation of why this classification was made",
    "conflicting_memory_ids": ["id1", "id2"] // Only for contradictory cases
}}

## Scoring Guidelines
- consistent: conflict_score = 0.0 - 0.2
- novel: conflict_score = 0.3 - 0.6
- contradictory: conflict_score = 0.7 - 1.0

Respond with valid JSON only."""

    def __init__(
        self,
        llm_client=None,
        embedding_model=None,
        alpha: float = None,
        use_mock: bool = False
    ):
        """
        Initialize the surprisal calculator.

        Args:
            llm_client: LLM client for conflict analysis
            embedding_model: Embedding model for semantic similarity
            alpha: Weight balance (0=pure LLM, 1=pure embedding). Default from config.
            use_mock: Use mock mode for testing
        """
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.alpha = alpha if alpha is not None else config.SURPRISAL_ALPHA
        self.use_mock = use_mock or config.USE_MOCK_LLM

    def _ensure_llm_client(self):
        """Lazy load LLM client if not provided."""
        if self.llm_client is None and not self.use_mock:
            from src.utils.llm_client import get_llm_client
            self.llm_client = get_llm_client()

    def _ensure_embedding_model(self):
        """Lazy load embedding model if not provided."""
        if self.embedding_model is None and not self.use_mock:
            from src.layers.layer2_world_model import get_embedding_model
            self.embedding_model = get_embedding_model()

    def compute_semantic_distance(
        self,
        user_input: str,
        memory_contents: List[str],
        memory_embeddings: Optional[List[np.ndarray]] = None
    ) -> float:
        """
        Compute semantic distance between user input and memories.

        Returns a value in [0, 1] where:
        - 0 = highly similar to memories
        - 1 = very different from memories
        """
        if not memory_contents:
            return 1.0  # Maximum distance when no memories

        if self.use_mock:
            # Mock: use simple word overlap
            input_words = set(user_input.lower().split())
            max_overlap = 0
            for content in memory_contents:
                memory_words = set(content.lower().split())
                overlap = len(input_words & memory_words) / max(len(input_words), 1)
                max_overlap = max(max_overlap, overlap)
            return 1.0 - max_overlap

        self._ensure_embedding_model()

        # Get user input embedding
        user_embedding = self.embedding_model.encode(user_input)

        # Get memory embeddings if not provided
        if memory_embeddings is None:
            memory_embeddings = [
                self.embedding_model.encode(content)
                for content in memory_contents
            ]

        # Compute max cosine similarity
        max_similarity = 0.0
        for mem_emb in memory_embeddings:
            similarity = self._cosine_similarity(user_embedding, mem_emb)
            max_similarity = max(max_similarity, similarity)

        # Convert similarity to distance
        return 1.0 - max_similarity

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        vec1 = np.array(vec1).flatten()
        vec2 = np.array(vec2).flatten()

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    async def analyze_conflict(
        self,
        user_input: str,
        memory_contents: List[str],
        memory_ids: List[str]
    ) -> ConflictAnalysisResult:
        """
        Use LLM to analyze semantic conflict between user input and memories.

        Returns ConflictAnalysisResult with relation, score, and reasoning.
        """
        if not memory_contents:
            return ConflictAnalysisResult(
                relation="novel",
                conflict_score=0.5,
                reasoning="No existing memories to compare against",
                conflicting_memory_ids=[]
            )

        if self.use_mock:
            return self._mock_conflict_analysis(user_input, memory_contents)

        self._ensure_llm_client()

        # Format memories for prompt
        memories_text = "\n".join([
            f"[{mid}] {content}"
            for mid, content in zip(memory_ids, memory_contents)
        ])

        prompt = self.CONFLICT_ANALYSIS_PROMPT.format(
            user_input=user_input,
            memories=memories_text
        )

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=config.MAIN_LLM_MODEL,
                temperature=0.2,
                max_tokens=500
            )

            content = response["content"]

            # Parse JSON response
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            data = json.loads(content)

            return ConflictAnalysisResult(
                relation=data.get("relation", "novel"),
                conflict_score=float(data.get("conflict_score", 0.5)),
                reasoning=data.get("reasoning", ""),
                conflicting_memory_ids=data.get("conflicting_memory_ids", [])
            )

        except Exception as e:
            print(f"Conflict analysis failed: {e}")
            # Fallback to semantic distance only
            return ConflictAnalysisResult(
                relation="novel",
                conflict_score=0.5,
                reasoning=f"Analysis failed: {str(e)}",
                conflicting_memory_ids=[]
            )

    def _mock_conflict_analysis(
        self,
        user_input: str,
        memory_contents: List[str]
    ) -> ConflictAnalysisResult:
        """Mock conflict analysis for testing."""
        negation_words = ["not", "no", "never", "don't", "doesn't", "isn't",
                         "aren't", "wasn't", "weren't", "won't", "wouldn't",
                         "can't", "couldn't", "shouldn't"]

        input_lower = user_input.lower()
        has_negation = any(word in input_lower for word in negation_words)

        # Check for potential contradiction keywords
        contradiction_keywords = ["actually", "but", "however", "instead",
                                  "wrong", "incorrect", "mistake", "change"]
        has_contradiction_hint = any(word in input_lower for word in contradiction_keywords)

        if has_contradiction_hint or has_negation:
            return ConflictAnalysisResult(
                relation="contradictory",
                conflict_score=0.8,
                reasoning="Mock: Detected potential contradiction keywords",
                conflicting_memory_ids=[]
            )

        # Check word overlap
        input_words = set(input_lower.split())
        max_overlap = 0
        for content in memory_contents:
            memory_words = set(content.lower().split())
            overlap = len(input_words & memory_words) / max(len(input_words), 1)
            max_overlap = max(max_overlap, overlap)

        if max_overlap > 0.5:
            return ConflictAnalysisResult(
                relation="consistent",
                conflict_score=0.1,
                reasoning="Mock: High word overlap with existing memories",
                conflicting_memory_ids=[]
            )
        else:
            return ConflictAnalysisResult(
                relation="novel",
                conflict_score=0.4,
                reasoning="Mock: New information not in existing memories",
                conflicting_memory_ids=[]
            )

    async def calculate_surprisal(
        self,
        user_input: str,
        memory_contents: List[str],
        memory_ids: List[str],
        memory_embeddings: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Calculate semantic surprisal using hybrid approach.

        S_eff = α × semantic_distance + (1-α) × conflict_score

        Args:
            user_input: User's input text
            memory_contents: Contents of retrieved memory nodes
            memory_ids: IDs of retrieved memory nodes
            memory_embeddings: Pre-computed embeddings (optional)

        Returns:
            Dict containing:
            - raw_score: Combined surprisal score (scaled to [0, 5])
            - semantic_distance: Embedding-based distance [0, 1]
            - conflict_score: LLM-based conflict score [0, 1]
            - relation: "consistent", "novel", or "contradictory"
            - reasoning: LLM explanation
            - conflicting_ids: List of conflicting memory IDs
        """
        # Step 1: Compute semantic distance (fast, local)
        semantic_distance = self.compute_semantic_distance(
            user_input, memory_contents, memory_embeddings
        )

        # Step 2: Analyze conflict with LLM (accurate, interpretable)
        conflict_result = await self.analyze_conflict(
            user_input, memory_contents, memory_ids
        )

        # Step 3: Combine scores
        # S_eff = α × semantic_distance + (1-α) × conflict_score
        combined_score = (
            self.alpha * semantic_distance +
            (1 - self.alpha) * conflict_result.conflict_score
        )

        # Scale to [0, 5] for threshold compatibility (theta_low=1.0, theta_high=3.0)
        scaled_score = combined_score * 5.0

        return {
            "raw_score": scaled_score,
            "semantic_distance": semantic_distance,
            "conflict_score": conflict_result.conflict_score,
            "relation": conflict_result.relation,
            "reasoning": conflict_result.reasoning,
            "conflicting_ids": conflict_result.conflicting_memory_ids
        }

    def calculate_sync(self, user_input: str, context: str = "") -> float:
        """
        Synchronous surprisal calculation (uses semantic distance only).

        For full analysis with LLM conflict detection, use calculate_surprisal().
        """
        if not context:
            return 2.5  # Default medium surprisal

        memory_contents = [line.strip("- ").split(" (confidence:")[0]
                          for line in context.split("\n") if line.strip()]

        semantic_distance = self.compute_semantic_distance(user_input, memory_contents)
        return semantic_distance * 5.0


# Singleton instance
_calculator: Optional[SurprisalCalculator] = None


def get_surprisal_calculator(use_mock: bool = False) -> SurprisalCalculator:
    """Get the surprisal calculator instance."""
    global _calculator

    if use_mock or config.USE_MOCK_LLM:
        return SurprisalCalculator(use_mock=True)

    if _calculator is None:
        _calculator = SurprisalCalculator()
    return _calculator
