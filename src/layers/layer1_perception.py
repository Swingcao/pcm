"""
Layer 1: Perception & Working Memory
=====================================
Handles real-time interaction, intent routing, and surprisal monitoring.

Components:
1. SlidingContextQueue - Working memory with eviction
2. IntentRouter - Intent classification
3. SurpriseMonitor - Surprisal calculation and effective surprisal
"""

import os
import json
from collections import deque
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.core.types import (
    ConversationTurn, EvictedData, Intent, SurprisalPacket, MemoryNode
)
from src.utils.llm_client import LLMClient, get_llm_client
from src.utils.metrics import get_surprisal_calculator, SurprisalCalculator
from src.utils.math_utils import compute_retrieval_entropy, softmax


class SlidingContextQueue:
    """
    Working Memory: Sliding context queue with fixed token window.

    Maintains recent conversation history and evicts oldest turns
    when the queue exceeds the token limit.

    Q_t = [u_{t-k}, r_{t-k}, ..., u_t]
    """

    def __init__(
        self,
        max_tokens: int = None,
        eviction_size: int = None
    ):
        """
        Initialize the sliding context queue.

        Args:
            max_tokens: Maximum tokens in the window
            eviction_size: Number of turns to evict when full
        """
        self.max_tokens = max_tokens or config.MAX_CONTEXT_TOKENS
        self.eviction_size = eviction_size or config.EVICTION_SIZE
        self._queue: deque[ConversationTurn] = deque()
        self._total_tokens = 0

    def add(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[EvictedData]:
        """
        Add a new turn to the queue.

        If the queue exceeds max_tokens, evict oldest turns.

        Args:
            role: 'user' or 'assistant'
            content: Message content
            timestamp: Optional timestamp

        Returns:
            EvictedData if eviction occurred, None otherwise
        """
        # Estimate token count (rough: ~4 chars per token)
        token_count = len(content) // 4 + 1

        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=timestamp or datetime.now(),
            token_count=token_count
        )

        self._queue.append(turn)
        self._total_tokens += token_count

        # Check if eviction is needed
        evicted = None
        if self._total_tokens > self.max_tokens:
            evicted = self._evict()

        return evicted

    def _evict(self) -> EvictedData:
        """
        Evict oldest turns from the queue.

        Returns:
            EvictedData containing evicted turns
        """
        evicted_turns = []
        evicted_count = 0

        while (
            self._total_tokens > self.max_tokens and
            len(self._queue) > 0 and
            evicted_count < self.eviction_size
        ):
            turn = self._queue.popleft()
            self._total_tokens -= turn.token_count
            evicted_turns.append(turn)
            evicted_count += 1

        # Create EvictedData packet
        raw_text = "\n".join([f"{t.role}: {t.content}" for t in evicted_turns])

        return EvictedData(
            turns=evicted_turns,
            raw_text=raw_text,
            evicted_at=datetime.now()
        )

    def get_context(self, max_turns: Optional[int] = None) -> str:
        """
        Get the current context as a formatted string.

        Args:
            max_turns: Optional limit on number of turns

        Returns:
            Formatted context string
        """
        turns = list(self._queue)
        if max_turns:
            turns = turns[-max_turns:]

        return "\n".join([f"{t.role}: {t.content}" for t in turns])

    def get_turns(self) -> List[ConversationTurn]:
        """Get all turns as a list."""
        return list(self._queue)

    @property
    def total_tokens(self) -> int:
        """Current total token count."""
        return self._total_tokens

    @property
    def num_turns(self) -> int:
        """Current number of turns."""
        return len(self._queue)

    def clear(self) -> EvictedData:
        """Clear all turns and return them as evicted data."""
        all_turns = list(self._queue)
        raw_text = "\n".join([f"{t.role}: {t.content}" for t in all_turns])

        self._queue.clear()
        self._total_tokens = 0

        return EvictedData(
            turns=all_turns,
            raw_text=raw_text,
            evicted_at=datetime.now()
        )


class IntentClassification(BaseModel):
    """Schema for intent classification LLM response."""
    label: str
    confidence: float
    distribution: Dict[str, float]


class IntentRouter:
    """
    Intent Router: Classifies user query into intent domains.

    P(I_t | u_t, Q_t) = Softmax(f_θ(u_t, Q_t))

    Used to generate intent mask for L2 retrieval.
    """

    INTENT_PROMPT = """Analyze the user query and classify it into one of these domains:
{domains}

Consider the conversation context when classifying.

Conversation context:
{context}

Current user query: {query}

Respond with JSON in this format:
{{
    "label": "primary domain",
    "confidence": 0.0-1.0,
    "distribution": {{"domain1": 0.x, "domain2": 0.x, ...}}
}}

The distribution should sum to 1.0 and include all domains."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        domains: Optional[List[str]] = None
    ):
        """
        Initialize the intent router.

        Args:
            llm_client: LLM client for classification
            domains: List of intent domains
        """
        self.llm_client = llm_client or get_llm_client()
        self.domains = domains or config.INTENT_DOMAINS

    async def classify(
        self,
        query: str,
        context: str = ""
    ) -> Intent:
        """
        Classify the intent of a user query.

        Args:
            query: User query
            context: Conversation context

        Returns:
            Intent classification result
        """
        prompt = self.INTENT_PROMPT.format(
            domains=", ".join(self.domains),
            context=context if context else "(No prior context)",
            query=query
        )

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=config.ROUTER_LLM_MODEL,
                temperature=0.3,
                max_tokens=300
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

            return Intent(
                label=data.get("label", "General"),
                confidence=data.get("confidence", 0.5),
                distribution=data.get("distribution", {})
            )

        except Exception as e:
            print(f"Intent classification failed: {e}")
            # Return default intent
            default_dist = {d: 1.0 / len(self.domains) for d in self.domains}
            return Intent(
                label="General",
                confidence=0.5,
                distribution=default_dist
            )

    def classify_simple(self, query: str) -> Intent:
        """
        Simple rule-based classification (fallback/testing).

        Uses keyword matching for fast classification.
        """
        query_lower = query.lower()

        # Keyword patterns for each domain
        patterns = {
            "Coding": ["code", "program", "function", "debug", "error", "python", "javascript", "api"],
            "Academic": ["research", "paper", "study", "theory", "hypothesis", "data", "analysis"],
            "Personal": ["feel", "think", "prefer", "like", "want", "my", "i am", "i'm"],
            "Casual": ["hello", "hi", "thanks", "bye", "how are", "weather"],
            "Professional": ["meeting", "deadline", "project", "client", "business", "work"],
            "Creative": ["write", "story", "design", "art", "music", "creative"]
        }

        scores = {}
        for domain, keywords in patterns.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[domain] = score + 0.1  # Small base score

        # Softmax to get distribution
        total = sum(scores.values())
        distribution = {d: s / total for d, s in scores.items()}

        max_domain = max(distribution, key=distribution.get)

        return Intent(
            label=max_domain,
            confidence=distribution[max_domain],
            distribution=distribution
        )


class SurpriseMonitor:
    """
    Surprise Monitor: The Gatekeeper

    Computes surprisal scores to measure expectation violation using
    semantic similarity + LLM conflict detection.

    S_eff = α × semantic_distance + (1-α) × conflict_score
    """

    def __init__(self, use_mock: bool = False):
        """
        Initialize the surprise monitor.

        Args:
            use_mock: Use mock calculator for testing
        """
        self.calculator = get_surprisal_calculator(use_mock=use_mock)
        self.lambda_factor = config.SURPRISAL_LAMBDA
        self.theta_low = config.THETA_LOW
        self.theta_high = config.THETA_HIGH

    def calculate_surprisal_sync(
        self,
        user_input: str,
        retrieved_nodes: List[MemoryNode],
        retrieval_scores: List[float]
    ) -> SurprisalPacket:
        """
        Calculate surprisal synchronously (uses semantic distance only).

        For full analysis with LLM conflict detection, use calculate_surprisal().

        Args:
            user_input: User's query
            retrieved_nodes: Nodes retrieved from L2
            retrieval_scores: Corresponding retrieval scores

        Returns:
            SurprisalPacket with raw and effective surprisal
        """
        # Build context from retrieved nodes
        if retrieved_nodes:
            context = "\n".join([
                f"- {node.content} (confidence: {node.weight:.2f})"
                for node in retrieved_nodes
            ])
        else:
            context = ""

        # Calculate raw surprisal (semantic distance only)
        raw_score = self.calculator.calculate_sync(user_input, context)

        # Calculate retrieval entropy
        retrieval_entropy = compute_retrieval_entropy(retrieval_scores)

        # Apply entropy adjustment
        effective_score = raw_score * (1.0 - 0.3 * retrieval_entropy)

        return SurprisalPacket(
            content=user_input,
            raw_score=raw_score,
            effective_score=effective_score,
            retrieval_entropy=retrieval_entropy,
            retrieved_context=[n.content for n in retrieved_nodes],
            retrieved_node_ids=[n.id for n in retrieved_nodes],
            timestamp=datetime.now()
        )

    async def calculate_surprisal(
        self,
        user_input: str,
        retrieved_nodes: List[MemoryNode],
        retrieval_scores: List[float]
    ) -> SurprisalPacket:
        """
        Calculate surprisal asynchronously with full semantic analysis.

        Uses embedding similarity + LLM conflict detection.

        Args:
            user_input: User's query
            retrieved_nodes: Nodes retrieved from L2
            retrieval_scores: Corresponding retrieval scores

        Returns:
            SurprisalPacket with detailed surprisal analysis
        """
        # Extract memory info
        memory_contents = [node.content for node in retrieved_nodes]
        memory_ids = [node.id for node in retrieved_nodes]
        memory_embeddings = [node.embedding for node in retrieved_nodes if node.embedding is not None]

        if len(memory_embeddings) != len(retrieved_nodes):
            memory_embeddings = None  # Let calculator compute embeddings

        # Calculate semantic surprisal with full analysis
        result = await self.calculator.calculate_surprisal(
            user_input=user_input,
            memory_contents=memory_contents,
            memory_ids=memory_ids,
            memory_embeddings=memory_embeddings
        )

        # Calculate retrieval entropy
        retrieval_entropy = compute_retrieval_entropy(retrieval_scores)

        # Apply entropy adjustment
        raw_score = result["raw_score"]
        effective_score = raw_score * (1.0 - 0.3 * retrieval_entropy)

        packet = SurprisalPacket(
            content=user_input,
            raw_score=raw_score,
            effective_score=effective_score,
            retrieval_entropy=retrieval_entropy,
            retrieved_context=memory_contents,
            retrieved_node_ids=memory_ids,
            timestamp=datetime.now()
        )

        # Add semantic analysis metadata
        packet.semantic_analysis = {
            "semantic_distance": result["semantic_distance"],
            "conflict_score": result["conflict_score"],
            "relation": result["relation"],
            "reasoning": result["reasoning"],
            "conflicting_ids": result["conflicting_ids"]
        }

        return packet

    def get_surprise_level(self, packet: SurprisalPacket) -> str:
        """
        Determine surprise level for routing to L3 agents.

        Returns:
            "low", "medium", or "high"
        """
        return packet.get_surprise_level(self.theta_low, self.theta_high)


class PerceptionLayer:
    """
    Combined Layer 1 interface.

    Integrates SlidingContextQueue, IntentRouter, and SurpriseMonitor.
    """

    def __init__(
        self,
        max_tokens: int = None,
        eviction_size: int = None,
        use_mock: bool = False
    ):
        """
        Initialize the perception layer.

        Args:
            max_tokens: Maximum context window tokens
            eviction_size: Turns to evict when full
            use_mock: Use mock components for testing
        """
        self.context_queue = SlidingContextQueue(max_tokens, eviction_size)
        self.intent_router = IntentRouter()
        self.surprise_monitor = SurpriseMonitor(use_mock=use_mock)

    async def process_input(
        self,
        user_input: str,
        retrieved_nodes: List[MemoryNode] = None,
        retrieval_scores: List[float] = None
    ) -> Tuple[Optional[EvictedData], Intent, SurprisalPacket]:
        """
        Process a user input through the perception layer.

        1. Add to context queue (may evict)
        2. Classify intent
        3. Calculate surprisal with full semantic analysis

        Args:
            user_input: User's message
            retrieved_nodes: Nodes retrieved from L2
            retrieval_scores: Retrieval scores

        Returns:
            Tuple of (evicted_data, intent, surprisal_packet)
        """
        retrieved_nodes = retrieved_nodes or []
        retrieval_scores = retrieval_scores or []

        # Get current context for intent classification
        context = self.context_queue.get_context(max_turns=5)

        # Add input to queue (may trigger eviction)
        evicted = self.context_queue.add("user", user_input)

        # Classify intent
        intent = await self.intent_router.classify(user_input, context)

        # Calculate surprisal with full semantic analysis
        surprisal_packet = await self.surprise_monitor.calculate_surprisal(
            user_input,
            retrieved_nodes,
            retrieval_scores
        )
        surprisal_packet.intent = intent

        return evicted, intent, surprisal_packet

    def add_response(self, response: str) -> None:
        """Add assistant response to context queue."""
        self.context_queue.add("assistant", response)

    def get_context(self) -> str:
        """Get current conversation context."""
        return self.context_queue.get_context()

    def get_statistics(self) -> Dict[str, Any]:
        """Get layer statistics."""
        return {
            "total_tokens": self.context_queue.total_tokens,
            "num_turns": self.context_queue.num_turns,
            "theta_low": self.surprise_monitor.theta_low,
            "theta_high": self.surprise_monitor.theta_high,
            "alpha": self.surprise_monitor.calculator.alpha
        }
