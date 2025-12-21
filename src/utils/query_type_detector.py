"""
Query Type Detector
===================
Detects query types to enable adaptive retrieval weight configuration.

This module addresses the issue where single-hop factual queries need higher
keyword weights, while temporal and multi-hop queries need different configurations.

Key Features:
- Rule-based query type detection (fast, no LLM calls)
- Supports: factual, temporal, multi-hop/inference, adversarial, default
- Returns appropriate RetrievalConfig for each query type
- Preserves performance gains on temporal and adversarial queries

Query Types:
- FACTUAL: Single-hop factual questions (What, Where, Who, Which)
- TEMPORAL: Time-related questions (When, How long, dates)
- MULTI_HOP: Inference/reasoning questions (Why, What would, likely)
- ADVERSARIAL: Questions with negation or contradiction patterns
- DEFAULT: General queries that don't match specific patterns
"""

import re
from enum import Enum
from typing import Optional, Tuple, List
from dataclasses import dataclass


class QueryType(Enum):
    """Types of queries for adaptive retrieval."""
    FACTUAL = "factual"           # Single-hop factual (high keyword weight)
    TEMPORAL = "temporal"          # Time-related (high recency weight)
    MULTI_HOP = "multi_hop"       # Inference/reasoning (high graph weight)
    ADVERSARIAL = "adversarial"   # Contradiction/negation patterns
    DEFAULT = "default"           # General queries


@dataclass
class QueryTypeResult:
    """Result of query type detection."""
    query_type: QueryType
    confidence: float  # 0.0 to 1.0
    matched_patterns: List[str]  # Which patterns matched

    def __str__(self) -> str:
        return f"QueryType({self.query_type.value}, conf={self.confidence:.2f})"


class QueryTypeDetector:
    """
    Detects query type for adaptive retrieval configuration.

    Uses rule-based pattern matching for fast detection without LLM calls.
    """

    # ==========================================================================
    # Pattern Definitions
    # ==========================================================================

    # Factual query patterns (single-hop, entity-centric)
    FACTUAL_PATTERNS = [
        # What/Which questions about specific attributes
        r"^what\s+(is|are|was|were)\s+\w+['\"]?s?\s+",  # "What is X's..."
        r"^what\s+\w+\s+(does|did|do|has|have)\s+",     # "What activities does X..."
        r"^what\s+(kind|type|sort)\s+of\s+",            # "What kind of..."
        r"^what\s+\w+\s+(has|have)\s+\w+\s+",           # "What pets has X..."
        r"^where\s+(is|are|was|were|does|did|do|has|have)\s+",  # "Where did X..."
        r"^who\s+(is|are|was|were|does|did)\s+",        # "Who is/does..."
        r"^which\s+\w+\s+(does|did|is|are|has|have)\s+", # "Which X does..."
        r"^how\s+many\s+",                               # "How many times..."
        r"^name\s+(the|all|some)\s+",                   # "Name the..."
        r"^list\s+",                                     # "List..."
        # Specific attribute questions
        r"\bidentity\b",                                 # Identity questions
        r"\brelationship\s+status\b",                   # Relationship status
        r"\bpet['\"]?s?\s+name",                        # Pet names
        r"\binstrument",                                 # Musical instruments
        r"\bartist|band",                               # Artists/bands
        r"\bactivit(y|ies)\b",                          # Activities
        r"\bhobb(y|ies)\b",                             # Hobbies
    ]

    # Temporal query patterns
    TEMPORAL_PATTERNS = [
        r"^when\s+(did|does|was|were|is|has|have|will)\s+",  # "When did X..."
        r"^how\s+long\s+(has|have|did|does|ago)\s+",         # "How long has X..."
        r"\b(year|month|week|day|date|time)\b",              # Time units
        r"\b(ago|last|previous|recent|next|upcoming)\b",     # Relative time
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\b20[0-9]{2}\b",                                   # Years like 2022, 2023
        r"\bduration\b",
        r"\bfrequency\b",
        r"\bsince\s+when\b",
        r"\bfor\s+how\s+long\b",
    ]

    # Multi-hop / inference patterns
    MULTI_HOP_PATTERNS = [
        r"^why\s+(did|does|do|is|are|was|were|would|might|could)\s+",  # "Why did X..."
        r"^what\s+would\s+",                             # "What would X..."
        r"^what\s+(might|could|should)\s+",              # Hypothetical
        r"\blikely\b",                                   # Inference indicator
        r"\bprobably\b",
        r"\bmight\s+be\b",
        r"\bcould\s+be\b",
        r"\bwould\s+\w+\s+be\b",
        r"\binfer|deduce|conclude|reason\b",
        r"\bbased\s+on\b",
        r"\bimpl(y|ies|ied)\b",
        r"\bsuggest(s|ed)?\b",
        r"\b(political|religious|ideological)\s+(leaning|stance|view|belief)\b",
    ]

    # Adversarial patterns (contradiction, negation, comparison)
    ADVERSARIAL_PATTERNS = [
        r"\bnot\s+(true|correct|right|accurate)\b",
        r"\bcontradicts?\b",
        r"\binconsistent\b",
        r"\bdisagree\b",
        r"\bfalse\b",
        r"\bwrong\b",
        r"\bnever\b",
        r"\bopposite\b",
        r"\bconflict(s|ing)?\b",
        r"\bdifferent\s+from\b",
        r"\bchange(d|s)?\s+(his|her|their)\s+mind\b",
        r"\bused\s+to\s+but\s+now\b",
        r"\bno\s+longer\b",
        r"\bbefore\s+but\s+now\b",
    ]

    def __init__(self, use_llm: bool = False):
        """
        Initialize the query type detector.

        Args:
            use_llm: If True, uses LLM for ambiguous cases (not implemented yet)
        """
        self.use_llm = use_llm

        # Compile patterns for efficiency
        self._factual_patterns = [re.compile(p, re.IGNORECASE) for p in self.FACTUAL_PATTERNS]
        self._temporal_patterns = [re.compile(p, re.IGNORECASE) for p in self.TEMPORAL_PATTERNS]
        self._multi_hop_patterns = [re.compile(p, re.IGNORECASE) for p in self.MULTI_HOP_PATTERNS]
        self._adversarial_patterns = [re.compile(p, re.IGNORECASE) for p in self.ADVERSARIAL_PATTERNS]

    def detect(self, query: str) -> QueryTypeResult:
        """
        Detect the type of a query.

        Args:
            query: The query string to analyze

        Returns:
            QueryTypeResult with detected type and confidence
        """
        query = query.strip()

        # Count matches for each type
        factual_matches = self._count_matches(query, self._factual_patterns)
        temporal_matches = self._count_matches(query, self._temporal_patterns)
        multi_hop_matches = self._count_matches(query, self._multi_hop_patterns)
        adversarial_matches = self._count_matches(query, self._adversarial_patterns)

        # Calculate scores (weighted by pattern specificity)
        scores = {
            QueryType.FACTUAL: len(factual_matches) * 1.0,
            QueryType.TEMPORAL: len(temporal_matches) * 1.2,  # Slightly favor temporal
            QueryType.MULTI_HOP: len(multi_hop_matches) * 1.3,  # Favor multi-hop patterns
            QueryType.ADVERSARIAL: len(adversarial_matches) * 1.1,
        }

        # Apply additional heuristics
        scores = self._apply_heuristics(query, scores)

        # Find best match
        max_score = max(scores.values())

        if max_score == 0:
            return QueryTypeResult(
                query_type=QueryType.DEFAULT,
                confidence=0.5,
                matched_patterns=[]
            )

        # Get the winning type
        best_type = max(scores, key=scores.get)

        # Calculate confidence (normalized)
        total_score = sum(scores.values())
        confidence = min(0.95, scores[best_type] / total_score) if total_score > 0 else 0.5

        # Get matched patterns for the winning type
        if best_type == QueryType.FACTUAL:
            matched = factual_matches
        elif best_type == QueryType.TEMPORAL:
            matched = temporal_matches
        elif best_type == QueryType.MULTI_HOP:
            matched = multi_hop_matches
        elif best_type == QueryType.ADVERSARIAL:
            matched = adversarial_matches
        else:
            matched = []

        return QueryTypeResult(
            query_type=best_type,
            confidence=confidence,
            matched_patterns=matched
        )

    def _count_matches(
        self,
        query: str,
        patterns: List[re.Pattern]
    ) -> List[str]:
        """Count how many patterns match the query."""
        matches = []
        for pattern in patterns:
            if pattern.search(query):
                matches.append(pattern.pattern)
        return matches

    def _apply_heuristics(
        self,
        query: str,
        scores: dict
    ) -> dict:
        """Apply additional heuristics to refine scores."""
        query_lower = query.lower()

        # Strong factual indicators (boost factual score)
        if any(word in query_lower for word in [
            "pet", "name", "instrument", "activity", "hobby",
            "identity", "status", "favorite", "type of"
        ]):
            scores[QueryType.FACTUAL] += 1.5

        # Strong temporal indicators (boost temporal score)
        if any(word in query_lower for word in [
            "when", "how long", "ago", "year", "date", "since"
        ]):
            scores[QueryType.TEMPORAL] += 1.5

        # Strong multi-hop indicators
        if any(phrase in query_lower for phrase in [
            "what would", "likely be", "political leaning",
            "infer", "based on", "implies"
        ]):
            scores[QueryType.MULTI_HOP] += 2.0

        # Strong adversarial indicators
        if any(phrase in query_lower for phrase in [
            "no longer", "changed", "used to", "contradiction"
        ]):
            scores[QueryType.ADVERSARIAL] += 2.0

        # Factual questions starting with specific patterns
        if re.match(r"^(what|where|who|which|how many)\s", query_lower):
            # But check if it's actually temporal or multi-hop
            if "when" not in query_lower and "would" not in query_lower:
                scores[QueryType.FACTUAL] += 0.5

        return scores

    def get_query_type_name(self, query: str) -> str:
        """Get just the query type name as a string."""
        result = self.detect(query)
        return result.query_type.value


# =============================================================================
# Singleton Instance
# =============================================================================

_detector: Optional[QueryTypeDetector] = None


def get_query_type_detector() -> QueryTypeDetector:
    """Get the singleton query type detector instance."""
    global _detector
    if _detector is None:
        _detector = QueryTypeDetector()
    return _detector


def detect_query_type(query: str) -> QueryTypeResult:
    """
    Convenience function to detect query type.

    Args:
        query: The query string

    Returns:
        QueryTypeResult with detected type
    """
    return get_query_type_detector().detect(query)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test cases
    test_queries = [
        # Factual (should get high keyword weight)
        "What instruments does Melanie play?",
        "Where did Caroline move from 4 years ago?",
        "What are Melanie's pets' names?",
        "What is Caroline's identity?",
        "What activities does Melanie partake in?",

        # Temporal (should maintain current weights)
        "When did Melanie paint a sunrise?",
        "How long has Caroline had her current group of friends for?",
        "When did Caroline apply to adoption agencies?",

        # Multi-hop (should get high graph weight)
        "What would Caroline's political leaning likely be?",
        "Why does Melanie enjoy painting?",

        # Adversarial (should maintain current weights)
        "Did Caroline change her mind about adoption?",
        "Is it true that Melanie no longer plays music?",

        # Default
        "Tell me about Caroline.",
    ]

    detector = QueryTypeDetector()

    print("Query Type Detection Test Results")
    print("=" * 60)

    for query in test_queries:
        result = detector.detect(query)
        print(f"\nQuery: {query}")
        print(f"  Type: {result.query_type.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        if result.matched_patterns:
            print(f"  Matched: {len(result.matched_patterns)} patterns")
