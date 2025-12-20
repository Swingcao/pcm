"""
Dynamic Topic Extractor
=======================
Replaces fixed intent domains with dynamic topic discovery.

Instead of classifying queries into predefined categories (Coding, Academic, etc.),
this module extracts relevant topics/themes directly from content.

Key Features:
1. Dynamic Topic Extraction: No predefined categories
2. Flexible Node Topics: Nodes tagged with extracted topics
3. Topic Similarity Matching: Query topics matched with node topics

This solves the limitation of fixed intent domains that can't handle
new or unexpected conversation topics.
"""

import re
import json
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from src.utils.llm_client import get_llm_client, LLMClient


@dataclass
class TopicSet:
    """
    Represents extracted topics from content.

    Replaces the fixed Intent class with dynamic topics.
    """
    topics: List[str]  # List of topic labels (e.g., ["music", "instruments", "hobbies"])
    primary_topic: str  # Most relevant topic
    topic_weights: Dict[str, float]  # Topic -> relevance weight
    entities: List[str]  # Named entities mentioned (people, places, etc.)
    extracted_at: datetime = field(default_factory=datetime.now)

    def get_weight(self, topic: str) -> float:
        """Get weight for a specific topic (case-insensitive)."""
        topic_lower = topic.lower()
        for t, w in self.topic_weights.items():
            if t.lower() == topic_lower:
                return w
        return 0.0

    def has_overlap(self, other_topics: List[str], threshold: float = 0.3) -> Tuple[bool, float]:
        """
        Check if there's significant topic overlap with another set.

        Returns:
            (has_overlap, overlap_score)
        """
        if not other_topics:
            return False, 0.0

        self_topics_lower = {t.lower() for t in self.topics}
        other_topics_lower = {t.lower() for t in other_topics}

        # Direct match
        intersection = self_topics_lower & other_topics_lower
        if intersection:
            # Weighted overlap score
            score = sum(self.topic_weights.get(t, 0.5) for t in intersection)
            return True, min(1.0, score)

        # Partial match (substring)
        partial_matches = 0
        for self_t in self_topics_lower:
            for other_t in other_topics_lower:
                if self_t in other_t or other_t in self_t:
                    partial_matches += 1
                    break

        if partial_matches > 0:
            score = partial_matches / max(len(self_topics_lower), len(other_topics_lower))
            return score >= threshold, score

        return False, 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "topics": self.topics,
            "primary_topic": self.primary_topic,
            "topic_weights": self.topic_weights,
            "entities": self.entities,
            "extracted_at": self.extracted_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TopicSet":
        """Create from dictionary."""
        return cls(
            topics=data.get("topics", []),
            primary_topic=data.get("primary_topic", "general"),
            topic_weights=data.get("topic_weights", {}),
            entities=data.get("entities", []),
            extracted_at=datetime.fromisoformat(data["extracted_at"])
                if "extracted_at" in data else datetime.now()
        )


class TopicExtractor:
    """
    Dynamic Topic Extractor using LLM.

    Replaces fixed intent classification with open-ended topic extraction.
    """

    EXTRACTION_PROMPT = """Analyze the following text and extract:
1. Topics: The main themes/subjects discussed (2-5 topics)
2. Primary Topic: The most relevant topic
3. Entities: Named entities mentioned (people, places, organizations)

Text to analyze:
---
{text}
---

{context_note}

Respond with JSON only:
{{
    "topics": ["topic1", "topic2", ...],
    "primary_topic": "main topic",
    "topic_weights": {{"topic1": 0.9, "topic2": 0.7, ...}},
    "entities": ["entity1", "entity2", ...]
}}

IMPORTANT:
- Topics should be general themes (e.g., "music", "travel", "family", "hobbies")
- Keep topics concise (1-3 words each)
- Weights should be 0.0-1.0 based on relevance
- Include ALL mentioned entities (names, places, etc.)"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize the topic extractor."""
        self.llm_client = llm_client or get_llm_client()
        self._cache: Dict[str, TopicSet] = {}

    async def extract(
        self,
        text: str,
        context: str = "",
        use_cache: bool = True
    ) -> TopicSet:
        """
        Extract topics from text.

        Args:
            text: Text to analyze
            context: Optional conversation context
            use_cache: Whether to use cached results

        Returns:
            TopicSet with extracted topics
        """
        # Check cache
        cache_key = hash(text[:100])
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        context_note = f"Conversation context:\n{context}\n" if context else ""

        prompt = self.EXTRACTION_PROMPT.format(
            text=text,
            context_note=context_note
        )

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=config.ROUTER_LLM_MODEL,  # Use lightweight model
                temperature=0.3,
                max_tokens=500
            )

            content = response["content"]

            # Parse JSON
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            data = json.loads(content)

            topic_set = TopicSet(
                topics=data.get("topics", ["general"]),
                primary_topic=data.get("primary_topic", "general"),
                topic_weights=data.get("topic_weights", {}),
                entities=data.get("entities", [])
            )

            # Cache result
            self._cache[cache_key] = topic_set

            return topic_set

        except Exception as e:
            print(f"Topic extraction failed: {e}")
            # Fallback to rule-based extraction
            return self.extract_simple(text)

    def extract_simple(self, text: str) -> TopicSet:
        """
        Simple rule-based topic extraction (fallback).

        Uses keyword patterns and entity extraction.
        """
        text_lower = text.lower()

        # Topic patterns
        topic_patterns = {
            "music": ["music", "song", "instrument", "play", "violin", "piano", "guitar", "clarinet", "band", "concert"],
            "travel": ["travel", "trip", "visit", "country", "city", "flight", "vacation", "moved from", "moved to"],
            "family": ["family", "parent", "mother", "father", "sibling", "brother", "sister", "child", "kid"],
            "work": ["work", "job", "career", "office", "meeting", "project", "deadline", "colleague"],
            "hobbies": ["hobby", "paint", "draw", "garden", "cook", "read", "game", "sport", "hiking"],
            "education": ["school", "university", "college", "study", "learn", "course", "degree", "research"],
            "technology": ["computer", "software", "app", "code", "program", "technology", "internet"],
            "health": ["health", "doctor", "hospital", "exercise", "diet", "medicine", "sick", "wellness"],
            "social": ["friend", "party", "event", "community", "lgbtq", "group", "social", "relationship"],
            "art": ["art", "paint", "draw", "museum", "gallery", "creative", "design", "photography"],
            "food": ["food", "cook", "recipe", "restaurant", "eat", "meal", "cuisine", "drink"],
            "location": ["live", "move", "country", "city", "house", "apartment", "neighborhood"]
        }

        # Find matching topics
        matched_topics: Dict[str, float] = {}
        for topic, keywords in topic_patterns.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                matched_topics[topic] = min(1.0, matches * 0.3)

        # Extract entities (simple pattern matching)
        entities = []

        # Names (capitalized words not at sentence start)
        name_pattern = r'(?<=[.!?\s])[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'
        names = re.findall(name_pattern, text)
        entities.extend(names)

        # Countries/places (common ones)
        places = ["Sweden", "France", "Germany", "Italy", "Spain", "Japan", "China",
                  "London", "Paris", "New York", "Tokyo", "Boston", "Seattle"]
        for place in places:
            if place.lower() in text_lower:
                entities.append(place)
                if "location" not in matched_topics:
                    matched_topics["location"] = 0.5
                matched_topics["location"] = min(1.0, matched_topics.get("location", 0) + 0.3)

        # Default topic if none matched
        if not matched_topics:
            matched_topics["general"] = 0.5

        # Sort by weight
        sorted_topics = sorted(matched_topics.items(), key=lambda x: x[1], reverse=True)
        topics = [t[0] for t in sorted_topics[:5]]

        return TopicSet(
            topics=topics,
            primary_topic=topics[0] if topics else "general",
            topic_weights=matched_topics,
            entities=list(set(entities))  # Deduplicate
        )

    def clear_cache(self):
        """Clear the topic cache."""
        self._cache.clear()


class TopicMatcher:
    """
    Matches query topics with node topics for retrieval weighting.

    Replaces the fixed domain-based intent relevance calculation.

    Supports both TopicSet and Intent objects (v1.2 compatibility).
    """

    def __init__(self, embedding_model=None):
        """
        Initialize the topic matcher.

        Args:
            embedding_model: Optional embedding model for semantic matching
        """
        self.embedding_model = embedding_model
        self._topic_embeddings: Dict[str, List[float]] = {}

    def compute_relevance(
        self,
        query_topics,  # Can be TopicSet or Intent
        node_topics: List[str],
        node_entities: Optional[List[str]] = None
    ) -> float:
        """
        Compute relevance score between query topics and node topics.

        Replaces: intent.distribution.get(node_domain, 0.1)

        Args:
            query_topics: Topics extracted from query (TopicSet or Intent)
            node_topics: Topics assigned to node
            node_entities: Entities mentioned in node

        Returns:
            Relevance score (0.0-1.0)
        """
        if not node_topics:
            return 0.3  # Default for nodes without topics

        # Extract topics from query_topics (support both TopicSet and Intent)
        if hasattr(query_topics, 'topics') and query_topics.topics:
            q_topics = query_topics.topics
        elif hasattr(query_topics, 'get_topics'):
            q_topics = query_topics.get_topics()
        elif hasattr(query_topics, 'distribution'):
            q_topics = list(query_topics.distribution.keys())
        else:
            q_topics = []

        # Extract entities from query_topics
        if hasattr(query_topics, 'entities') and query_topics.entities:
            q_entities = query_topics.entities
        else:
            q_entities = []

        # Extract topic weights
        if hasattr(query_topics, 'topic_weights'):
            q_weights = query_topics.topic_weights
        elif hasattr(query_topics, 'distribution'):
            q_weights = query_topics.distribution
        else:
            q_weights = {t: 0.5 for t in q_topics}

        # Method 1: Direct topic overlap
        has_overlap, overlap_score = self._compute_overlap(q_topics, q_weights, node_topics)
        if has_overlap:
            return max(0.5, overlap_score)

        # Method 2: Entity overlap
        if node_entities and q_entities:
            query_entities_lower = {e.lower() for e in q_entities}
            node_entities_lower = {e.lower() for e in node_entities}
            entity_overlap = query_entities_lower & node_entities_lower
            if entity_overlap:
                return 0.8  # High relevance for entity match

        # Method 3: Semantic similarity (if embedding model available)
        if self.embedding_model and q_topics and node_topics:
            try:
                query_text = " ".join(q_topics)
                node_text = " ".join(node_topics)

                query_emb = self._get_topic_embedding(query_text)
                node_emb = self._get_topic_embedding(node_text)

                # Cosine similarity
                similarity = self._cosine_similarity(query_emb, node_emb)
                if similarity > 0.5:
                    return similarity
            except Exception:
                pass

        # Default: low relevance
        return 0.2

    def _compute_overlap(
        self,
        q_topics: List[str],
        q_weights: Dict[str, float],
        node_topics: List[str],
        threshold: float = 0.3
    ) -> Tuple[bool, float]:
        """
        Compute topic overlap between query topics and node topics.

        Returns:
            (has_overlap, overlap_score)
        """
        if not q_topics or not node_topics:
            return False, 0.0

        q_topics_lower = {t.lower() for t in q_topics}
        node_topics_lower = {t.lower() for t in node_topics}

        # Direct match
        intersection = q_topics_lower & node_topics_lower
        if intersection:
            # Weighted overlap score
            score = sum(q_weights.get(t, 0.5) for t in intersection)
            return True, min(1.0, score)

        # Partial match (substring)
        partial_matches = 0
        for q_t in q_topics_lower:
            for n_t in node_topics_lower:
                if q_t in n_t or n_t in q_t:
                    partial_matches += 1
                    break

        if partial_matches > 0:
            score = partial_matches / max(len(q_topics_lower), len(node_topics_lower))
            return score >= threshold, score

        return False, 0.0

    def _get_topic_embedding(self, text: str) -> List[float]:
        """Get embedding for topic text (with caching)."""
        if text in self._topic_embeddings:
            return self._topic_embeddings[text]

        embedding = self.embedding_model.encode(text).tolist()
        self._topic_embeddings[text] = embedding
        return embedding

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# =============================================================================
# Convenience Functions
# =============================================================================

_topic_extractor: Optional[TopicExtractor] = None


def get_topic_extractor() -> TopicExtractor:
    """Get singleton topic extractor instance."""
    global _topic_extractor
    if _topic_extractor is None:
        _topic_extractor = TopicExtractor()
    return _topic_extractor


async def extract_topics(
    text: str,
    context: str = "",
    use_llm: bool = True
) -> TopicSet:
    """
    Extract topics from text.

    Args:
        text: Text to analyze
        context: Optional context
        use_llm: Whether to use LLM (True) or rule-based extraction (False)

    Returns:
        TopicSet with extracted topics
    """
    extractor = get_topic_extractor()

    if use_llm:
        return await extractor.extract(text, context)
    else:
        return extractor.extract_simple(text)


def extract_topics_sync(
    text: str,
    context: str = "",
    use_llm: bool = True
) -> TopicSet:
    """
    Synchronous wrapper for topic extraction.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, use simple extraction
            return get_topic_extractor().extract_simple(text)
        return loop.run_until_complete(extract_topics(text, context, use_llm))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(extract_topics(text, context, use_llm))
