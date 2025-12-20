"""
Knowledge Graph Edge Creator
============================
Creates semantic edges between memory nodes to enable graph-based reasoning.

This module addresses the problem of having 0 edges in the knowledge graph,
which makes graph expansion during retrieval ineffective.

Key Features:
- Subject-based linking (same entity -> "related")
- Contradiction detection (conflicting facts -> "supersedes")
- Temporal sequence linking (time-ordered events -> "follows")
- Semantic similarity linking (similar content -> "similar_to")
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
import re

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.core.types import MemoryNode, MemoryEdge, NodeType
from src.utils.llm_client import get_llm_client


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class EdgeCandidate:
    """A potential edge to be created."""
    source_id: str
    target_id: str
    relation: str
    weight: float
    reason: str
    metadata: Dict[str, Any] = None

    def to_memory_edge(self) -> MemoryEdge:
        """Convert to MemoryEdge."""
        return MemoryEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            relation=self.relation,
            weight=self.weight,
            metadata=self.metadata or {}
        )


class ContradictionAnalysis(BaseModel):
    """LLM output for contradiction detection."""
    is_contradictory: bool = Field(default=False)
    contradiction_type: str = Field(default="none")  # none, direct, temporal, partial
    confidence: float = Field(default=0.0)
    explanation: str = Field(default="")


# =============================================================================
# Edge Relation Types
# =============================================================================

class EdgeRelationType:
    """Standard edge relation types."""
    RELATED = "related"           # Same subject/entity
    SUPERSEDES = "supersedes"     # New fact replaces old (contradiction)
    FOLLOWS = "follows"           # Temporal sequence
    DERIVED_FROM = "derived_from" # Hypothesis derived from observation
    SIMILAR_TO = "similar_to"     # Semantically similar content
    SUPPORTS = "supports"         # Evidence supporting a fact
    CONTRADICTS = "contradicts"   # Directly contradictory (both may be valid)
    ELABORATES = "elaborates"     # Adds detail to existing fact


# =============================================================================
# Entity Extraction
# =============================================================================

class EntityExtractor:
    """Extract entities from memory node content."""

    # Patterns for common entity types
    PATTERNS = [
        # [timestamp] Speaker: message format
        (r'\] (\w+):', 'speaker'),
        # Names (capitalized words)
        (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', 'name'),
        # Locations
        (r'(?:in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'location'),
    ]

    @classmethod
    def extract_speaker(cls, content: str) -> Optional[str]:
        """Extract speaker from message format."""
        match = re.search(r'\] (\w+):', content)
        return match.group(1) if match else None

    @classmethod
    def extract_entities(cls, content: str) -> Set[str]:
        """Extract named entities from content."""
        entities = set()

        # Extract speaker
        speaker = cls.extract_speaker(content)
        if speaker:
            entities.add(speaker.lower())

        # Extract capitalized names (simple NER)
        names = re.findall(r'\b([A-Z][a-z]+)\b', content)
        for name in names:
            # Filter common words
            if name.lower() not in {'the', 'i', 'a', 'an', 'it', 'hey', 'hi', 'wow', 'oh'}:
                entities.add(name.lower())

        return entities

    @classmethod
    def extract_topics(cls, content: str) -> Set[str]:
        """Extract topic keywords from content."""
        # Remove timestamp prefix
        content_clean = re.sub(r'\[.*?\]', '', content)

        # Common topic indicators
        topic_patterns = [
            r'(?:about|discussing|talking about|mentioned)\s+(\w+)',
            r'(\w+)\s+(?:support group|conference|meeting|class)',
        ]

        topics = set()
        for pattern in topic_patterns:
            matches = re.findall(pattern, content_clean, re.IGNORECASE)
            topics.update(m.lower() for m in matches)

        return topics


# =============================================================================
# Edge Creation Rules
# =============================================================================

class EdgeCreator:
    """
    Creates edges between memory nodes based on semantic relationships.

    Edge creation rules:
    1. Same subject -> "related" (weight: 0.6-0.8)
    2. Contradictory facts -> "supersedes" (weight: 0.9)
    3. Temporal sequence -> "follows" (weight: 0.7)
    4. High similarity -> "similar_to" (weight: based on similarity)
    5. Supporting evidence -> "supports" (weight: 0.7)
    """

    def __init__(self, use_llm: bool = True, similarity_threshold: float = 0.8):
        """
        Initialize edge creator.

        Args:
            use_llm: Use LLM for contradiction detection
            similarity_threshold: Threshold for creating similarity edges
        """
        self.use_llm = use_llm
        self.llm_client = get_llm_client() if use_llm else None
        self.similarity_threshold = similarity_threshold
        self.entity_extractor = EntityExtractor()

    async def create_edges_for_node(
        self,
        new_node: MemoryNode,
        existing_nodes: List[MemoryNode],
        embeddings: Optional[Dict[str, List[float]]] = None
    ) -> List[EdgeCandidate]:
        """
        Create edges between a new node and existing nodes.

        Args:
            new_node: The newly added node
            existing_nodes: List of existing nodes to compare against
            embeddings: Optional pre-computed embeddings {node_id: embedding}

        Returns:
            List of EdgeCandidate objects
        """
        edges = []

        # Extract entities from new node
        new_entities = self.entity_extractor.extract_entities(new_node.content)
        new_speaker = self.entity_extractor.extract_speaker(new_node.content)

        for existing in existing_nodes:
            if existing.id == new_node.id:
                continue

            # Extract entities from existing node
            existing_entities = self.entity_extractor.extract_entities(existing.content)
            existing_speaker = self.entity_extractor.extract_speaker(existing.content)

            # Rule 1: Same subject/speaker -> "related"
            if new_speaker and existing_speaker and new_speaker == existing_speaker:
                edges.append(EdgeCandidate(
                    source_id=new_node.id,
                    target_id=existing.id,
                    relation=EdgeRelationType.RELATED,
                    weight=0.6,
                    reason=f"Same speaker: {new_speaker}"
                ))

            # Rule 2: Shared entities -> "related"
            shared_entities = new_entities & existing_entities
            if shared_entities:
                weight = min(0.8, 0.5 + 0.1 * len(shared_entities))
                edges.append(EdgeCandidate(
                    source_id=new_node.id,
                    target_id=existing.id,
                    relation=EdgeRelationType.RELATED,
                    weight=weight,
                    reason=f"Shared entities: {shared_entities}"
                ))

            # Rule 3: Check for contradiction
            contradiction = await self._check_contradiction(new_node, existing)
            if contradiction:
                edges.append(contradiction)

            # Rule 4: Temporal sequence
            temporal_edge = self._check_temporal_sequence(new_node, existing)
            if temporal_edge:
                edges.append(temporal_edge)

        # Rule 5: Similarity-based edges (if embeddings provided)
        if embeddings:
            similarity_edges = self._create_similarity_edges(
                new_node, existing_nodes, embeddings
            )
            edges.extend(similarity_edges)

        # Deduplicate edges (keep highest weight for same source-target pair)
        edges = self._deduplicate_edges(edges)

        return edges

    async def _check_contradiction(
        self,
        new_node: MemoryNode,
        existing_node: MemoryNode
    ) -> Optional[EdgeCandidate]:
        """
        Check if two nodes contain contradictory information.

        Uses LLM for semantic contradiction detection.
        """
        # Quick filter: nodes must share entities
        new_entities = self.entity_extractor.extract_entities(new_node.content)
        existing_entities = self.entity_extractor.extract_entities(existing_node.content)

        if not (new_entities & existing_entities):
            return None

        if not self.use_llm or not self.llm_client:
            # Rule-based contradiction detection (simple)
            return self._rule_based_contradiction(new_node, existing_node)

        # LLM-based contradiction detection
        prompt = f"""Analyze if these two statements contain contradictory information.

Statement 1 (newer): {new_node.content}
Statement 2 (older): {existing_node.content}

Consider:
- Direct contradictions (e.g., "likes X" vs "dislikes X")
- Temporal contradictions (e.g., different times for same event)
- Partial contradictions (e.g., "plays violin" vs "plays guitar")

Output JSON:
{{
    "is_contradictory": true/false,
    "contradiction_type": "none|direct|temporal|partial",
    "confidence": 0.0-1.0,
    "explanation": "brief explanation"
}}"""

        try:
            result = await self.llm_client.structured_output(
                messages=[{"role": "user", "content": prompt}],
                output_schema=ContradictionAnalysis,
                temperature=0.1
            )

            if result.is_contradictory and result.confidence >= 0.7:
                return EdgeCandidate(
                    source_id=new_node.id,
                    target_id=existing_node.id,
                    relation=EdgeRelationType.SUPERSEDES,
                    weight=0.9,
                    reason=result.explanation,
                    metadata={
                        "contradiction_type": result.contradiction_type,
                        "confidence": result.confidence
                    }
                )
        except Exception as e:
            print(f"LLM contradiction check failed: {e}")

        return None

    def _rule_based_contradiction(
        self,
        new_node: MemoryNode,
        existing_node: MemoryNode
    ) -> Optional[EdgeCandidate]:
        """Simple rule-based contradiction detection."""
        # Patterns that indicate potential contradiction
        contradiction_patterns = [
            # Opposite sentiment
            (r'(?:like|love|enjoy)', r'(?:dislike|hate|don\'t like)'),
            # Location change
            (r'(?:live|living) in (\w+)', r'(?:moved from|used to live in)'),
            # Status change
            (r'(?:am|is) (\w+)', r'(?:was|used to be) (\w+)'),
        ]

        new_content = new_node.content.lower()
        existing_content = existing_node.content.lower()

        for pattern1, pattern2 in contradiction_patterns:
            if (re.search(pattern1, new_content) and re.search(pattern2, existing_content)) or \
               (re.search(pattern2, new_content) and re.search(pattern1, existing_content)):
                return EdgeCandidate(
                    source_id=new_node.id,
                    target_id=existing_node.id,
                    relation=EdgeRelationType.SUPERSEDES,
                    weight=0.7,
                    reason="Pattern-based contradiction detected"
                )

        return None

    def _check_temporal_sequence(
        self,
        new_node: MemoryNode,
        existing_node: MemoryNode
    ) -> Optional[EdgeCandidate]:
        """Check if nodes form a temporal sequence."""
        # Extract timestamps
        new_time = self._extract_timestamp(new_node.content)
        existing_time = self._extract_timestamp(existing_node.content)

        if not new_time or not existing_time:
            return None

        # Check if same speaker
        new_speaker = self.entity_extractor.extract_speaker(new_node.content)
        existing_speaker = self.entity_extractor.extract_speaker(existing_node.content)

        if new_speaker != existing_speaker:
            return None

        # Create temporal edge if new node is after existing
        if new_time > existing_time:
            time_diff = (new_time - existing_time).days
            # Weight decreases with time difference
            weight = max(0.3, 0.8 - time_diff * 0.01)

            return EdgeCandidate(
                source_id=new_node.id,
                target_id=existing_node.id,
                relation=EdgeRelationType.FOLLOWS,
                weight=weight,
                reason=f"Temporal sequence: {existing_time.date()} -> {new_time.date()}"
            )

        return None

    def _extract_timestamp(self, content: str) -> Optional[datetime]:
        """Extract timestamp from message content."""
        # Pattern: "1:56 pm on 8 May, 2023"
        patterns = [
            (r'(\d{1,2}:\d{2}\s*[ap]m)\s+on\s+(\d{1,2}\s+\w+,?\s+\d{4})', '%I:%M %p %d %B %Y'),
            (r'(\d{1,2}:\d{2}\s*[ap]m)\s+on\s+(\w+\s+\d{1,2},?\s+\d{4})', '%I:%M %p %B %d %Y'),
        ]

        for pattern, fmt in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    time_str = match.group(1).replace(' ', '')
                    date_str = match.group(2).replace(',', '')
                    full_str = f"{time_str} {date_str}"
                    # Normalize format
                    full_str = re.sub(r'(\d{1,2})(\w{2})\s', r'\1 ', full_str)
                    return datetime.strptime(full_str, fmt)
                except Exception:
                    continue

        return None

    def _create_similarity_edges(
        self,
        new_node: MemoryNode,
        existing_nodes: List[MemoryNode],
        embeddings: Dict[str, List[float]]
    ) -> List[EdgeCandidate]:
        """Create edges based on embedding similarity."""
        edges = []

        if new_node.id not in embeddings:
            return edges

        new_emb = embeddings[new_node.id]

        for existing in existing_nodes:
            if existing.id == new_node.id or existing.id not in embeddings:
                continue

            existing_emb = embeddings[existing.id]

            # Cosine similarity
            similarity = self._cosine_similarity(new_emb, existing_emb)

            if similarity >= self.similarity_threshold:
                edges.append(EdgeCandidate(
                    source_id=new_node.id,
                    target_id=existing.id,
                    relation=EdgeRelationType.SIMILAR_TO,
                    weight=similarity,
                    reason=f"Semantic similarity: {similarity:.3f}"
                ))

        return edges

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _deduplicate_edges(
        self,
        edges: List[EdgeCandidate]
    ) -> List[EdgeCandidate]:
        """Remove duplicate edges, keeping the one with highest weight."""
        edge_map: Dict[Tuple[str, str], EdgeCandidate] = {}

        for edge in edges:
            key = (edge.source_id, edge.target_id)
            if key not in edge_map or edge.weight > edge_map[key].weight:
                edge_map[key] = edge

        return list(edge_map.values())


# =============================================================================
# Batch Edge Creation
# =============================================================================

async def create_edges_for_graph(
    nodes: List[MemoryNode],
    embeddings: Optional[Dict[str, List[float]]] = None,
    use_llm: bool = True,
    max_edges_per_node: int = 5
) -> List[MemoryEdge]:
    """
    Create edges for all nodes in a graph.

    Args:
        nodes: List of all nodes
        embeddings: Optional embeddings for similarity edges
        use_llm: Whether to use LLM for contradiction detection
        max_edges_per_node: Maximum edges to create per node

    Returns:
        List of MemoryEdge objects
    """
    creator = EdgeCreator(use_llm=use_llm)
    all_edges: List[MemoryEdge] = []

    for i, node in enumerate(nodes):
        # Compare with other nodes
        other_nodes = nodes[:i] + nodes[i+1:]

        candidates = await creator.create_edges_for_node(
            node, other_nodes, embeddings
        )

        # Limit edges per node
        candidates.sort(key=lambda x: x.weight, reverse=True)
        top_candidates = candidates[:max_edges_per_node]

        for candidate in top_candidates:
            all_edges.append(candidate.to_memory_edge())

    return all_edges


def create_edges_sync(
    nodes: List[MemoryNode],
    embeddings: Optional[Dict[str, List[float]]] = None,
    use_llm: bool = False
) -> List[MemoryEdge]:
    """Synchronous wrapper for edge creation."""
    return asyncio.run(create_edges_for_graph(nodes, embeddings, use_llm))
