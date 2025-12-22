"""
Hybrid Retriever
================
Combines semantic (embedding) and lexical (keyword) retrieval for improved recall.

This module addresses the problem of embedding-only retrieval missing exact
keyword matches, especially for rare terms like proper nouns.

Key Features:
- Combines embedding similarity with BM25 keyword matching
- Configurable fusion weights
- Graph-based expansion using edges
- Reciprocal Rank Fusion (RRF) for score combination
- [v1.3] Query-adaptive weights based on query type detection

v1.3 Updates:
- Added QueryType detection for adaptive retrieval weights
- Factual queries now use keyword_weight=0.5 for better entity matching
- Temporal queries preserve recency_weight for good temporal performance
- Multi-hop queries use higher graph_weight for reasoning traversal
"""

import os
import json
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.core.types import MemoryNode, MemoryEdge, Intent
from src.utils.keyword_index import InvertedIndex, SearchResult, create_index_from_nodes
from src.utils.query_type_detector import (
    QueryType, QueryTypeResult, QueryTypeDetector, detect_query_type
)
from src.utils.entity_index import EntityCentricIndex, create_entity_index_from_nodes


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class HybridSearchResult:
    """Result from hybrid retrieval."""
    node: MemoryNode
    final_score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    graph_score: float = 0.0
    matched_keywords: List[str] = field(default_factory=list)
    expansion_path: List[str] = field(default_factory=list)  # IDs of nodes in expansion path
    query_type: Optional[str] = None  # v1.3: Detected query type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node.id,
            "content": self.node.content,
            "final_score": self.final_score,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "graph_score": self.graph_score,
            "matched_keywords": self.matched_keywords,
            "weight": self.node.weight,
            "query_type": self.query_type
        }


@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval."""
    # Fusion weights
    semantic_weight: float = 0.4    # α: weight for embedding similarity
    keyword_weight: float = 0.3     # β: weight for BM25 keyword score
    graph_weight: float = 0.2       # γ: weight for graph expansion score
    recency_weight: float = 0.1     # δ: weight for recency bonus

    # v1.4: Source text search weight (ε)
    source_text_weight: float = 0.0  # ε: weight for original text matching (default 0, add to enable)

    # Retrieval parameters
    top_k: int = 10
    over_fetch_factor: int = 3      # Fetch top_k * factor for re-ranking
    min_weight: float = 0.1         # Minimum node weight threshold

    # Graph expansion
    expansion_depth: int = 1
    edge_discount: float = 0.5      # Discount factor for neighbor scores

    # Keyword search parameters
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # Score normalization
    normalize_scores: bool = True

    def validate(self) -> None:
        """Validate that weights sum to 1.0."""
        total = (self.semantic_weight + self.keyword_weight +
                 self.graph_weight + self.recency_weight + self.source_text_weight)
        if abs(total - 1.0) > 0.01:
            # Normalize weights
            self.semantic_weight /= total
            self.keyword_weight /= total
            self.graph_weight /= total
            self.recency_weight /= total
            self.source_text_weight /= total


# =============================================================================
# Query-Adaptive Configuration (v1.3)
# =============================================================================

class AdaptiveRetrievalConfig:
    """
    Provides query-type specific retrieval configurations.

    This class maintains different weight configurations optimized for
    different query types to improve retrieval accuracy.

    Weight configurations are designed to:
    - FACTUAL: High keyword weight (0.50) for exact entity matching
    - TEMPORAL: Balanced weights with recency emphasis (preserved from v1.2)
    - MULTI_HOP: High graph weight (0.40) for reasoning traversal
    - ADVERSARIAL: Balanced weights (preserved from v1.2)
    - DEFAULT: Original balanced configuration

    v1.4: All configs now include source_text_weight for original text matching.
    """

    # Optimized configurations for each query type
    # These weights are tuned based on v1-2 analysis results
    # v1.4: Adjusted to include source_text_weight (taking from semantic)

    FACTUAL_CONFIG = RetrievalConfig(
        semantic_weight=0.20,
        keyword_weight=0.45,  # High keyword for entity matching
        graph_weight=0.10,
        recency_weight=0.10,
        source_text_weight=0.15  # v1.4: Search original text
    )

    TEMPORAL_CONFIG = RetrievalConfig(
        semantic_weight=0.30,
        keyword_weight=0.25,
        graph_weight=0.15,
        recency_weight=0.20,  # Higher recency for temporal queries
        source_text_weight=0.10  # v1.4: Search original text
    )

    MULTI_HOP_CONFIG = RetrievalConfig(
        semantic_weight=0.20,
        keyword_weight=0.20,
        graph_weight=0.40,  # High graph for multi-hop reasoning
        recency_weight=0.10,
        source_text_weight=0.10,  # v1.4: Search original text
        expansion_depth=2  # Enable 2-hop expansion for inference
    )

    ADVERSARIAL_CONFIG = RetrievalConfig(
        semantic_weight=0.25,
        keyword_weight=0.30,
        graph_weight=0.20,
        recency_weight=0.10,
        source_text_weight=0.15  # v1.4: Higher for preference changes
    )

    DEFAULT_CONFIG = RetrievalConfig(
        semantic_weight=0.35,
        keyword_weight=0.25,
        graph_weight=0.20,
        recency_weight=0.10,
        source_text_weight=0.10  # v1.4: Search original text
    )

    @classmethod
    def get_config_for_query_type(
        cls,
        query_type: QueryType,
        base_config: Optional[RetrievalConfig] = None
    ) -> RetrievalConfig:
        """
        Get the appropriate configuration for a query type.

        Args:
            query_type: The detected query type
            base_config: Optional base config to use for non-weight parameters

        Returns:
            RetrievalConfig optimized for the query type
        """
        # Select weight configuration based on query type
        if query_type == QueryType.FACTUAL:
            type_config = cls.FACTUAL_CONFIG
        elif query_type == QueryType.TEMPORAL:
            type_config = cls.TEMPORAL_CONFIG
        elif query_type == QueryType.MULTI_HOP:
            type_config = cls.MULTI_HOP_CONFIG
        elif query_type == QueryType.ADVERSARIAL:
            type_config = cls.ADVERSARIAL_CONFIG
        else:
            type_config = cls.DEFAULT_CONFIG

        # If no base config, return the type config directly
        if base_config is None:
            return type_config

        # Merge: use type config weights with base config parameters
        return RetrievalConfig(
            semantic_weight=type_config.semantic_weight,
            keyword_weight=type_config.keyword_weight,
            graph_weight=type_config.graph_weight,
            recency_weight=type_config.recency_weight,
            source_text_weight=type_config.source_text_weight,  # v1.4
            top_k=base_config.top_k,
            over_fetch_factor=base_config.over_fetch_factor,
            min_weight=base_config.min_weight,
            expansion_depth=type_config.expansion_depth if query_type == QueryType.MULTI_HOP else base_config.expansion_depth,
            edge_discount=base_config.edge_discount,
            bm25_k1=base_config.bm25_k1,
            bm25_b=base_config.bm25_b,
            normalize_scores=base_config.normalize_scores
        )

    @classmethod
    def get_weights_for_query(
        cls,
        query: str
    ) -> Tuple[float, float, float, float, QueryType]:
        """
        Convenience method to get weights and expansion depth for a query.

        Args:
            query: The query string

        Returns:
            Tuple of (semantic_weight, keyword_weight, graph_weight, recency_weight, query_type)
        """
        result = detect_query_type(query)
        config = cls.get_config_for_query_type(result.query_type)
        return (
            config.semantic_weight,
            config.keyword_weight,
            config.graph_weight,
            config.recency_weight,
            result.query_type
        )


# =============================================================================
# Score Fusion Methods
# =============================================================================

class ScoreFusion:
    """Methods for combining scores from different retrieval sources."""

    @staticmethod
    def linear_combination(
        scores: Dict[str, Dict[str, float]],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Linear combination of normalized scores.

        Args:
            scores: {source_name: {doc_id: score}}
            weights: {source_name: weight}

        Returns:
            {doc_id: combined_score}
        """
        # Normalize each source
        normalized = {}
        for source, source_scores in scores.items():
            if not source_scores:
                normalized[source] = {}
                continue

            max_score = max(source_scores.values())
            if max_score > 0:
                normalized[source] = {
                    doc_id: score / max_score
                    for doc_id, score in source_scores.items()
                }
            else:
                normalized[source] = source_scores

        # Combine
        all_docs = set()
        for source_scores in normalized.values():
            all_docs.update(source_scores.keys())

        combined = {}
        for doc_id in all_docs:
            score = 0.0
            for source, weight in weights.items():
                score += weight * normalized.get(source, {}).get(doc_id, 0.0)
            combined[doc_id] = score

        return combined

    @staticmethod
    def reciprocal_rank_fusion(
        rankings: List[List[str]],
        k: int = 60
    ) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion (RRF) for combining multiple rankings.

        RRF(d) = Σ 1 / (k + rank_i(d))

        Args:
            rankings: List of ranked doc_id lists
            k: Constant to prevent high ranks from dominating

        Returns:
            {doc_id: rrf_score}
        """
        rrf_scores: Dict[str, float] = {}

        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (k + rank)

        return rrf_scores

    @staticmethod
    def max_score_fusion(
        scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Take maximum score from any source.

        Useful when different sources are good for different query types.
        """
        all_docs: Set[str] = set()
        for source_scores in scores.values():
            all_docs.update(source_scores.keys())

        combined = {}
        for doc_id in all_docs:
            max_score = 0.0
            for source_scores in scores.values():
                if doc_id in source_scores:
                    max_score = max(max_score, source_scores[doc_id])
            combined[doc_id] = max_score

        return combined


# =============================================================================
# Hybrid Retriever
# =============================================================================

class HybridRetriever:
    """
    Hybrid retriever combining semantic and keyword search.

    Retrieval formula:
    Score(d) = α × semantic_sim + β × bm25_score + γ × graph_score + δ × recency + ε × source_text

    v1.3: Supports adaptive weights based on query type detection.
    v1.3: Supports entity-centric boosting for improved entity retrieval.
    v1.4: Supports source text search for original message matching.
    """

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        keyword_index: Optional[InvertedIndex] = None,
        use_adaptive_weights: bool = True,
        use_entity_boost: bool = True,
        entity_boost_factor: float = 1.5
    ):
        """
        Initialize hybrid retriever.

        Args:
            config: Retrieval configuration (used as base/default config)
            keyword_index: Pre-built keyword index (will be created if not provided)
            use_adaptive_weights: If True, automatically adjusts weights based on query type
            use_entity_boost: If True, boosts documents matching query entities
            entity_boost_factor: Boost multiplier for entity matches (1.0 = no boost)
        """
        self.config = config or RetrievalConfig()
        self.config.validate()

        self.keyword_index = keyword_index or InvertedIndex()

        # v1.3: Adaptive weight configuration
        self.use_adaptive_weights = use_adaptive_weights
        self._query_type_detector = QueryTypeDetector() if use_adaptive_weights else None

        # v1.3: Entity-centric indexing
        self.use_entity_boost = use_entity_boost
        self.entity_boost_factor = entity_boost_factor
        self._entity_index = EntityCentricIndex() if use_entity_boost else None

        # v1.4: Source text index for searching original messages
        self._source_text_index = InvertedIndex()

        # Node cache
        self._nodes: Dict[str, MemoryNode] = {}

        # Graph structure for expansion
        self._edges: Dict[str, List[Tuple[str, float]]] = {}  # source_id -> [(target_id, weight)]

        # Embeddings cache
        self._embeddings: Dict[str, np.ndarray] = {}

    def add_node(
        self,
        node: MemoryNode,
        embedding: Optional[List[float]] = None
    ) -> None:
        """
        Add a node to the retriever.

        Args:
            node: MemoryNode to add
            embedding: Optional pre-computed embedding
        """
        self._nodes[node.id] = node

        # Add to keyword index
        self.keyword_index.add_document(
            doc_id=node.id,
            content=node.content,
            metadata={
                "domain": node.domain,
                "weight": node.weight,
                "node_type": node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type)
            }
        )

        # v1.3: Add to entity index for entity-centric boosting
        if self._entity_index:
            self._entity_index.add_document(
                doc_id=node.id,
                content=node.content,
                metadata={"domain": node.domain}
            )

        # v1.4: Add source_text to source text index if available
        if node.source_text:
            self._source_text_index.add_document(
                doc_id=node.id,
                content=node.source_text,
                metadata={
                    "source_index": node.source_index,
                    "source_speaker": node.source_speaker or "unknown"
                }
            )

        # Store embedding
        if embedding:
            self._embeddings[node.id] = np.array(embedding)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        weight: float = 0.5
    ) -> None:
        """Add an edge for graph expansion."""
        if source_id not in self._edges:
            self._edges[source_id] = []
        self._edges[source_id].append((target_id, weight))

    def build_from_nodes(
        self,
        nodes: List[MemoryNode],
        embeddings: Optional[Dict[str, List[float]]] = None,
        edges: Optional[List[MemoryEdge]] = None
    ) -> None:
        """
        Build the retriever from a list of nodes.

        Args:
            nodes: List of MemoryNode objects
            embeddings: Optional {node_id: embedding}
            edges: Optional list of edges
        """
        for node in nodes:
            emb = embeddings.get(node.id) if embeddings else None
            self.add_node(node, emb)

        if edges:
            for edge in edges:
                self.add_edge(edge.source_id, edge.target_id, edge.weight)

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        intent: Optional[Intent] = None,
        top_k: Optional[int] = None
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid retrieval.

        v1.3: Uses adaptive weights based on detected query type when enabled.

        Args:
            query: Search query
            query_embedding: Optional pre-computed query embedding
            intent: Optional intent for domain filtering
            top_k: Number of results (uses config default if not specified)

        Returns:
            List of HybridSearchResult sorted by score
        """
        top_k = top_k or self.config.top_k
        fetch_count = top_k * self.config.over_fetch_factor

        # v1.3: Detect query type and get adaptive config
        active_config = self.config
        detected_query_type = None

        if self.use_adaptive_weights and self._query_type_detector:
            query_type_result = self._query_type_detector.detect(query)
            detected_query_type = query_type_result.query_type
            active_config = AdaptiveRetrievalConfig.get_config_for_query_type(
                query_type_result.query_type,
                base_config=self.config
            )

        # 1. Keyword search
        keyword_results = self.keyword_index.bm25_search(
            query,
            top_k=fetch_count,
            k1=active_config.bm25_k1,
            b=active_config.bm25_b
        )
        keyword_scores = {r.doc_id: r.score for r in keyword_results}
        keyword_terms = {r.doc_id: r.matched_terms for r in keyword_results}

        # 2. Semantic search (if embedding provided)
        semantic_scores: Dict[str, float] = {}
        if query_embedding and self._embeddings:
            query_emb = np.array(query_embedding)
            for node_id, node_emb in self._embeddings.items():
                similarity = self._cosine_similarity(query_emb, node_emb)
                if similarity > 0:
                    semantic_scores[node_id] = similarity

        # 3. Graph expansion scores
        graph_scores = self._compute_graph_scores(
            set(keyword_scores.keys()) | set(semantic_scores.keys())
        )

        # 4. Recency scores
        recency_scores = self._compute_recency_scores()

        # 5. v1.4: Source text search (search original messages)
        source_text_scores: Dict[str, float] = {}
        if active_config.source_text_weight > 0 and self._source_text_index.count() > 0:
            source_results = self._source_text_index.bm25_search(
                query,
                top_k=fetch_count,
                k1=active_config.bm25_k1,
                b=active_config.bm25_b
            )
            source_text_scores = {r.doc_id: r.score for r in source_results}

        # 6. Combine scores using adaptive weights
        all_scores = {
            "semantic": semantic_scores,
            "keyword": keyword_scores,
            "graph": graph_scores,
            "recency": recency_scores,
            "source_text": source_text_scores  # v1.4
        }

        # Use adaptive config weights (different per query type)
        weights = {
            "semantic": active_config.semantic_weight,
            "keyword": active_config.keyword_weight,
            "graph": active_config.graph_weight,
            "recency": active_config.recency_weight,
            "source_text": active_config.source_text_weight  # v1.4
        }

        combined = ScoreFusion.linear_combination(all_scores, weights)

        # v1.3: Extract query entities for entity-centric boosting
        query_entities = set()
        if self.use_entity_boost and self._entity_index:
            query_entities = self._entity_index.extract_query_entities(query)

        # 6. Apply weight threshold, entity boost, and intent filtering
        filtered_results: List[HybridSearchResult] = []

        for node_id, score in combined.items():
            node = self._nodes.get(node_id)
            if not node:
                continue

            # Weight threshold
            if node.weight < self.config.min_weight:
                continue

            # v1.3: Entity-centric boosting
            if self.use_entity_boost and self._entity_index and query_entities:
                entity_boost = self._entity_index.compute_entity_boost(
                    node_id,
                    query_entities,
                    boost_factor=self.entity_boost_factor
                )
                score *= entity_boost

            # Intent filtering
            if intent and intent.distribution:
                intent_boost = intent.distribution.get(node.domain, 0.1)
                score *= (0.5 + 0.5 * intent_boost)  # Soft boost, not hard filter

            result = HybridSearchResult(
                node=node,
                final_score=score,
                semantic_score=semantic_scores.get(node_id, 0.0),
                keyword_score=keyword_scores.get(node_id, 0.0),
                graph_score=graph_scores.get(node_id, 0.0),
                matched_keywords=keyword_terms.get(node_id, []),
                query_type=detected_query_type.value if detected_query_type else None
            )
            filtered_results.append(result)

        # 7. Sort and limit
        filtered_results.sort(key=lambda x: x.final_score, reverse=True)
        return filtered_results[:top_k]

    def retrieve_with_expansion(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        intent: Optional[Intent] = None,
        top_k: Optional[int] = None,
        expansion_depth: Optional[int] = None
    ) -> List[HybridSearchResult]:
        """
        Retrieve with explicit graph expansion.

        First retrieves initial results, then expands via graph neighbors.
        """
        top_k = top_k or self.config.top_k
        expansion_depth = expansion_depth or self.config.expansion_depth

        # Initial retrieval
        initial_results = self.retrieve(query, query_embedding, intent, top_k * 2)

        if expansion_depth < 1 or not self._edges:
            return initial_results[:top_k]

        # Expand via graph
        expanded_nodes: Dict[str, HybridSearchResult] = {
            r.node.id: r for r in initial_results
        }

        for result in initial_results:
            neighbors = self._get_neighbors(result.node.id, expansion_depth)
            for neighbor_id, edge_weight in neighbors:
                if neighbor_id in expanded_nodes:
                    continue

                neighbor_node = self._nodes.get(neighbor_id)
                if not neighbor_node:
                    continue

                # Discounted score
                neighbor_score = result.final_score * edge_weight * self.config.edge_discount

                expanded_nodes[neighbor_id] = HybridSearchResult(
                    node=neighbor_node,
                    final_score=neighbor_score,
                    graph_score=neighbor_score,
                    expansion_path=[result.node.id]
                )

        # Re-sort all results
        all_results = list(expanded_nodes.values())
        all_results.sort(key=lambda x: x.final_score, reverse=True)
        return all_results[:top_k]

    def _compute_graph_scores(
        self,
        seed_nodes: Set[str]
    ) -> Dict[str, float]:
        """Compute graph-based scores from seed nodes."""
        scores: Dict[str, float] = {}

        for seed_id in seed_nodes:
            if seed_id not in self._edges:
                continue

            for neighbor_id, weight in self._edges[seed_id]:
                if neighbor_id not in scores:
                    scores[neighbor_id] = 0.0
                scores[neighbor_id] += weight * self.config.edge_discount

        # Normalize
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        return scores

    def _compute_recency_scores(self) -> Dict[str, float]:
        """Compute recency-based scores."""
        from datetime import datetime

        now = datetime.now()
        scores: Dict[str, float] = {}

        for node_id, node in self._nodes.items():
            if hasattr(node, 'last_accessed'):
                time_diff = (now - node.last_accessed).total_seconds()
                # Exponential decay with 7-day half-life
                decay = 0.5 ** (time_diff / (7 * 24 * 3600))
                scores[node_id] = decay
            else:
                scores[node_id] = 0.5  # Default

        return scores

    def _get_neighbors(
        self,
        node_id: str,
        depth: int
    ) -> List[Tuple[str, float]]:
        """Get neighbors up to specified depth."""
        if depth < 1:
            return []

        neighbors = []
        visited = {node_id}

        current_level = [(node_id, 1.0)]
        for _ in range(depth):
            next_level = []
            for current_id, current_weight in current_level:
                for neighbor_id, edge_weight in self._edges.get(current_id, []):
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)
                    combined_weight = current_weight * edge_weight
                    neighbors.append((neighbor_id, combined_weight))
                    next_level.append((neighbor_id, combined_weight))
            current_level = next_level

        return neighbors

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        stats = {
            "node_count": len(self._nodes),
            "edge_count": sum(len(v) for v in self._edges.values()),
            "embedding_count": len(self._embeddings),
            "keyword_index_stats": self.keyword_index.get_statistics(),
            "config": {
                "semantic_weight": self.config.semantic_weight,
                "keyword_weight": self.config.keyword_weight,
                "graph_weight": self.config.graph_weight,
                "recency_weight": self.config.recency_weight
            },
            # v1.3: Adaptive weights info
            "adaptive_weights_enabled": self.use_adaptive_weights,
            # v1.3: Entity-centric indexing info
            "entity_boost_enabled": self.use_entity_boost,
            "entity_boost_factor": self.entity_boost_factor,
        }

        if self.use_adaptive_weights:
            stats["adaptive_configs"] = {
                "factual": {
                    "semantic": AdaptiveRetrievalConfig.FACTUAL_CONFIG.semantic_weight,
                    "keyword": AdaptiveRetrievalConfig.FACTUAL_CONFIG.keyword_weight,
                    "graph": AdaptiveRetrievalConfig.FACTUAL_CONFIG.graph_weight,
                    "recency": AdaptiveRetrievalConfig.FACTUAL_CONFIG.recency_weight
                },
                "temporal": {
                    "semantic": AdaptiveRetrievalConfig.TEMPORAL_CONFIG.semantic_weight,
                    "keyword": AdaptiveRetrievalConfig.TEMPORAL_CONFIG.keyword_weight,
                    "graph": AdaptiveRetrievalConfig.TEMPORAL_CONFIG.graph_weight,
                    "recency": AdaptiveRetrievalConfig.TEMPORAL_CONFIG.recency_weight
                },
                "multi_hop": {
                    "semantic": AdaptiveRetrievalConfig.MULTI_HOP_CONFIG.semantic_weight,
                    "keyword": AdaptiveRetrievalConfig.MULTI_HOP_CONFIG.keyword_weight,
                    "graph": AdaptiveRetrievalConfig.MULTI_HOP_CONFIG.graph_weight,
                    "recency": AdaptiveRetrievalConfig.MULTI_HOP_CONFIG.recency_weight,
                    "expansion_depth": AdaptiveRetrievalConfig.MULTI_HOP_CONFIG.expansion_depth
                },
                "adversarial": {
                    "semantic": AdaptiveRetrievalConfig.ADVERSARIAL_CONFIG.semantic_weight,
                    "keyword": AdaptiveRetrievalConfig.ADVERSARIAL_CONFIG.keyword_weight,
                    "graph": AdaptiveRetrievalConfig.ADVERSARIAL_CONFIG.graph_weight,
                    "recency": AdaptiveRetrievalConfig.ADVERSARIAL_CONFIG.recency_weight
                }
            }

        # v1.3: Entity index stats
        if self.use_entity_boost and self._entity_index:
            stats["entity_index_stats"] = self._entity_index.get_statistics()

        return stats

    def save(self, directory: str) -> None:
        """Save retriever state to directory."""
        os.makedirs(directory, exist_ok=True)

        # Save keyword index
        self.keyword_index.save(os.path.join(directory, "keyword_index.json"))

        # Save embeddings
        if self._embeddings:
            embeddings_data = {
                node_id: emb.tolist()
                for node_id, emb in self._embeddings.items()
            }
            with open(os.path.join(directory, "embeddings.json"), 'w') as f:
                json.dump(embeddings_data, f)

        # Save edges
        edges_data = {
            source: [(target, weight) for target, weight in targets]
            for source, targets in self._edges.items()
        }
        with open(os.path.join(directory, "edges.json"), 'w') as f:
            json.dump(edges_data, f)

        # Save config
        config_data = {
            "semantic_weight": self.config.semantic_weight,
            "keyword_weight": self.config.keyword_weight,
            "graph_weight": self.config.graph_weight,
            "recency_weight": self.config.recency_weight,
            "top_k": self.config.top_k,
            "expansion_depth": self.config.expansion_depth
        }
        with open(os.path.join(directory, "config.json"), 'w') as f:
            json.dump(config_data, f, indent=2)

    def load(self, directory: str) -> bool:
        """Load retriever state from directory."""
        try:
            # Load keyword index
            keyword_path = os.path.join(directory, "keyword_index.json")
            if os.path.exists(keyword_path):
                self.keyword_index.load(keyword_path)

            # Load embeddings
            emb_path = os.path.join(directory, "embeddings.json")
            if os.path.exists(emb_path):
                with open(emb_path, 'r') as f:
                    embeddings_data = json.load(f)
                self._embeddings = {
                    k: np.array(v) for k, v in embeddings_data.items()
                }

            # Load edges
            edges_path = os.path.join(directory, "edges.json")
            if os.path.exists(edges_path):
                with open(edges_path, 'r') as f:
                    edges_data = json.load(f)
                self._edges = {
                    source: [(t, w) for t, w in targets]
                    for source, targets in edges_data.items()
                }

            # Load config
            config_path = os.path.join(directory, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)

            return True

        except Exception as e:
            print(f"Failed to load retriever state: {e}")
            return False


# =============================================================================
# Factory Functions
# =============================================================================

def create_hybrid_retriever(
    nodes: List[MemoryNode],
    embeddings: Optional[Dict[str, List[float]]] = None,
    edges: Optional[List[MemoryEdge]] = None,
    config: Optional[RetrievalConfig] = None
) -> HybridRetriever:
    """
    Create a hybrid retriever from nodes.

    Args:
        nodes: List of MemoryNode objects
        embeddings: Optional {node_id: embedding}
        edges: Optional list of MemoryEdge objects
        config: Optional retrieval configuration

    Returns:
        Configured HybridRetriever
    """
    retriever = HybridRetriever(config=config)
    retriever.build_from_nodes(nodes, embeddings, edges)
    return retriever
