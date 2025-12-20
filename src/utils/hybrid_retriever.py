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
            "weight": self.node.weight
        }


@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval."""
    # Fusion weights
    semantic_weight: float = 0.4    # α: weight for embedding similarity
    keyword_weight: float = 0.3     # β: weight for BM25 keyword score
    graph_weight: float = 0.2       # γ: weight for graph expansion score
    recency_weight: float = 0.1     # δ: weight for recency bonus

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
        total = self.semantic_weight + self.keyword_weight + self.graph_weight + self.recency_weight
        if abs(total - 1.0) > 0.01:
            # Normalize weights
            self.semantic_weight /= total
            self.keyword_weight /= total
            self.graph_weight /= total
            self.recency_weight /= total


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
    Score(d) = α × semantic_sim + β × bm25_score + γ × graph_score + δ × recency
    """

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        keyword_index: Optional[InvertedIndex] = None
    ):
        """
        Initialize hybrid retriever.

        Args:
            config: Retrieval configuration
            keyword_index: Pre-built keyword index (will be created if not provided)
        """
        self.config = config or RetrievalConfig()
        self.config.validate()

        self.keyword_index = keyword_index or InvertedIndex()

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

        # 1. Keyword search
        keyword_results = self.keyword_index.bm25_search(
            query,
            top_k=fetch_count,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b
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

        # 5. Combine scores
        all_scores = {
            "semantic": semantic_scores,
            "keyword": keyword_scores,
            "graph": graph_scores,
            "recency": recency_scores
        }

        weights = {
            "semantic": self.config.semantic_weight,
            "keyword": self.config.keyword_weight,
            "graph": self.config.graph_weight,
            "recency": self.config.recency_weight
        }

        combined = ScoreFusion.linear_combination(all_scores, weights)

        # 6. Apply weight threshold and intent filtering
        filtered_results: List[HybridSearchResult] = []

        for node_id, score in combined.items():
            node = self._nodes.get(node_id)
            if not node:
                continue

            # Weight threshold
            if node.weight < self.config.min_weight:
                continue

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
                matched_keywords=keyword_terms.get(node_id, [])
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
        return {
            "node_count": len(self._nodes),
            "edge_count": sum(len(v) for v in self._edges.values()),
            "embedding_count": len(self._embeddings),
            "keyword_index_stats": self.keyword_index.get_statistics(),
            "config": {
                "semantic_weight": self.config.semantic_weight,
                "keyword_weight": self.config.keyword_weight,
                "graph_weight": self.config.graph_weight,
                "recency_weight": self.config.recency_weight
            }
        }

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
