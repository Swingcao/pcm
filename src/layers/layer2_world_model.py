"""
Layer 2: Probabilistic World Model
==================================
Weighted Knowledge Graph with vector index for intent-masked retrieval.

This layer maintains:
1. NetworkX DiGraph for structural relationships
2. JSON-based vector storage for semantic retrieval (replaces ChromaDB)
3. Soft update mechanisms for Bayesian confidence weights

All data is stored locally in JSON format for easy inspection and portability.
"""

import os
import json
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.core.types import (
    MemoryNode, MemoryEdge, Intent, NodeType, HypothesisNode
)
from src.utils.math_utils import (
    reinforcement_update, decay_update, compute_retrieval_score,
    cosine_similarity, compute_retrieval_entropy
)
from src.utils.json_storage import JSONVectorStore, ResultsManager


# Global embedding model instance (lazy loaded)
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Get or create a shared embedding model instance.

    Returns:
        SentenceTransformer model
    """
    global _embedding_model
    if _embedding_model is None:
        device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_ID).to(device)
    return _embedding_model


class WeightedKnowledgeGraph:
    """
    Layer 2: Probabilistic World Model

    Maintains a weighted knowledge graph G_t = (V_t, E_t) with:
    - Nodes: Entities, Attributes, Hypotheses
    - Edges: Weighted relationships with confidence scores
    - Vector Index: For semantic similarity search (JSON-based local storage)

    All data is stored locally in JSON format in the unified results folder.
    """

    def __init__(
        self,
        graph_path: Optional[str] = None,
        vector_store_path: Optional[str] = None,
        embedding_model: Optional[SentenceTransformer] = None,
        results_base: Optional[str] = None
    ):
        """
        Initialize the knowledge graph.

        Args:
            graph_path: Path to save/load the NetworkX graph (JSON format)
            vector_store_path: Path for JSON vector storage
            embedding_model: SentenceTransformer model for embeddings
            results_base: Base path for unified results folder
        """
        # Initialize results manager for unified storage
        self.results_manager = ResultsManager(results_base or config.RESULTS_DIR)

        # Set paths for graph and vector store
        self.graph_path = graph_path or self.results_manager.get_path(
            'knowledge_graphs', 'knowledge_graph.json'
        )
        self.vector_store_path = vector_store_path or self.results_manager.folders['vector_stores']

        # Initialize NetworkX DiGraph
        self.graph = nx.DiGraph()

        # Initialize embedding model
        if embedding_model is None:
            device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_ID).to(device)
        else:
            self.embedding_model = embedding_model

        # Initialize JSON-based vector store (replaces ChromaDB)
        self.vector_store = JSONVectorStore(
            storage_path=self.vector_store_path,
            collection_name="pcm_nodes"
        )

        # In-memory node cache for quick access
        self._node_cache: Dict[str, MemoryNode] = {}

        # Load existing graph if available
        self._load_graph()

    def add_node(self, node: MemoryNode) -> str:
        """
        Add a node to both the graph and vector database.

        Args:
            node: MemoryNode to add

        Returns:
            Node ID
        """
        # Add to NetworkX graph
        self.graph.add_node(
            node.id,
            **node.to_graph_dict()
        )

        # Generate embedding
        embedding = self.embedding_model.encode(node.content).tolist()

        # Add to JSON vector store (replaces ChromaDB)
        self.vector_store.upsert(
            ids=[node.id],
            embeddings=[embedding],
            metadatas=[{
                "content": node.content,
                "node_type": node.node_type.value if isinstance(node.node_type, NodeType) else node.node_type,
                "domain": node.domain,
                "weight": node.weight,
                "created_at": node.created_at.isoformat(),
                "last_accessed": node.last_accessed.isoformat()
            }],
            documents=[node.content]
        )

        # Update cache
        self._node_cache[node.id] = node

        return node.id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 0.5
    ) -> MemoryEdge:
        """
        Add a weighted edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation: Semantic relation type
            weight: Initial confidence weight

        Returns:
            Created MemoryEdge
        """
        edge = MemoryEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight
        )

        self.graph.add_edge(
            source_id,
            target_id,
            **edge.to_graph_dict()
        )

        return edge

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a node by ID."""
        if node_id in self._node_cache:
            return self._node_cache[node_id]

        if node_id not in self.graph:
            return None

        node_data = self.graph.nodes[node_id]
        node = MemoryNode(
            id=node_id,
            content=node_data.get("content", ""),
            node_type=NodeType(node_data.get("node_type", "entity")),
            domain=node_data.get("domain", "General"),
            weight=node_data.get("weight", 0.5)
        )
        self._node_cache[node_id] = node
        return node

    def soft_update_weight(
        self,
        node_id: str,
        new_weight: float
    ) -> bool:
        """
        Soft update a node's weight (no deletion).

        Args:
            node_id: Node ID to update
            new_weight: New weight value

        Returns:
            True if successful
        """
        if node_id not in self.graph:
            return False

        # Update graph
        self.graph.nodes[node_id]["weight"] = new_weight
        self.graph.nodes[node_id]["last_accessed"] = datetime.now().isoformat()

        # Update JSON vector store metadata
        try:
            existing = self.vector_store.get(ids=[node_id])
            if existing["metadatas"]:
                metadata = existing["metadatas"][0]
                metadata["weight"] = new_weight
                metadata["last_accessed"] = datetime.now().isoformat()
                self.vector_store.update(
                    ids=[node_id],
                    metadatas=[metadata]
                )
        except Exception:
            pass  # Vector store update is best-effort

        # Update cache
        if node_id in self._node_cache:
            self._node_cache[node_id].weight = new_weight
            self._node_cache[node_id].last_accessed = datetime.now()

        return True

    def reinforce_node(self, node_id: str, learning_rate: float = None) -> float:
        """
        Maintenance Agent: Reinforce a node's weight.

        w_{t+1} = w_t + η * (1 - w_t)

        Returns:
            New weight
        """
        if learning_rate is None:
            learning_rate = config.ETA

        node = self.get_node(node_id)
        if not node:
            return 0.0

        new_weight = reinforcement_update(node.weight, learning_rate)
        self.soft_update_weight(node_id, new_weight)
        return new_weight

    def decay_node(
        self,
        node_id: str,
        surprisal_score: float,
        decay_factor: float = None
    ) -> float:
        """
        Correction Agent: Decay a node's weight due to conflict.

        w_{t+1} = w_t * exp(-β * S_eff)

        Returns:
            New weight
        """
        if decay_factor is None:
            decay_factor = config.BETA

        node = self.get_node(node_id)
        if not node:
            return 0.0

        new_weight = decay_update(node.weight, surprisal_score, decay_factor)
        self.soft_update_weight(node_id, new_weight)
        return new_weight

    def retrieve(
        self,
        query: str,
        intent: Optional[Intent] = None,
        top_k: int = None,
        min_weight: float = 0.1
    ) -> Tuple[List[MemoryNode], List[float]]:
        """
        Intent-masked retrieval from the knowledge graph.

        Score(ε_k) = sim(emb(u_t), emb(ε_k)) * P(d(ε_k) | u_t) * w_k

        Args:
            query: Query text
            intent: Intent classification result
            top_k: Number of results to return
            min_weight: Minimum weight threshold

        Returns:
            Tuple of (nodes, scores)
        """
        if top_k is None:
            top_k = config.RETRIEVAL_TOP_K

        # Get query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Query JSON vector store
        results = self.vector_store.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 3, self.vector_store.count() or 1),  # Over-fetch for filtering
            include=["metadatas", "distances", "documents"]
        )

        if not results["ids"] or not results["ids"][0]:
            return [], []

        nodes = []
        scores = []

        for idx, node_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][idx]
            distance = results["distances"][0][idx]

            # Convert distance to similarity (distance is 1 - similarity for cosine)
            similarity = 1.0 - distance

            # Get node weight
            weight = metadata.get("weight", 0.5)
            if weight < min_weight:
                continue

            # Compute intent relevance
            node_domain = metadata.get("domain", "General")
            if intent and intent.distribution:
                intent_relevance = intent.distribution.get(node_domain, 0.1)
            else:
                intent_relevance = 1.0

            # Compute final score
            score = compute_retrieval_score(similarity, intent_relevance, weight)

            # Create node object
            node = MemoryNode(
                id=node_id,
                content=metadata.get("content", results["documents"][0][idx]),
                node_type=NodeType(metadata.get("node_type", "entity")),
                domain=node_domain,
                weight=weight
            )

            nodes.append(node)
            scores.append(score)

        # Sort by score and limit
        sorted_pairs = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)
        nodes = [p[0] for p in sorted_pairs[:top_k]]
        scores = [p[1] for p in sorted_pairs[:top_k]]

        # Update access times for retrieved nodes
        for node in nodes:
            self.soft_update_weight(node.id, node.weight)  # Just updates timestamp

        return nodes, scores

    def get_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        min_weight: float = 0.3
    ) -> List[Tuple[MemoryNode, MemoryEdge]]:
        """
        Get neighboring nodes up to a certain depth.

        Prioritizes high-weight edges.

        Returns:
            List of (node, edge) tuples
        """
        if node_id not in self.graph:
            return []

        results = []
        visited = {node_id}

        def explore(current_id: str, current_depth: int):
            if current_depth > depth:
                return

            for neighbor_id in self.graph.neighbors(current_id):
                if neighbor_id in visited:
                    continue

                edge_data = self.graph.edges[current_id, neighbor_id]
                edge_weight = edge_data.get("weight", 0.5)

                if edge_weight < min_weight:
                    continue

                visited.add(neighbor_id)

                neighbor_node = self.get_node(neighbor_id)
                if neighbor_node:
                    edge = MemoryEdge(
                        source_id=current_id,
                        target_id=neighbor_id,
                        relation=edge_data.get("relation", "related"),
                        weight=edge_weight
                    )
                    results.append((neighbor_node, edge))
                    explore(neighbor_id, current_depth + 1)

        explore(node_id, 1)
        return results

    def retrieve_with_expansion(
        self,
        query: str,
        intent: Optional[Intent] = None,
        top_k: int = None,
        expansion_depth: int = 1
    ) -> Tuple[List[MemoryNode], List[float]]:
        """
        Retrieve nodes and expand with 1-hop neighbors.

        Returns:
            Tuple of (nodes, scores) including expanded neighbors
        """
        # Initial retrieval
        nodes, scores = self.retrieve(query, intent, top_k)

        if not nodes or expansion_depth < 1:
            return nodes, scores

        # Expand with neighbors
        expanded_nodes = list(nodes)
        expanded_scores = list(scores)
        seen_ids = {n.id for n in nodes}

        for node, score in zip(nodes, scores):
            neighbors = self.get_neighbors(node.id, depth=expansion_depth)
            for neighbor_node, edge in neighbors:
                if neighbor_node.id not in seen_ids:
                    seen_ids.add(neighbor_node.id)
                    # Neighbor score is discounted by edge weight
                    neighbor_score = score * edge.weight * 0.5
                    expanded_nodes.append(neighbor_node)
                    expanded_scores.append(neighbor_score)

        # Re-sort by score
        sorted_pairs = sorted(
            zip(expanded_nodes, expanded_scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]

    def promote_hypothesis_to_fact(self, hypothesis_id: str) -> bool:
        """
        Promote a hypothesis node to a fact node.

        Called when a hypothesis receives sufficient supporting evidence.
        """
        if hypothesis_id not in self.graph:
            return False

        node_data = self.graph.nodes[hypothesis_id]
        if node_data.get("node_type") != NodeType.HYPOTHESIS.value:
            return False

        # Update node type
        self.graph.nodes[hypothesis_id]["node_type"] = NodeType.FACT.value
        self.graph.nodes[hypothesis_id]["weight"] = min(
            1.0,
            node_data.get("weight", 0.5) + 0.3
        )

        # Update JSON vector store
        try:
            existing = self.vector_store.get(ids=[hypothesis_id])
            if existing["metadatas"]:
                metadata = existing["metadatas"][0]
                metadata["node_type"] = NodeType.FACT.value
                metadata["weight"] = self.graph.nodes[hypothesis_id]["weight"]
                self.vector_store.update(
                    ids=[hypothesis_id],
                    metadatas=[metadata]
                )
        except Exception:
            pass

        # Update cache
        if hypothesis_id in self._node_cache:
            self._node_cache[hypothesis_id].node_type = NodeType.FACT
            self._node_cache[hypothesis_id].weight = self.graph.nodes[hypothesis_id]["weight"]

        return True

    def prune_low_weight_nodes(self, threshold: float = 0.05) -> int:
        """
        Remove nodes with very low weights (soft delete from active use).

        Note: We don't actually delete, just mark as inactive.

        Returns:
            Number of nodes pruned
        """
        pruned = 0
        for node_id in list(self.graph.nodes()):
            weight = self.graph.nodes[node_id].get("weight", 0.5)
            if weight < threshold:
                # Mark as inactive rather than delete
                self.graph.nodes[node_id]["inactive"] = True
                pruned += 1
        return pruned

    def save(self) -> None:
        """Save the graph to disk in both JSON and GML formats."""
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)

        # === Save JSON format (for easy viewing and analysis) ===
        graph_data = json_graph.node_link_data(self.graph)
        save_data = {
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges()
            },
            "graph": graph_data
        }
        with open(self.graph_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)

        # === Save GML format (original format for NetworkX compatibility) ===
        gml_path = self.graph_path.replace('.json', '.gml')
        try:
            nx.write_gml(self.graph, gml_path)
        except Exception as e:
            print(f"Warning: Failed to save GML format: {e}")

        # Also save intermediate state for debugging
        self.results_manager.save_intermediate(
            name="graph_snapshot",
            data={
                "statistics": self.get_statistics(),
                "node_types": self._count_node_types()
            }
        )

    def _load_graph(self) -> None:
        """Load the graph from disk if exists (supports both JSON and legacy GML)."""
        # Try JSON format first
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if "graph" in data:
                    self.graph = json_graph.node_link_graph(data["graph"])
                else:
                    # Direct graph data without metadata wrapper
                    self.graph = json_graph.node_link_graph(data)

                print(f"Loaded graph with {self.graph.number_of_nodes()} nodes from JSON")
                return
            except Exception as e:
                print(f"Failed to load JSON graph: {e}")

        # Try legacy GML format for backwards compatibility
        legacy_gml_path = self.graph_path.replace('.json', '.gml')
        if os.path.exists(legacy_gml_path):
            try:
                self.graph = nx.read_gml(legacy_gml_path)
                print(f"Loaded graph with {self.graph.number_of_nodes()} nodes from legacy GML")
                # Save in new JSON format
                self.save()
                return
            except Exception as e:
                print(f"Failed to load GML graph: {e}")

        # Initialize empty graph
        self.graph = nx.DiGraph()

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": self._count_node_types(),
            "avg_weight": self._average_weight(),
            "vector_count": self.vector_store.count()
        }

    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type."""
        counts = {}
        for node_id in self.graph.nodes():
            node_type = self.graph.nodes[node_id].get("node_type", "entity")
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts

    def _average_weight(self) -> float:
        """Calculate average node weight."""
        if self.graph.number_of_nodes() == 0:
            return 0.0
        weights = [
            self.graph.nodes[n].get("weight", 0.5)
            for n in self.graph.nodes()
        ]
        return sum(weights) / len(weights)
