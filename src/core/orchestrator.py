"""
PCM System Orchestrator
=======================
Central orchestrator that connects L1 (Perception), L2 (World Model), and L3 (Evolution).

This is the main entry point for interacting with the PCM system.
"""

import os
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.core.types import (
    MemoryNode, Intent, SurprisalPacket, EvictedData, NodeType
)
from src.layers.layer1_perception import PerceptionLayer
from src.layers.layer2_world_model import WeightedKnowledgeGraph
from src.layers.layer3_evolution import CognitiveEngine
from src.utils.llm_client import LLMClient, get_llm_client


class PCMSystem:
    """
    Proactive Cognitive Memory System

    Main orchestrator that manages the three-layer architecture:
    - L1: Perception & Working Memory
    - L2: Probabilistic World Model
    - L3: Cognitive Evolution Engine

    The system processes user interactions through a closed loop:
    1. L1 receives input, classifies intent, calculates surprisal
    2. L2 retrieves relevant context using intent-masked retrieval
    3. L3 evolves the world model based on surprisal levels
    """

    def __init__(
        self,
        graph_path: Optional[str] = None,
        chroma_path: Optional[str] = None,
        max_context_tokens: int = None,
        use_mock: bool = False
    ):
        """
        Initialize the PCM system.

        Args:
            graph_path: Path for graph persistence
            chroma_path: Path for vector DB persistence
            max_context_tokens: Maximum tokens in working memory
            use_mock: Use mock components for testing
        """
        self.use_mock = use_mock or config.USE_MOCK_LLM

        # Initialize Layer 2 first (it's the foundation)
        print("Initializing Layer 2: World Model...")
        self.world_model = WeightedKnowledgeGraph(
            graph_path=graph_path,
            chroma_path=chroma_path
        )

        # Initialize Layer 1
        print("Initializing Layer 1: Perception...")
        self.perception = PerceptionLayer(
            max_tokens=max_context_tokens,
            use_mock=self.use_mock
        )

        # Initialize Layer 3
        print("Initializing Layer 3: Cognitive Engine...")
        self.cognitive_engine = CognitiveEngine()

        # LLM client for response generation
        self.llm_client = get_llm_client()

        # Interaction counter
        self._interaction_count = 0

        print("PCM System initialized successfully!")

    async def process_input(
        self,
        user_input: str,
        generate_response: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user input through the complete PCM pipeline.

        Pipeline:
        1. L2 Retrieval: Get relevant context
        2. L1 Processing: Intent classification + surprisal calculation
        3. L3 Evolution: Update world model based on surprisal
        4. Response Generation: Generate assistant response (optional)

        Args:
            user_input: The user's message
            generate_response: Whether to generate an assistant response

        Returns:
            Dict containing processing results
        """
        self._interaction_count += 1
        timestamp = datetime.now()

        # === Step 1: Initial L2 Retrieval ===
        # First pass retrieval to get context for surprisal calculation
        initial_nodes, initial_scores = self.world_model.retrieve(
            query=user_input,
            intent=None,  # No intent yet
            top_k=config.RETRIEVAL_TOP_K
        )

        # === Step 2: L1 Processing ===
        # Intent classification and surprisal calculation
        evicted, intent, surprisal_packet = await self.perception.process_input(
            user_input=user_input,
            retrieved_nodes=initial_nodes,
            retrieval_scores=initial_scores
        )

        # === Step 3: Intent-Masked Re-retrieval ===
        # Now that we have intent, do a more targeted retrieval
        retrieval_nodes, retrieval_scores = self.world_model.retrieve_with_expansion(
            query=user_input,
            intent=intent,
            top_k=config.RETRIEVAL_TOP_K,
            expansion_depth=1
        )

        # === Step 4: L3 Evolution ===
        evolution_result = await self.cognitive_engine.process_packet(
            packet=surprisal_packet,
            knowledge_graph=self.world_model
        )

        # === Step 5: Process Evicted Data (if any) ===
        if evicted and evicted.turns:
            await self._process_evicted_data(evicted)

        # === Step 6: Response Generation (optional) ===
        response = None
        if generate_response:
            response = await self._generate_response(
                user_input=user_input,
                context=retrieval_nodes,
                intent=intent
            )
            # Add response to working memory
            self.perception.add_response(response)

        # === Step 7: Save State ===
        self.world_model.save()

        return {
            "interaction_id": self._interaction_count,
            "timestamp": timestamp.isoformat(),
            "user_input": user_input,
            "intent": {
                "label": intent.label,
                "confidence": intent.confidence,
                "distribution": intent.distribution
            },
            "surprisal": {
                "raw": surprisal_packet.raw_score,
                "effective": surprisal_packet.effective_score,
                "entropy": surprisal_packet.retrieval_entropy,
                "level": surprisal_packet.get_surprise_level(
                    config.THETA_LOW, config.THETA_HIGH
                )
            },
            "evolution": {
                "level": evolution_result["level"],
                "agent": evolution_result["agent"]
            },
            "retrieval": {
                "num_nodes": len(retrieval_nodes),
                "top_contents": [n.content[:100] for n in retrieval_nodes[:3]]
            },
            "response": response,
            "evicted_data": evicted.to_text() if evicted else None
        }

    async def _process_evicted_data(self, evicted: EvictedData) -> None:
        """
        Process data evicted from working memory.

        Extract knowledge and add to L2.
        """
        # Extract entities and facts from evicted content
        for turn in evicted.turns:
            if turn.role == "user":
                # Create a node for user statement
                node = MemoryNode(
                    content=turn.content,
                    node_type=NodeType.FACT,
                    domain="General",  # Could use intent classification
                    weight=0.6
                )
                self.world_model.add_node(node)

    async def _generate_response(
        self,
        user_input: str,
        context: List[MemoryNode],
        intent: Intent
    ) -> str:
        """
        Generate an assistant response using retrieved context.
        """
        # Build context string
        context_str = "\n".join([
            f"- {node.content} (confidence: {node.weight:.2f})"
            for node in context[:5]
        ]) if context else "No relevant context available."

        prompt = f"""You are a helpful assistant with access to memory about the user.

Relevant Memory Context:
{context_str}

Current Intent: {intent.label} (confidence: {intent.confidence:.2f})

User Message: {user_input}

Respond helpfully, utilizing relevant context when appropriate. If the context helps answer the question or personalize the response, use it naturally."""

        try:
            response = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant with long-term memory capabilities."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response["content"]
        except Exception as e:
            return f"I apologize, I encountered an error: {str(e)}"

    async def add_knowledge(
        self,
        content: str,
        node_type: NodeType = NodeType.FACT,
        domain: str = "General",
        weight: float = 0.7
    ) -> str:
        """
        Directly add knowledge to the world model.

        Args:
            content: The knowledge content
            node_type: Type of node
            domain: Intent domain
            weight: Initial confidence weight

        Returns:
            Node ID
        """
        node = MemoryNode(
            content=content,
            node_type=node_type,
            domain=domain,
            weight=weight
        )
        return self.world_model.add_node(node)

    def query_knowledge(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query the world model for relevant knowledge.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of matching nodes with scores
        """
        nodes, scores = self.world_model.retrieve(query, top_k=top_k)
        return [
            {
                "content": node.content,
                "type": node.node_type.value if hasattr(node.node_type, 'value') else node.node_type,
                "domain": node.domain,
                "weight": node.weight,
                "score": score
            }
            for node, score in zip(nodes, scores)
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        """
        return {
            "interactions": self._interaction_count,
            "perception": self.perception.get_statistics(),
            "world_model": self.world_model.get_statistics(),
            "cognitive_engine": self.cognitive_engine.get_statistics()
        }

    def get_context(self) -> str:
        """Get current working memory context."""
        return self.perception.get_context()


# Convenience function for quick initialization
def create_pcm_system(
    data_dir: Optional[str] = None,
    use_mock: bool = False
) -> PCMSystem:
    """
    Create a PCM system with default paths.

    Args:
        data_dir: Base directory for data storage
        use_mock: Use mock components for testing

    Returns:
        Initialized PCMSystem
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    os.makedirs(data_dir, exist_ok=True)

    return PCMSystem(
        graph_path=os.path.join(data_dir, "knowledge_graph.gml"),
        chroma_path=os.path.join(data_dir, "chroma_db"),
        use_mock=use_mock
    )
