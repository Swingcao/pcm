"""
Layer 3: Cognitive Evolution Engine
====================================
Async processing layer that evolves L2 based on surprisal levels.

Components:
1. SurpriseDispatcher - Routes packets to appropriate agents
2. CorrectionAgent - Handles high surprise (conflicts)
3. ProfilingAgent - Handles medium surprise (hypothesis generation)
4. MaintenanceAgent - Handles low surprise (reinforcement)
"""

import os
import json
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.core.types import (
    SurprisalPacket, MemoryNode, MemoryEdge, NodeType,
    HypothesisNode, CorrectionResult, ProfilingResult, MaintenanceResult
)
from src.utils.llm_client import LLMClient, get_llm_client
from src.utils.math_utils import initialize_hypothesis_weight


class CorrectionAgent:
    """
    Correction Agent: Handles High Surprise (S_eff > θ_high)

    Scenario: Severe conflict with existing knowledge.

    Actions:
    1. Diagnose: Analyze conflict source
    2. Decay: Reduce weights of conflicting nodes
    3. Overwrite: Create new nodes with high initial weight
    """

    CORRECTION_PROMPT = """You are analyzing a conflict between new user input and existing memory.

User Input (New Information):
{user_input}

Retrieved Memory (Existing Knowledge):
{retrieved_context}

The new input has a HIGH surprisal score ({surprisal:.2f}), indicating significant conflict.

Analyze:
1. What is the nature of the conflict?
2. Which existing memories should be marked as outdated?
3. What new knowledge should be added?

Respond with JSON:
{{
    "diagnosis": "explanation of the conflict",
    "outdated_memories": ["id1", "id2"],
    "new_facts": [
        {{"content": "fact content", "relation_to_outdated": "id or null"}}
    ],
    "conflict_type": "preference_change" | "factual_update" | "contradiction"
}}"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        decay_factor: float = None
    ):
        self.llm_client = llm_client or get_llm_client()
        self.decay_factor = decay_factor or config.BETA

    async def process(
        self,
        packet: SurprisalPacket,
        knowledge_graph  # WeightedKnowledgeGraph - avoid circular import
    ) -> CorrectionResult:
        """
        Process a high-surprise packet.

        Args:
            packet: SurprisalPacket with high S_eff
            knowledge_graph: L2 world model

        Returns:
            CorrectionResult with actions taken
        """
        # Build context string
        retrieved_context = "\n".join([
            f"[{i}] {ctx}" for i, ctx in enumerate(packet.retrieved_context)
        ])

        prompt = self.CORRECTION_PROMPT.format(
            user_input=packet.content,
            retrieved_context=retrieved_context or "(No relevant memories found)",
            surprisal=packet.effective_score
        )

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
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

            # Apply corrections
            decay_applied = {}
            new_nodes = []

            # Decay outdated memories
            for node_id in data.get("outdated_memories", []):
                if node_id in packet.retrieved_node_ids:
                    idx = packet.retrieved_node_ids.index(node_id)
                    actual_id = packet.retrieved_node_ids[idx]
                    new_weight = knowledge_graph.decay_node(
                        actual_id,
                        packet.effective_score,
                        self.decay_factor
                    )
                    decay_applied[actual_id] = new_weight

            # Create new nodes for facts
            for fact in data.get("new_facts", []):
                new_node = MemoryNode(
                    content=fact["content"],
                    node_type=NodeType.FACT,
                    domain=packet.intent.label if packet.intent else "General",
                    weight=0.8  # High initial weight for corrections
                )
                knowledge_graph.add_node(new_node)
                new_nodes.append(new_node)

                # If related to outdated node, create edge
                related_id = fact.get("relation_to_outdated")
                if related_id and related_id in packet.retrieved_node_ids:
                    knowledge_graph.add_edge(
                        new_node.id,
                        related_id,
                        "supersedes",
                        weight=0.9
                    )

            return CorrectionResult(
                conflicting_node_ids=packet.retrieved_node_ids,
                decay_applied=decay_applied,
                new_nodes_created=new_nodes,
                diagnosis=data.get("diagnosis", "")
            )

        except Exception as e:
            print(f"Correction agent failed: {e}")
            return CorrectionResult(
                diagnosis=f"Error during correction: {str(e)}"
            )


class ProfilingAgent:
    """
    Profiling Agent: Handles Medium Surprise (θ_low < S_eff ≤ θ_high)

    Scenario: Novel information without direct conflict.

    Core Innovation:
    - Generate hypotheses about user's latent goals
    - Hypotheses are validated implicitly through future observations
    """

    PROFILING_PROMPT = """You are generating a hypothesis about the user based on novel information.

User Input (New Information):
{user_input}

Retrieved Memory (Context):
{retrieved_context}

The input has MEDIUM surprisal ({surprisal:.2f}) - it's novel but not conflicting.

Internal Monologue:
Think about what this new information might reveal about the user's:
- Current goals or projects
- Shifting interests
- Hidden preferences or skills

Generate a hypothesis that could explain this pattern.

Respond with JSON:
{{
    "reasoning": "Your internal monologue explaining the hypothesis",
    "hypothesis": "A concise hypothesis statement about the user",
    "confidence": 0.3-0.5,
    "domain": "relevant domain (Coding/Academic/Personal/etc.)",
    "evidence_keywords": ["key", "words", "that", "would", "support"]
}}"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        weight_min: float = None,
        weight_max: float = None
    ):
        self.llm_client = llm_client or get_llm_client()
        self.weight_min = weight_min or config.HYPOTHESIS_WEIGHT_MIN
        self.weight_max = weight_max or config.HYPOTHESIS_WEIGHT_MAX

    async def process(
        self,
        packet: SurprisalPacket,
        knowledge_graph  # WeightedKnowledgeGraph
    ) -> ProfilingResult:
        """
        Process a medium-surprise packet.

        Args:
            packet: SurprisalPacket with medium S_eff
            knowledge_graph: L2 world model

        Returns:
            ProfilingResult with generated hypothesis
        """
        retrieved_context = "\n".join([
            f"- {ctx}" for ctx in packet.retrieved_context
        ])

        prompt = self.PROFILING_PROMPT.format(
            user_input=packet.content,
            retrieved_context=retrieved_context or "(No prior relevant context)",
            surprisal=packet.effective_score
        )

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=600
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

            # Create hypothesis node
            hypothesis = HypothesisNode.create(
                content=data.get("hypothesis", f"Hypothesis based on: {packet.content[:50]}"),
                domain=data.get("domain", packet.intent.label if packet.intent else "General"),
                surprisal_score=packet.effective_score,
                weight_min=self.weight_min,
                weight_max=self.weight_max
            )

            # Store evidence keywords in metadata
            hypothesis.metadata["evidence_keywords"] = data.get("evidence_keywords", [])
            hypothesis.supporting_evidence = [packet.content]

            # Add to knowledge graph
            knowledge_graph.add_node(hypothesis)

            # Link to related retrieved nodes
            for node_id in packet.retrieved_node_ids[:3]:  # Top 3 related
                knowledge_graph.add_edge(
                    hypothesis.id,
                    node_id,
                    "derived_from",
                    weight=0.4
                )

            return ProfilingResult(
                hypothesis=hypothesis,
                reasoning=data.get("reasoning", "")
            )

        except Exception as e:
            print(f"Profiling agent failed: {e}")
            # Create a simple hypothesis as fallback
            hypothesis = HypothesisNode.create(
                content=f"User interest: {packet.content[:100]}",
                domain=packet.intent.label if packet.intent else "General",
                surprisal_score=packet.effective_score
            )
            knowledge_graph.add_node(hypothesis)

            return ProfilingResult(
                hypothesis=hypothesis,
                reasoning=f"Fallback hypothesis due to error: {str(e)}"
            )


class MaintenanceAgent:
    """
    Maintenance Agent: Handles Low Surprise (S_eff ≤ θ_low)

    Scenario: Information matches expectations (reinforcement).

    Actions:
    1. Boost: Increase weights of matching nodes
    2. Update Timestamp: Prevent forgetting
    3. Promote Hypotheses: If supported by new evidence
    """

    def __init__(
        self,
        learning_rate: float = None,
        promotion_threshold: float = 0.7
    ):
        self.learning_rate = learning_rate or config.ETA
        self.promotion_threshold = promotion_threshold

    async def process(
        self,
        packet: SurprisalPacket,
        knowledge_graph  # WeightedKnowledgeGraph
    ) -> MaintenanceResult:
        """
        Process a low-surprise packet.

        Args:
            packet: SurprisalPacket with low S_eff
            knowledge_graph: L2 world model

        Returns:
            MaintenanceResult with reinforcement actions
        """
        weight_updates = {}
        reinforced_ids = []
        promoted_hypotheses = []

        for node_id in packet.retrieved_node_ids:
            node = knowledge_graph.get_node(node_id)
            if not node:
                continue

            # Reinforce the node
            new_weight = knowledge_graph.reinforce_node(node_id, self.learning_rate)
            weight_updates[node_id] = new_weight
            reinforced_ids.append(node_id)

            # Check if hypothesis should be promoted to fact
            if node.node_type == NodeType.HYPOTHESIS:
                # Increment verification count (stored in metadata or graph)
                graph_node = knowledge_graph.graph.nodes.get(node_id, {})
                verification_count = graph_node.get("verification_count", 0) + 1
                knowledge_graph.graph.nodes[node_id]["verification_count"] = verification_count

                # Promote if weight exceeds threshold
                if new_weight >= self.promotion_threshold:
                    if knowledge_graph.promote_hypothesis_to_fact(node_id):
                        promoted_hypotheses.append(node_id)
                        print(f"Hypothesis promoted to fact: {node.content[:50]}...")

        return MaintenanceResult(
            reinforced_node_ids=reinforced_ids,
            weight_updates=weight_updates,
            hypotheses_promoted=promoted_hypotheses
        )


class SurpriseDispatcher:
    """
    Surprise Dispatcher: Routes packets to appropriate agents based on S_eff.

    - S_eff > θ_high -> Correction Agent
    - θ_low < S_eff ≤ θ_high -> Profiling Agent
    - S_eff ≤ θ_low -> Maintenance Agent
    """

    def __init__(
        self,
        theta_low: float = None,
        theta_high: float = None
    ):
        self.theta_low = theta_low or config.THETA_LOW
        self.theta_high = theta_high or config.THETA_HIGH

        self.correction_agent = CorrectionAgent()
        self.profiling_agent = ProfilingAgent()
        self.maintenance_agent = MaintenanceAgent()

    async def dispatch(
        self,
        packet: SurprisalPacket,
        knowledge_graph  # WeightedKnowledgeGraph
    ) -> Dict[str, Any]:
        """
        Dispatch packet to appropriate agent based on surprise level.

        Args:
            packet: SurprisalPacket from L1
            knowledge_graph: L2 world model

        Returns:
            Dict with agent type and result
        """
        level = packet.get_surprise_level(self.theta_low, self.theta_high)

        if level == "high":
            result = await self.correction_agent.process(packet, knowledge_graph)
            return {
                "level": "high",
                "agent": "correction",
                "result": result,
                "effective_score": packet.effective_score
            }

        elif level == "medium":
            result = await self.profiling_agent.process(packet, knowledge_graph)
            return {
                "level": "medium",
                "agent": "profiling",
                "result": result,
                "effective_score": packet.effective_score
            }

        else:  # low
            result = await self.maintenance_agent.process(packet, knowledge_graph)
            return {
                "level": "low",
                "agent": "maintenance",
                "result": result,
                "effective_score": packet.effective_score
            }


class CognitiveEngine:
    """
    Complete Layer 3 interface.

    Integrates the dispatcher and manages the cognitive evolution process.
    """

    def __init__(
        self,
        theta_low: float = None,
        theta_high: float = None
    ):
        self.dispatcher = SurpriseDispatcher(theta_low, theta_high)
        self._processing_history: List[Dict[str, Any]] = []

    async def process_packet(
        self,
        packet: SurprisalPacket,
        knowledge_graph
    ) -> Dict[str, Any]:
        """
        Process a surprisal packet through the cognitive engine.

        Args:
            packet: SurprisalPacket from L1
            knowledge_graph: L2 world model

        Returns:
            Processing result with agent info and actions
        """
        result = await self.dispatcher.dispatch(packet, knowledge_graph)

        # Record in history
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "content_preview": packet.content[:100],
            "raw_surprisal": packet.raw_score,
            "effective_surprisal": packet.effective_score,
            "retrieval_entropy": packet.retrieval_entropy,
            "level": result["level"],
            "agent": result["agent"]
        }
        self._processing_history.append(history_entry)

        # Keep history bounded
        if len(self._processing_history) > 1000:
            self._processing_history = self._processing_history[-500:]

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get cognitive engine statistics."""
        if not self._processing_history:
            return {
                "total_processed": 0,
                "level_distribution": {},
                "avg_surprisal": 0.0
            }

        levels = [h["level"] for h in self._processing_history]
        level_counts = {
            "low": levels.count("low"),
            "medium": levels.count("medium"),
            "high": levels.count("high")
        }

        avg_surprisal = sum(
            h["effective_surprisal"] for h in self._processing_history
        ) / len(self._processing_history)

        return {
            "total_processed": len(self._processing_history),
            "level_distribution": level_counts,
            "avg_surprisal": avg_surprisal,
            "recent_history": self._processing_history[-10:]
        }
