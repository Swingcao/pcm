"""
PCM Data Structure Definitions
==============================
Pydantic models for type-safe data structures used throughout the PCM system.
"""

from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    ENTITY = "entity"           # Concrete entities (e.g., "Python", "Machine Learning")
    ATTRIBUTE = "attribute"     # Attributes of entities (e.g., "skill_level: expert")
    HYPOTHESIS = "hypothesis"   # Hypotheses about user intent/preferences
    FACT = "fact"               # Verified facts


class MemoryNode(BaseModel):
    """
    A node in the weighted knowledge graph.

    Represents entities, attributes, or hypotheses stored in L2.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="The textual content of this node")
    node_type: NodeType = Field(default=NodeType.ENTITY, description="Type of the node")
    domain: str = Field(default="General", description="Intent domain this node belongs to")
    weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence weight")
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding of content")

    def update_access_time(self) -> None:
        """Update the last accessed timestamp."""
        self.last_accessed = datetime.now()

    def to_graph_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for NetworkX serialization."""
        # Handle both enum and string node_type (due to Pydantic use_enum_values)
        node_type_value = self.node_type.value if isinstance(self.node_type, NodeType) else self.node_type
        return {
            "content": self.content,
            "node_type": node_type_value,
            "domain": self.domain,
            "weight": self.weight,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "metadata": str(self.metadata),
            "has_embedding": self.embedding is not None
        }

    class Config:
        use_enum_values = True


class MemoryEdge(BaseModel):
    """
    An edge in the weighted knowledge graph.

    Represents relationships between nodes with confidence weights.
    Mathematical definition: ε_k = (v_i, v_j, rel, w_k, τ_k)
    """
    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")
    relation: str = Field(..., description="Semantic relation type")
    weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence weight P(True|History)")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_graph_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for NetworkX serialization."""
        return {
            "relation": self.relation,
            "weight": self.weight,
            "timestamp": self.timestamp.isoformat(),
            "metadata": str(self.metadata)
        }

    class Config:
        use_enum_values = True


class Intent(BaseModel):
    """
    Intent classification result from the Intent Router.

    P(I_t | u_t, Q_{t-1}) = Softmax(f_θ(u_t, Q_{t-1}))
    """
    label: str = Field(..., description="Primary intent domain label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Full probability distribution over all domains"
    )

    @classmethod
    def from_distribution(cls, distribution: Dict[str, float]) -> "Intent":
        """Create Intent from a probability distribution."""
        if not distribution:
            return cls(label="General", confidence=1.0, distribution={"General": 1.0})

        max_label = max(distribution, key=distribution.get)
        return cls(
            label=max_label,
            confidence=distribution[max_label],
            distribution=distribution
        )


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="The message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    token_count: int = Field(default=0, description="Estimated token count")


class EvictedData(BaseModel):
    """
    Data packet evicted from the sliding context queue.

    Sent to L3 for processing along with surprisal score.
    """
    turns: List[ConversationTurn] = Field(default_factory=list)
    raw_text: str = Field(default="", description="Concatenated raw text of evicted turns")
    evicted_at: datetime = Field(default_factory=datetime.now)

    def to_text(self) -> str:
        """Convert evicted turns to a single text string."""
        if self.raw_text:
            return self.raw_text
        return "\n".join([f"{t.role}: {t.content}" for t in self.turns])


class SurprisalPacket(BaseModel):
    """
    A packet containing evicted data with its surprisal score.

    This is the primary data structure passed from L1 to L3.
    Contains:
    - The evicted content from working memory
    - Raw surprisal score S_raw
    - Effective surprisal score S_eff (after entropy adjustment)
    - Retrieved context used for calculation
    - Semantic analysis metadata (when using semantic method)
    """
    content: str = Field(..., description="The evicted/new content being processed")
    raw_score: float = Field(default=0.0, description="Raw NLL-based surprisal S_raw")
    effective_score: float = Field(default=0.0, description="Entropy-adjusted surprisal S_eff")
    retrieval_entropy: float = Field(default=0.0, description="Entropy of retrieved context H(C)")
    retrieved_context: List[str] = Field(default_factory=list, description="Context retrieved from L2")
    retrieved_node_ids: List[str] = Field(default_factory=list, description="IDs of retrieved nodes")
    intent: Optional[Intent] = Field(default=None, description="Intent classification result")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Semantic analysis metadata (populated when using semantic surprisal method)
    semantic_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Semantic analysis details: relation, conflict_score, reasoning, conflicting_ids"
    )

    def get_surprise_level(self, theta_low: float, theta_high: float) -> str:
        """
        Determine the surprise level based on thresholds.

        Returns:
            "low" if S_eff <= theta_low (-> Maintenance)
            "medium" if theta_low < S_eff <= theta_high (-> Profiling)
            "high" if S_eff > theta_high (-> Correction)
        """
        if self.effective_score <= theta_low:
            return "low"
        elif self.effective_score <= theta_high:
            return "medium"
        else:
            return "high"

    def get_relation(self) -> Optional[str]:
        """Get the semantic relation type if available."""
        if self.semantic_analysis:
            return self.semantic_analysis.get("relation")
        return None

    def get_reasoning(self) -> Optional[str]:
        """Get the LLM reasoning for the surprisal classification."""
        if self.semantic_analysis:
            return self.semantic_analysis.get("reasoning")
        return None

    def get_conflicting_ids(self) -> List[str]:
        """Get IDs of conflicting memories if any."""
        if self.semantic_analysis:
            return self.semantic_analysis.get("conflicting_ids", [])
        return []


class HypothesisNode(MemoryNode):
    """
    A specialized node type for hypotheses generated by the Profiling Agent.

    Hypotheses are tentative beliefs about user intent/preferences that
    require implicit verification through future observations.
    """
    node_type: NodeType = Field(default=NodeType.HYPOTHESIS)
    hypothesis_content: str = Field(..., description="The hypothesis statement")
    initial_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    supporting_evidence: List[str] = Field(default_factory=list)
    verification_count: int = Field(default=0, description="Times this hypothesis was reinforced")

    @classmethod
    def create(
        cls,
        content: str,
        domain: str,
        surprisal_score: float,
        weight_min: float = 0.3,
        weight_max: float = 0.5
    ) -> "HypothesisNode":
        """
        Create a hypothesis node with weight initialized via sigmoid.

        w_init = σ(S_eff) mapped to [weight_min, weight_max]
        """
        import math
        # Sigmoid mapping
        sigmoid = 1 / (1 + math.exp(-surprisal_score))
        # Map to [weight_min, weight_max]
        weight = weight_min + sigmoid * (weight_max - weight_min)

        return cls(
            content=content,
            hypothesis_content=content,
            domain=domain,
            weight=weight,
            initial_confidence=weight
        )


class CorrectionResult(BaseModel):
    """Result from the Correction Agent."""
    conflicting_node_ids: List[str] = Field(default_factory=list)
    decay_applied: Dict[str, float] = Field(default_factory=dict, description="node_id -> new_weight")
    new_nodes_created: List[MemoryNode] = Field(default_factory=list)
    diagnosis: str = Field(default="", description="Explanation of the conflict")


class ProfilingResult(BaseModel):
    """Result from the Profiling Agent."""
    hypothesis: Optional[HypothesisNode] = None
    reasoning: str = Field(default="", description="Internal monologue explaining the hypothesis")


class MaintenanceResult(BaseModel):
    """Result from the Maintenance Agent."""
    reinforced_node_ids: List[str] = Field(default_factory=list)
    weight_updates: Dict[str, float] = Field(default_factory=dict, description="node_id -> new_weight")
    hypotheses_promoted: List[str] = Field(default_factory=list, description="Hypothesis IDs promoted to facts")


class LLMResponse(BaseModel):
    """Structured response from LLM calls."""
    content: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
