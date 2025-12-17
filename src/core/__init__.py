"""
Core module for PCM system.
Contains orchestrator and type definitions.
"""

from .types import (
    MemoryNode,
    MemoryEdge,
    Intent,
    SurprisalPacket,
    NodeType,
    EvictedData,
    HypothesisNode,
    ConversationTurn,
    CorrectionResult,
    ProfilingResult,
    MaintenanceResult
)

from .orchestrator import PCMSystem, create_pcm_system

__all__ = [
    # Types
    "MemoryNode",
    "MemoryEdge",
    "Intent",
    "SurprisalPacket",
    "NodeType",
    "EvictedData",
    "HypothesisNode",
    "ConversationTurn",
    "CorrectionResult",
    "ProfilingResult",
    "MaintenanceResult",
    # Orchestrator
    "PCMSystem",
    "create_pcm_system"
]
