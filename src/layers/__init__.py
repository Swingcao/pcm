"""
Layers module for PCM system.
Contains the three-layer architecture implementation.
"""

from .layer1_perception import (
    SlidingContextQueue,
    IntentRouter,
    SurpriseMonitor,
    PerceptionLayer
)
from .layer2_world_model import WeightedKnowledgeGraph
from .layer3_evolution import (
    CorrectionAgent,
    ProfilingAgent,
    MaintenanceAgent,
    SurpriseDispatcher,
    CognitiveEngine
)

__all__ = [
    # Layer 1
    "SlidingContextQueue",
    "IntentRouter",
    "SurpriseMonitor",
    "PerceptionLayer",
    # Layer 2
    "WeightedKnowledgeGraph",
    # Layer 3
    "CorrectionAgent",
    "ProfilingAgent",
    "MaintenanceAgent",
    "SurpriseDispatcher",
    "CognitiveEngine"
]
