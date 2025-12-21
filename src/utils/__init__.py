"""
Utilities module for PCM system.
Contains helper functions and tools.

Optimization Modules (v1.1):
- fact_extractor: Structured fact extraction from dialogue
- keyword_index: BM25 inverted index for keyword search
- edge_creator: Knowledge graph edge creation
- hybrid_retriever: Combined semantic + keyword retrieval

Dynamic Topics (v1.2):
- topic_extractor: Dynamic topic discovery (replaces fixed intent domains)

Query Adaptive Retrieval (v1.3):
- query_type_detector: Detects query type for adaptive weight configuration
- AdaptiveRetrievalConfig: Provides query-type specific weights
"""

from .llm_client import LLMClient, get_llm_client, chat
from .math_utils import (
    sigmoid,
    reinforcement_update,
    decay_update,
    time_decay,
    compute_retrieval_entropy,
    compute_effective_surprisal,
    compute_retrieval_score,
    cosine_similarity,
    softmax
)
from .metrics import SurprisalCalculator, get_surprisal_calculator

# Optimization modules for improved retrieval
from .fact_extractor import (
    StructuredFact,
    FactExtractor,
    TemporalResolver,
    extract_facts_from_dialogue,
    extract_facts_sync
)
from .keyword_index import (
    InvertedIndex,
    SearchResult,
    Tokenizer,
    STOPWORDS,
    create_index_from_nodes,
    merge_with_semantic_scores
)
from .edge_creator import (
    EdgeCreator,
    EdgeCandidate,
    EdgeRelationType,
    EntityExtractor,
    create_edges_for_graph,
    create_edges_sync
)
from .hybrid_retriever import (
    HybridRetriever,
    HybridSearchResult,
    RetrievalConfig,
    ScoreFusion,
    create_hybrid_retriever,
    AdaptiveRetrievalConfig  # v1.3
)

# v1.3: Query type detection for adaptive retrieval
from .query_type_detector import (
    QueryType,
    QueryTypeResult,
    QueryTypeDetector,
    get_query_type_detector,
    detect_query_type
)

# v1.3: Entity-centric indexing
from .entity_index import (
    EntityCentricIndex,
    EntityFact,
    EntityInfo,
    EntityExtractorSimple,
    create_entity_index_from_nodes
)

# v1.2: Dynamic topic discovery
from .topic_extractor import (
    TopicSet,
    TopicExtractor,
    TopicMatcher,
    get_topic_extractor,
    extract_topics,
    extract_topics_sync
)

__all__ = [
    # LLM Client
    "LLMClient",
    "get_llm_client",
    "chat",
    # Math Utils
    "sigmoid",
    "reinforcement_update",
    "decay_update",
    "time_decay",
    "compute_retrieval_entropy",
    "compute_effective_surprisal",
    "compute_retrieval_score",
    "cosine_similarity",
    "softmax",
    # Metrics
    "SurprisalCalculator",
    "get_surprisal_calculator",
    # Fact Extraction (Optimization)
    "StructuredFact",
    "FactExtractor",
    "TemporalResolver",
    "extract_facts_from_dialogue",
    "extract_facts_sync",
    # Keyword Index (Optimization)
    "InvertedIndex",
    "SearchResult",
    "Tokenizer",
    "STOPWORDS",
    "create_index_from_nodes",
    "merge_with_semantic_scores",
    # Edge Creator (Optimization)
    "EdgeCreator",
    "EdgeCandidate",
    "EdgeRelationType",
    "EntityExtractor",
    "create_edges_for_graph",
    "create_edges_sync",
    # Hybrid Retriever (Optimization)
    "HybridRetriever",
    "HybridSearchResult",
    "RetrievalConfig",
    "ScoreFusion",
    "create_hybrid_retriever",
    "AdaptiveRetrievalConfig",  # v1.3
    # Query Type Detection (v1.3)
    "QueryType",
    "QueryTypeResult",
    "QueryTypeDetector",
    "get_query_type_detector",
    "detect_query_type",
    # Entity-Centric Indexing (v1.3)
    "EntityCentricIndex",
    "EntityFact",
    "EntityInfo",
    "EntityExtractorSimple",
    "create_entity_index_from_nodes",
    # Dynamic Topics (v1.2)
    "TopicSet",
    "TopicExtractor",
    "TopicMatcher",
    "get_topic_extractor",
    "extract_topics",
    "extract_topics_sync"
]
