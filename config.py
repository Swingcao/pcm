"""
PCM System Configuration
========================
Global configuration for the Proactive Cognitive Memory system.

Configuration Priority:
1. config.yaml (if exists)
2. Environment variables
3. Default values
"""

import os
from typing import Any, Dict, Optional

# Try to load YAML config first
_config: Dict[str, Any] = {}
_config_loaded = False

def _load_yaml_config() -> Dict[str, Any]:
    """Load configuration from config.yaml file."""
    global _config, _config_loaded

    if _config_loaded:
        return _config

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                _config = yaml.safe_load(f) or {}
            print(f"Loaded configuration from {config_path}")
        except ImportError:
            print("Warning: PyYAML not installed. Using default configuration.")
            _config = {}
        except Exception as e:
            print(f"Warning: Failed to load config.yaml: {e}")
            _config = {}
    else:
        # Try .env file as fallback
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        _config = {}

    _config_loaded = True
    return _config


def _get(section: str, key: str, default: Any = None, env_var: Optional[str] = None) -> Any:
    """
    Get a configuration value with fallback chain.

    Priority: YAML config -> Environment variable -> Default value
    """
    config = _load_yaml_config()

    # Try YAML config first
    if section in config and key in config[section]:
        return config[section][key]

    # Try environment variable
    if env_var:
        env_value = os.getenv(env_var)
        if env_value is not None:
            return env_value

    return default


# =============================================================================
# LLM Configuration
# =============================================================================

OPENAI_API_KEY = _get("model", "openai_api_key", "", "OPENAI_API_KEY")
OPENAI_BASE_URL = _get("model", "openai_base_url", "https://api.openai.com/v1", "OPENAI_BASE_URL")
MAIN_LLM_MODEL = _get("model", "llm_model", "gpt-4o")
ROUTER_LLM_MODEL = _get("model", "router_model", "gpt-4o-mini")


# =============================================================================
# Embedding Model Configuration
# =============================================================================

# Local model path (relative to project root)
_base_dir_for_models = os.path.dirname(__file__)
_default_local_model_path = os.path.join(_base_dir_for_models, "models", "all-MiniLM-L6-v2")

EMBEDDING_MODEL_ID = _get("embedding", "model_id", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_LOCAL_PATH = _get("embedding", "local_path", _default_local_model_path)
EMBEDDING_USE_LOCAL = _get("embedding", "use_local", True)  # Default to use local model
EMBEDDING_DIMENSION = _get("embedding", "dimension", 384)


# =============================================================================
# Surprisal Configuration
# =============================================================================

# Alpha: weight for embedding distance vs LLM conflict score
# 0 = pure LLM, 1 = pure embedding
SURPRISAL_ALPHA = _get("surprisal", "alpha", 0.4)


# =============================================================================
# Surprisal Thresholds
# =============================================================================

THETA_HIGH = _get("thresholds", "theta_high", 3.0)
THETA_LOW = _get("thresholds", "theta_low", 1.0)
SURPRISAL_LAMBDA = _get("thresholds", "lambda_factor", 0.5)


# =============================================================================
# Weight Update Parameters
# =============================================================================

ETA = _get("weights", "eta", 0.05)
BETA = _get("weights", "beta", 0.3)
GAMMA = _get("weights", "gamma", 0.01)
HYPOTHESIS_WEIGHT_MIN = _get("weights", "hypothesis_min", 0.3)
HYPOTHESIS_WEIGHT_MAX = _get("weights", "hypothesis_max", 0.5)


# =============================================================================
# Working Memory Configuration
# =============================================================================

MAX_CONTEXT_TOKENS = _get("working_memory", "max_context_tokens", 2000)
EVICTION_SIZE = _get("working_memory", "eviction_size", 3)


# =============================================================================
# Knowledge Graph Configuration
# =============================================================================

RETRIEVAL_TOP_K = _get("knowledge_graph", "retrieval_top_k", 10)
RETRIEVAL_MIN_SCORE = _get("knowledge_graph", "retrieval_min_score", 0.3)


# =============================================================================
# Hybrid Retrieval Configuration (v1.1 Optimization)
# =============================================================================

# Master switch: Enable hybrid retrieval (semantic + keyword + graph)
# When False, uses original embedding-only retrieval
USE_HYBRID_RETRIEVAL = _get("hybrid_retrieval", "enabled", False, "USE_HYBRID_RETRIEVAL")
if isinstance(USE_HYBRID_RETRIEVAL, str):
    USE_HYBRID_RETRIEVAL = USE_HYBRID_RETRIEVAL.lower() == "true"

# Enable structured fact extraction from dialogue
USE_FACT_EXTRACTION = _get("hybrid_retrieval", "fact_extraction", False)
if isinstance(USE_FACT_EXTRACTION, str):
    USE_FACT_EXTRACTION = USE_FACT_EXTRACTION.lower() == "true"

# Enable automatic edge creation in knowledge graph
USE_EDGE_CREATION = _get("hybrid_retrieval", "edge_creation", False)
if isinstance(USE_EDGE_CREATION, str):
    USE_EDGE_CREATION = USE_EDGE_CREATION.lower() == "true"

# Use LLM for fact extraction and contradiction detection (slower but more accurate)
HYBRID_USE_LLM = _get("hybrid_retrieval", "use_llm", False)
if isinstance(HYBRID_USE_LLM, str):
    HYBRID_USE_LLM = HYBRID_USE_LLM.lower() == "true"

# Retrieval weight configuration
# Score = α×semantic + β×keyword + γ×graph + δ×recency
HYBRID_SEMANTIC_WEIGHT = _get("hybrid_retrieval", "semantic_weight", 0.4)
HYBRID_KEYWORD_WEIGHT = _get("hybrid_retrieval", "keyword_weight", 0.3)
HYBRID_GRAPH_WEIGHT = _get("hybrid_retrieval", "graph_weight", 0.2)
HYBRID_RECENCY_WEIGHT = _get("hybrid_retrieval", "recency_weight", 0.1)

# Graph expansion depth for multi-hop retrieval
HYBRID_EXPANSION_DEPTH = _get("hybrid_retrieval", "expansion_depth", 1)

# BM25 parameters
BM25_K1 = _get("hybrid_retrieval", "bm25_k1", 1.2)
BM25_B = _get("hybrid_retrieval", "bm25_b", 0.75)


# =============================================================================
# Query-Adaptive Retrieval Configuration (v1.3 Optimization)
# =============================================================================

# Enable query-type adaptive weights
# When True: Dynamically adjusts retrieval weights based on detected query type
#   - Factual queries: keyword_weight=0.50 for better entity matching
#   - Temporal queries: recency_weight=0.20 for time-aware retrieval
#   - Multi-hop queries: graph_weight=0.40 for reasoning traversal
USE_ADAPTIVE_WEIGHTS = _get("adaptive_retrieval", "enabled", True, "USE_ADAPTIVE_WEIGHTS")
if isinstance(USE_ADAPTIVE_WEIGHTS, str):
    USE_ADAPTIVE_WEIGHTS = USE_ADAPTIVE_WEIGHTS.lower() == "true"

# Enable entity-centric boosting
# When True: Boosts documents that match entities mentioned in the query
USE_ENTITY_BOOST = _get("adaptive_retrieval", "entity_boost", True, "USE_ENTITY_BOOST")
if isinstance(USE_ENTITY_BOOST, str):
    USE_ENTITY_BOOST = USE_ENTITY_BOOST.lower() == "true"

# Entity boost factor (multiplier for documents matching query entities)
# 1.0 = no boost, 1.5 = 50% boost for entity matches
ENTITY_BOOST_FACTOR = _get("adaptive_retrieval", "entity_boost_factor", 1.5)


# =============================================================================
# Original Text Preservation Configuration (v1.4)
# =============================================================================

# Master switch: Enable original text preservation
# When True: Preserves original message text in source_text fields
PRESERVE_ORIGINAL_TEXT = _get("original_text_preservation", "enabled", True, "PRESERVE_ORIGINAL_TEXT")
if isinstance(PRESERVE_ORIGINAL_TEXT, str):
    PRESERVE_ORIGINAL_TEXT = PRESERVE_ORIGINAL_TEXT.lower() == "true"

# Store reference nodes for reinforcing messages (MaintenanceAgent)
# When True: Creates reference nodes with original text even for low-surprise messages
STORE_REINFORCEMENTS = _get("original_text_preservation", "store_reinforcements", True)
if isinstance(STORE_REINFORCEMENTS, str):
    STORE_REINFORCEMENTS = STORE_REINFORCEMENTS.lower() == "true"

# Cache assistant messages alongside user messages
# When True: Both user and assistant messages are stored in WM cache
CACHE_ASSISTANT_MESSAGES = _get("original_text_preservation", "cache_assistant_messages", True)
if isinstance(CACHE_ASSISTANT_MESSAGES, str):
    CACHE_ASSISTANT_MESSAGES = CACHE_ASSISTANT_MESSAGES.lower() == "true"

# Source text weight in hybrid retrieval (ε)
# Weight for original text keyword matching in retrieval score
# Score = α×semantic + β×keyword + γ×graph + δ×recency + ε×source_text
SOURCE_TEXT_WEIGHT = _get("original_text_preservation", "source_text_weight", 0.15)

# Working Memory Cache path
WM_CACHE_PATH = _get("original_text_preservation", "wm_cache_path", None)
# If not specified, will default to {results_dir}/wm_cache.json


# =============================================================================
# Intent Domains (Legacy - used when USE_DYNAMIC_TOPICS=False)
# =============================================================================

_default_domains = ["Coding", "Academic", "Personal", "Casual", "Professional", "Creative"]
INTENT_DOMAINS = _get("intent_domains", None, _default_domains) or _default_domains
# Handle case where intent_domains is at root level in YAML
if isinstance(INTENT_DOMAINS, dict):
    INTENT_DOMAINS = _default_domains
config = _load_yaml_config()
if "intent_domains" in config and isinstance(config["intent_domains"], list):
    INTENT_DOMAINS = config["intent_domains"]


# =============================================================================
# Dynamic Topic Discovery (v1.2 - Replaces fixed intent domains)
# =============================================================================

# Master switch: Use dynamic topic extraction instead of fixed intent domains
# When True: LLM extracts topics dynamically from content
# When False: Uses predefined intent_domains for classification
USE_DYNAMIC_TOPICS = _get("dynamic_topics", "enabled", True, "USE_DYNAMIC_TOPICS")
if isinstance(USE_DYNAMIC_TOPICS, str):
    USE_DYNAMIC_TOPICS = USE_DYNAMIC_TOPICS.lower() == "true"

# Use LLM for topic extraction (slower but more accurate)
# When False: Uses rule-based topic extraction
TOPIC_EXTRACTION_USE_LLM = _get("dynamic_topics", "use_llm", True)
if isinstance(TOPIC_EXTRACTION_USE_LLM, str):
    TOPIC_EXTRACTION_USE_LLM = TOPIC_EXTRACTION_USE_LLM.lower() == "true"


# =============================================================================
# Data Paths
# =============================================================================

_base_dir = os.path.dirname(__file__)
DATA_DIR = _get("paths", "data_dir", os.path.join(_base_dir, "data"))
RESULTS_DIR = _get("paths", "results_dir", os.path.join(_base_dir, "results"))
GRAPH_SAVE_PATH = _get("paths", "graph_save", os.path.join(RESULTS_DIR, "knowledge_graphs", "knowledge_graph.json"))
VECTOR_STORE_DIR = _get("paths", "vector_store", os.path.join(RESULTS_DIR, "vector_stores"))
DATASET_PATH = _get("paths", "dataset", os.path.join(DATA_DIR, "locomo10.json"))

# Legacy path for backwards compatibility
CHROMA_PERSIST_DIR = _get("paths", "chroma_persist", os.path.join(DATA_DIR, "chroma_db"))


# =============================================================================
# Debug / Testing Mode
# =============================================================================

USE_MOCK_LLM = _get("debug", "use_mock_llm", False, "USE_MOCK_LLM")
if isinstance(USE_MOCK_LLM, str):
    USE_MOCK_LLM = USE_MOCK_LLM.lower() == "true"

VERBOSE = _get("debug", "verbose", False)


# =============================================================================
# Utility Functions
# =============================================================================

def get_config() -> Dict[str, Any]:
    """Get the full configuration dictionary."""
    return {
        "model": {
            "llm_model": MAIN_LLM_MODEL,
            "router_model": ROUTER_LLM_MODEL,
            "openai_base_url": OPENAI_BASE_URL,
        },
        "embedding": {
            "model_id": EMBEDDING_MODEL_ID,
            "local_path": EMBEDDING_LOCAL_PATH,
            "use_local": EMBEDDING_USE_LOCAL,
            "dimension": EMBEDDING_DIMENSION,
        },
        "thresholds": {
            "theta_high": THETA_HIGH,
            "theta_low": THETA_LOW,
            "lambda_factor": SURPRISAL_LAMBDA,
        },
        "weights": {
            "eta": ETA,
            "beta": BETA,
            "gamma": GAMMA,
        },
        "hybrid_retrieval": {
            "enabled": USE_HYBRID_RETRIEVAL,
            "fact_extraction": USE_FACT_EXTRACTION,
            "edge_creation": USE_EDGE_CREATION,
            "use_llm": HYBRID_USE_LLM,
            "semantic_weight": HYBRID_SEMANTIC_WEIGHT,
            "keyword_weight": HYBRID_KEYWORD_WEIGHT,
            "graph_weight": HYBRID_GRAPH_WEIGHT,
            "recency_weight": HYBRID_RECENCY_WEIGHT,
            "expansion_depth": HYBRID_EXPANSION_DEPTH,
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
        },
        "adaptive_retrieval": {
            "enabled": USE_ADAPTIVE_WEIGHTS,
            "entity_boost": USE_ENTITY_BOOST,
            "entity_boost_factor": ENTITY_BOOST_FACTOR,
        },
        "original_text_preservation": {  # v1.4
            "enabled": PRESERVE_ORIGINAL_TEXT,
            "store_reinforcements": STORE_REINFORCEMENTS,
            "cache_assistant_messages": CACHE_ASSISTANT_MESSAGES,
            "source_text_weight": SOURCE_TEXT_WEIGHT,
            "wm_cache_path": WM_CACHE_PATH,
        },
        "dynamic_topics": {
            "enabled": USE_DYNAMIC_TOPICS,
            "use_llm": TOPIC_EXTRACTION_USE_LLM,
        },
        "paths": {
            "data_dir": DATA_DIR,
            "results_dir": RESULTS_DIR,
            "graph_save": GRAPH_SAVE_PATH,
            "vector_store": VECTOR_STORE_DIR,
            "dataset": DATASET_PATH,
        },
        "debug": {
            "use_mock_llm": USE_MOCK_LLM,
            "verbose": VERBOSE,
        }
    }


def print_config():
    """Print current configuration (without sensitive data)."""
    import json
    cfg = get_config()
    print("Current PCM Configuration:")
    print(json.dumps(cfg, indent=2))
