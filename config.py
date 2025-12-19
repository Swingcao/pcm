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
# Intent Domains
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
