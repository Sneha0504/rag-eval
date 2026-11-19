"""Central configuration for the RAG evaluation pipeline.

Every tunable (random seed, paths, model name, k values) lives here so runs
are deterministic and reproducible.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed every source of randomness we touch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # torch is optional at config-import time; embedding step seeds it too.
        pass


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PEPS_REPO_DIR = RAW_DIR / "peps-repo"
# Modern layout keeps the .rst files under peps/; older layout had them at root.
PEPS_SOURCE_DIRS = [PEPS_REPO_DIR / "peps", PEPS_REPO_DIR]

RESULTS_DIR = ROOT / "results"
CHROMA_DIR = DATA_DIR / "chroma"
PROCESSED_DIR = DATA_DIR / "processed"

GOLDEN_SET_PATH = DATA_DIR / "golden_set.json"
INGESTION_REPORT_PATH = RESULTS_DIR / "ingestion_report.json"
INGESTION_LOG_PATH = RESULTS_DIR / "ingestion.log"
QUERY_LOG_PATH = RESULTS_DIR / "query_log.jsonl"
DOCUMENTS_PATH = PROCESSED_DIR / "documents.json"

# --------------------------------------------------------------------------
# Corpus provenance (recorded in README)
# --------------------------------------------------------------------------
CORPUS_SOURCE_URL = "https://github.com/python/peps"
CORPUS_LICENSE = "Public domain (per PEP 1)"

# --------------------------------------------------------------------------
# Embeddings / vector store
# --------------------------------------------------------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64
# Pin embeddings to CPU: GPU/MPS matmuls are not bit-reproducible, which would
# make retrieval metrics drift run-to-run even with the seed fixed.
EMBED_DEVICE = "cpu"

STRATEGY_FIXED = "fixed"
STRATEGY_SECTION = "section"
STRATEGIES = [STRATEGY_FIXED, STRATEGY_SECTION]
COLLECTION_PREFIX = "peps_"

# Chroma HNSW settings. num_threads=1 makes index construction deterministic
# (hnswlib's internal random seed is already fixed; parallel insertion order was
# the only nondeterminism). High construction/search ef + M push recall to
# effectively-exact for this corpus size, so retrieval metrics are both stable
# and reproducible across fresh index builds.
HNSW_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:num_threads": 1,
    "hnsw:construction_ef": 200,
    "hnsw:search_ef": 200,
    "hnsw:M": 32,
}


def collection_name(strategy: str) -> str:
    return f"{COLLECTION_PREFIX}{strategy}"


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
# Fixed-size chunking is measured in characters (model-agnostic, deterministic).
FIXED_CHUNK_CHARS = 2000      # ~500 tokens at ~4 chars/token
FIXED_CHUNK_OVERLAP = 200     # ~50 tokens of overlap
# Section-aware chunks that exceed this many characters are sub-split.
SECTION_MAX_CHARS = 2600
MIN_CHUNK_CHARS = 40          # anything shorter is flagged as an empty/trivial chunk

# --------------------------------------------------------------------------
# Retrieval / evaluation
# --------------------------------------------------------------------------
K_VALUES = [1, 3, 5, 10]
TOP_K = max(K_VALUES)

# --------------------------------------------------------------------------
# LLM (answer generation + judge) -- only used when a key is present
# --------------------------------------------------------------------------
# Provider-agnostic LLM settings driven entirely by environment variables so the
# answer generator (src/generate.py) and the groundedness judge (src/judge.py)
# can target any of several back ends without touching the retrieval layer.
#
#   LLM_PROVIDER  one of: "gemini" (default), "groq", "anthropic", "openai", "custom"
#
# The default is Google Gemini's FREE tier, reached through its OpenAI-compatible
# endpoint with the `openai` SDK, so a working run costs nothing. The Anthropic
# path is still selectable for parity with the original design.
#
# Per-field overrides (take precedence over the provider preset):
#   LLM_MODEL, LLM_BASE_URL, LLM_API_KEY,
#   LLM_PRICE_INPUT_PER_MTOK, LLM_PRICE_OUTPUT_PER_MTOK
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

PROVIDER_GEMINI = "gemini"
PROVIDER_GROQ = "groq"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_CUSTOM = "custom"

# Each preset: which SDK ("openai" vs "anthropic"), base_url, default model,
# the ordered env vars checked for a key, whether it is a $0 free tier, and the
# default published price (USD per 1M tokens) used only to compute real cost from
# real token counts. The Gemini Flash free tier is $0, so its price is 0.0 and a
# keyed run honestly reports $0/query -- no token counts are ever fabricated.
_PROVIDER_PRESETS = {
    PROVIDER_GEMINI: {
        "sdk": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        # A self-updating alias: stays valid as specific versions retire (avoids
        # the 404s that pinned, now-removed versions cause). Override with
        # LLM_MODEL to pin a specific one, e.g. gemini-2.5-flash.
        "model": "gemini-flash-latest",
        "key_envs": ("GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"),
        "free_tier": True,
        "price_input_per_mtok": 0.0,
        "price_output_per_mtok": 0.0,
    },
    PROVIDER_GROQ: {
        "sdk": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        # A general-purpose chat model on Groq's free tier. Model availability
        # changes over time; override with LLM_MODEL, and list what your key can
        # use via GET https://api.groq.com/openai/v1/models.
        "model": "openai/gpt-oss-120b",
        "key_envs": ("GROQ_API_KEY", "LLM_API_KEY"),
        "free_tier": True,
        "price_input_per_mtok": 0.0,
        "price_output_per_mtok": 0.0,
    },
    PROVIDER_ANTHROPIC: {
        "sdk": "anthropic",
        "base_url": None,  # the Anthropic SDK uses its own default endpoint
        "model": "claude-haiku-5-5",
        "key_envs": (ANTHROPIC_API_KEY_ENV, "LLM_API_KEY"),
        "free_tier": False,
        # Published price for claude-haiku-5-5: $0.10 in / $0.50 out per 1M tokens.
        "price_input_per_mtok": 0.10,
        "price_output_per_mtok": 0.50,
    },
    PROVIDER_OPENAI: {
        "sdk": "openai",
        "base_url": "https://api.openai.com/v1/",
        "model": "gpt-4o-mini",
        "key_envs": ("OPENAI_API_KEY", "LLM_API_KEY"),
        "free_tier": False,
        # Price unknown by default; set LLM_PRICE_* to report real cost.
        "price_input_per_mtok": 0.0,
        "price_output_per_mtok": 0.0,
    },
    PROVIDER_CUSTOM: {
        "sdk": "openai",
        "base_url": None,  # must be supplied via LLM_BASE_URL
        "model": "",       # must be supplied via LLM_MODEL
        "key_envs": ("LLM_API_KEY", "OPENAI_API_KEY"),
        "free_tier": False,
        "price_input_per_mtok": 0.0,
        "price_output_per_mtok": 0.0,
    },
}


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


LLM_PROVIDER = (_env("LLM_PROVIDER") or PROVIDER_GEMINI).lower()
_PRESET = _PROVIDER_PRESETS.get(LLM_PROVIDER, _PROVIDER_PRESETS[PROVIDER_CUSTOM])

LLM_SDK = _PRESET["sdk"]
LLM_BASE_URL = _env("LLM_BASE_URL") or _PRESET["base_url"]
LLM_MODEL = _env("LLM_MODEL") or _PRESET["model"]
LLM_IS_FREE_TIER = _PRESET["free_tier"]
LLM_API_KEY_ENVS = _PRESET["key_envs"]


def _float_env(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


LLM_PRICE_INPUT_PER_MTOK = _float_env(
    "LLM_PRICE_INPUT_PER_MTOK", _PRESET["price_input_per_mtok"]
)
LLM_PRICE_OUTPUT_PER_MTOK = _float_env(
    "LLM_PRICE_OUTPUT_PER_MTOK", _PRESET["price_output_per_mtok"]
)

# Resilience for flaky free tiers (Gemini free tier often returns 429/503).
# The SDK retries with exponential backoff; a small pause between calls also
# reduces overload. All overridable via env.
LLM_MAX_RETRIES = int(_float_env("LLM_MAX_RETRIES", 6))
LLM_TIMEOUT_S = _float_env("LLM_TIMEOUT_S", 60.0)
LLM_REQUEST_PAUSE_S = _float_env("LLM_REQUEST_PAUSE_S", 0.5)
# Client-side rate limit: minimum seconds between consecutive LLM calls, to stay
# under a free tier's requests-per-minute cap and avoid 429s in the first place.
# 0 = disabled. e.g. 4.0 => ~15 calls/min; 2.0 => ~30 calls/min.
LLM_MIN_REQUEST_INTERVAL_S = _float_env("LLM_MIN_REQUEST_INTERVAL_S", 0.0)


def resolve_api_key() -> str:
    """Return the API key for the selected provider, or '' if none is set.

    LLM_API_KEY is always honoured as a generic override; otherwise the
    provider's own env vars are checked in order (e.g. GEMINI_API_KEY first for
    the default Gemini free tier).
    """
    explicit = _env("LLM_API_KEY")
    if explicit:
        return explicit
    for env in LLM_API_KEY_ENVS:
        val = _env(env)
        if val:
            return val
    return ""


def has_api_key() -> bool:
    return bool(resolve_api_key())


def api_key_env_hint() -> str:
    """Human-readable name of the env var the user should set for this provider."""
    return LLM_API_KEY_ENVS[0] if LLM_API_KEY_ENVS else "LLM_API_KEY"


def ensure_dirs() -> None:
    for d in (DATA_DIR, RESULTS_DIR, CHROMA_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)
