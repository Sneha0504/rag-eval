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
LLM_MODEL = "claude-haiku-5-5"
# Published price for claude-haiku-5-5 (USD per 1M tokens): $0.10 in / $0.50 out.
LLM_PRICE_INPUT_PER_MTOK = 0.10
LLM_PRICE_OUTPUT_PER_MTOK = 0.50
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"


def has_api_key() -> bool:
    return bool(os.environ.get(ANTHROPIC_API_KEY_ENV, "").strip())


def ensure_dirs() -> None:
    for d in (DATA_DIR, RESULTS_DIR, CHROMA_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)
