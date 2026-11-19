"""Top-k similarity retrieval over a strategy's Chroma collection, with
optional metadata filtering (status / type / PEP number)."""
from __future__ import annotations

import logging
import time
from typing import Any

import config
from src import embed_index

logger = logging.getLogger("retrieve")

_CLIENT = None
_COLLECTIONS: dict[str, Any] = {}


def _client():
    global _CLIENT
    if _CLIENT is None:
        import chromadb
        from chromadb.config import Settings

        logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
        _CLIENT = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _CLIENT


def get_collection(strategy: str):
    if strategy not in _COLLECTIONS:
        _COLLECTIONS[strategy] = _client().get_collection(config.collection_name(strategy))
    return _COLLECTIONS[strategy]


def _build_where(metadata_filter: dict | None) -> dict | None:
    """Translate a simple {field: value} filter into a Chroma `where` clause."""
    if not metadata_filter:
        return None
    clauses = [{k: {"$eq": v}} for k, v in metadata_filter.items()]
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def retrieve(
    query: str,
    strategy: str,
    k: int = config.TOP_K,
    metadata_filter: dict | None = None,
) -> dict:
    """Return top-k chunks for a query. Includes wall-clock latency in ms."""
    collection = get_collection(strategy)
    q_emb = embed_index.embed_texts([query])[0].tolist()
    where = _build_where(metadata_filter)

    t0 = time.perf_counter()
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    hits = []
    ids = res["ids"][0]
    for i, cid in enumerate(ids):
        hits.append({
            "chunk_id": cid,
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
            "score": 1.0 - res["distances"][0][i],  # cosine similarity
        })
    # Deterministic ordering: break distance ties by chunk_id so ranks (and thus
    # hit@k / MRR) are stable run-to-run even when several chunks tie on distance.
    hits.sort(key=lambda h: (round(h["distance"], 9), h["chunk_id"]))
    for i, h in enumerate(hits):
        h["rank"] = i + 1
    return {
        "query": query,
        "strategy": strategy,
        "k": k,
        "metadata_filter": metadata_filter,
        "latency_ms": latency_ms,
        "hits": hits,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = retrieve("What is the style guide for Python code?", config.STRATEGY_SECTION, k=5)
    for h in out["hits"]:
        print(h["rank"], h["chunk_id"], round(h["score"], 3), h["metadata"]["title"][:50])
