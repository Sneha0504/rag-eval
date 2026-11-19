"""Embed chunks with sentence-transformers and store them in Chroma.

One persistent collection per chunking strategy. Embedding and insertion order
are deterministic (chunks are pre-sorted by chunk_id and the seed is fixed).
"""
from __future__ import annotations

import json
import logging
import shutil

import config
from src import chunking

logger = logging.getLogger("embed_index")

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        config.set_seed()
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s on %s", config.EMBED_MODEL_NAME, config.EMBED_DEVICE)
        # Force CPU: MPS/GPU matmuls are not bit-reproducible, which would make
        # retrieval metrics vary run-to-run. CPU float32 keeps the seed honest.
        _MODEL = SentenceTransformer(config.EMBED_MODEL_NAME, device=config.EMBED_DEVICE)
    return _MODEL


def embed_texts(texts: list[str]):
    model = get_model()
    return model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def _get_client():
    import chromadb
    from chromadb.config import Settings

    # Silence the known harmless chromadb 0.5.3 posthog telemetry stderr noise.
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def index_strategy(strategy: str, docs: list[dict] | None = None) -> dict:
    """Build chunks for a strategy, embed them, and (re)create its collection."""
    config.set_seed()
    chunks = chunking.build_chunks(strategy, docs)
    dq = chunking.chunk_dq(chunks)
    logger.info(
        "Strategy %s: %d chunks (empty=%d, dup=%d)",
        strategy, dq["n_chunks"], dq["empty_chunks"], dq["exact_duplicate_chunks"],
    )

    # Drop empty chunks before indexing (they are logged in dq).
    chunks = [c for c in chunks if len(c["text"].strip()) >= config.MIN_CHUNK_CHARS]

    client = _get_client()
    name = config.collection_name(strategy)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name=name, metadata=dict(config.HNSW_METADATA))

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]

    embeddings = embed_texts(texts)

    batch = 512
    for start in range(0, len(ids), batch):
        collection.add(
            ids=ids[start : start + batch],
            documents=texts[start : start + batch],
            metadatas=metas[start : start + batch],
            embeddings=embeddings[start : start + batch].tolist(),
        )

    logger.info("Indexed %d chunks into collection '%s'", len(ids), name)
    return {
        "strategy": strategy,
        "collection": name,
        "indexed_chunks": len(ids),
        "chunk_dq": dq,
    }


def build_all() -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    config.ensure_dirs()
    docs = chunking.load_documents()
    summary = {"model": config.EMBED_MODEL_NAME, "strategies": {}}
    for strategy in config.STRATEGIES:
        summary["strategies"][strategy] = index_strategy(strategy, docs)
    out = config.RESULTS_DIR / "index_summary.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("Index summary written to %s", out)
    return summary


if __name__ == "__main__":
    build_all()
