"""Two chunking strategies over the ingested PEP documents.

  * fixed    -- fixed-size character windows with overlap (model-agnostic,
                deterministic).
  * section  -- one chunk per RST section (heading kept as metadata); long
                sections are sub-split with overlap.

Both produce chunk records with full metadata and a stable, sorted chunk_id so
indexing order is deterministic.
"""
from __future__ import annotations

import json
import logging
from typing import Iterable

import config

logger = logging.getLogger("chunking")


def load_documents() -> list[dict]:
    if not config.DOCUMENTS_PATH.exists():
        raise FileNotFoundError(
            f"{config.DOCUMENTS_PATH} not found -- run ingestion first."
        )
    with config.DOCUMENTS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _base_meta(doc: dict) -> dict:
    return {
        "doc_id": doc["doc_id"],
        "pep_number": doc.get("pep_number") if doc.get("pep_number") is not None else -1,
        "title": doc.get("title", ""),
        "author": doc.get("author", ""),
        "status": doc.get("status", ""),
        "type": doc.get("type", ""),
        "created": doc.get("created", ""),
    }


def _window(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            out.append(piece)
        if start + size >= len(text):
            break
    return out


def chunk_fixed(docs: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for doc in docs:
        pieces = _window(doc["body"], config.FIXED_CHUNK_CHARS, config.FIXED_CHUNK_OVERLAP)
        for i, piece in enumerate(pieces):
            meta = _base_meta(doc)
            meta.update({
                "strategy": config.STRATEGY_FIXED,
                "section": "",
                "chunk_index": i,
            })
            chunks.append({
                "chunk_id": f"{doc['doc_id']}::fixed::{i:04d}",
                "text": piece,
                "metadata": meta,
            })
    chunks.sort(key=lambda c: c["chunk_id"])
    return chunks


def chunk_section(docs: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for doc in docs:
        sections = doc.get("sections") or [{"heading": "", "text": doc["body"]}]
        for s_idx, sec in enumerate(sections):
            heading = sec.get("heading", "")
            sec_text = (sec.get("text") or "").strip()
            if not sec_text:
                continue
            # Prepend the heading so the chunk is self-describing for retrieval.
            full = f"{heading}\n\n{sec_text}".strip() if heading else sec_text
            pieces = _window(full, config.SECTION_MAX_CHARS, config.FIXED_CHUNK_OVERLAP)
            for sub_idx, piece in enumerate(pieces):
                meta = _base_meta(doc)
                meta.update({
                    "strategy": config.STRATEGY_SECTION,
                    "section": heading,
                    "chunk_index": s_idx,
                    "sub_index": sub_idx,
                })
                chunks.append({
                    "chunk_id": f"{doc['doc_id']}::section::{s_idx:03d}::{sub_idx:03d}",
                    "text": piece,
                    "metadata": meta,
                })
    chunks.sort(key=lambda c: c["chunk_id"])
    return chunks


def build_chunks(strategy: str, docs: list[dict] | None = None) -> list[dict]:
    docs = docs if docs is not None else load_documents()
    if strategy == config.STRATEGY_FIXED:
        return chunk_fixed(docs)
    if strategy == config.STRATEGY_SECTION:
        return chunk_section(docs)
    raise ValueError(f"unknown strategy: {strategy}")


def chunk_dq(chunks: list[dict]) -> dict:
    """Chunk-level data-quality summary (empty + exact-duplicate chunks)."""
    empty = [c["chunk_id"] for c in chunks if len(c["text"].strip()) < config.MIN_CHUNK_CHARS]
    seen: dict[str, str] = {}
    dup = []
    for c in chunks:
        key = " ".join(c["text"].lower().split())
        if key in seen:
            dup.append(c["chunk_id"])
        else:
            seen[key] = c["chunk_id"]
    return {
        "n_chunks": len(chunks),
        "empty_chunks": len(empty),
        "exact_duplicate_chunks": len(dup),
        "empty_chunk_ids": empty[:50],
        "duplicate_chunk_ids": dup[:50],
    }
