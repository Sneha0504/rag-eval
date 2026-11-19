"""Answer generation through a provider-agnostic LLM layer.

The generator grounds its answer in the retrieved chunks and cites the chunk
IDs it used. The LLM provider is selected in config.py (default: Google Gemini's
free tier via its OpenAI-compatible endpoint; Anthropic and other OpenAI-
compatible back ends are also selectable). If no API key is set for the selected
provider, it raises NoApiKeyError with a clear message -- callers catch this and
continue without fabricating output.
"""
from __future__ import annotations

import logging

import config
from src import llm_client
from src.llm_client import NoApiKeyError  # re-exported for downstream callers

logger = logging.getLogger("generate")

__all__ = ["NoApiKeyError", "generate_answer"]


def _format_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        meta = h["metadata"]
        header = f"[{h['chunk_id']}] PEP {meta.get('pep_number')} - {meta.get('title')}"
        section = meta.get("section") or ""
        if section:
            header += f" (section: {section})"
        blocks.append(f"{header}\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, hits: list[dict], model: str | None = None) -> dict:
    """Generate a cited answer from retrieved chunks.

    Returns {answer, cited_chunk_ids, model, usage, cost_usd}. Raises
    NoApiKeyError when no key is available for the selected provider.
    """
    context = _format_context(hits)
    allowed_ids = [h["chunk_id"] for h in hits]

    system = (
        "You are a precise assistant answering questions about Python "
        "Enhancement Proposals (PEPs). Answer ONLY from the provided context. "
        "If the context does not contain the answer, say so. Every answer must "
        "end with a 'Sources:' line listing the chunk IDs (in square brackets) "
        "you used. Only cite IDs from the provided context."
    )
    user = (
        f"Context chunks (cite by their bracketed IDs):\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the context above and cite the chunk IDs you used."
    )

    # Raises NoApiKeyError when no key is set for the selected provider.
    result = llm_client.chat(user=user, system=system, max_tokens=900, model=model)
    answer = result["text"]

    cited = [cid for cid in allowed_ids if cid in answer]
    return {
        "answer": answer,
        "cited_chunk_ids": cited,
        "model": result["model"],
        "usage": {
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
        },
        "cost_usd": result["cost_usd"],
    }
