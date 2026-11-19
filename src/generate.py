"""Answer generation with the Anthropic API.

The generator grounds its answer in the retrieved chunks and cites the chunk
IDs it used. If ANTHROPIC_API_KEY is not set, it raises NoApiKeyError with a
clear message -- callers catch this and continue without fabricating output.
"""
from __future__ import annotations

import logging

import config

logger = logging.getLogger("generate")


class NoApiKeyError(RuntimeError):
    """Raised when answer generation is requested without an API key."""


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
    NoApiKeyError when no key is available.
    """
    if not config.has_api_key():
        raise NoApiKeyError(
            "ANTHROPIC_API_KEY is not set; cannot generate an answer. "
            "Set the key and re-run to produce real LLM output."
        )

    import anthropic

    model = model or config.LLM_MODEL
    client = anthropic.Anthropic()
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

    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    answer = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")

    cited = [cid for cid in allowed_ids if cid in answer]
    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens
    cost = (
        in_tok / 1_000_000 * config.LLM_PRICE_INPUT_PER_MTOK
        + out_tok / 1_000_000 * config.LLM_PRICE_OUTPUT_PER_MTOK
    )
    return {
        "answer": answer,
        "cited_chunk_ids": cited,
        "model": model,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
        "cost_usd": cost,
    }
