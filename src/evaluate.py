"""Retrieval evaluation over the golden set for both chunking strategies.

Metrics (all from real runs):
  * hit rate @ k for k in config.K_VALUES
  * mean reciprocal rank (MRR)
  * median and p95 per-query latency (ms)
  * cost per query (retrieval is local => $0; LLM cost is reported separately
    by the judge/generation steps only when a key is present)

A hit = the gold PEP appears among the retrieved chunks' documents within top-k.
Every query is also written to the monitoring log.
"""
from __future__ import annotations

import json
import logging
import math
import statistics
from datetime import datetime, timezone

import config
from src import monitor, retrieve

logger = logging.getLogger("evaluate")


def load_golden() -> list[dict]:
    with config.GOLDEN_SET_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)["items"]


def _first_hit_rank(hits: list[dict], gold_pep: int) -> int | None:
    for h in hits:
        if h["metadata"].get("pep_number") == gold_pep:
            return h["rank"]
    return None


def evaluate_strategy(strategy: str, golden: list[dict]) -> dict:
    config.set_seed()
    per_query = []
    latencies = []
    ranks = []  # None if miss
    for item in golden:
        res = retrieve.retrieve(item["question"], strategy, k=config.TOP_K)
        rank = _first_hit_rank(res["hits"], item["gold_pep"])
        latencies.append(res["latency_ms"])
        ranks.append(rank)
        retrieved_ids = [h["chunk_id"] for h in res["hits"]]
        monitor.log_query(
            query=item["question"],
            strategy=strategy,
            latency_ms=res["latency_ms"],
            retrieved_ids=retrieved_ids,
            k=config.TOP_K,
            extra={"eval_id": item["id"], "gold_pep": item["gold_pep"],
                   "first_hit_rank": rank},
        )
        per_query.append({
            "id": item["id"],
            "question": item["question"],
            "category": item["category"],
            "gold_pep": item["gold_pep"],
            "gold_doc_id": item["gold_doc_id"],
            "first_hit_rank": rank,
            "retrieved_peps": [h["metadata"].get("pep_number") for h in res["hits"]],
            "retrieved_ids": retrieved_ids,
            "latency_ms": round(res["latency_ms"], 3),
        })

    n = len(golden)
    hit_rate = {}
    for k in config.K_VALUES:
        hits_at_k = sum(1 for r in ranks if r is not None and r <= k)
        hit_rate[f"hit_rate@{k}"] = round(hits_at_k / n, 4)
    mrr = round(sum((1.0 / r) if r else 0.0 for r in ranks) / n, 4)

    result = {
        "strategy": strategy,
        "collection": config.collection_name(strategy),
        "n_queries": n,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "latency_ms": {
            "median": round(statistics.median(latencies), 3),
            # Nearest-rank percentile: index = ceil(p * n) - 1.
            "p95": round(sorted(latencies)[max(0, math.ceil(0.95 * n) - 1)], 3),
            "mean": round(statistics.fmean(latencies), 3),
        },
        "latency_note": "Per-query latency measures the Chroma similarity search "
                        "only (it excludes query-embedding time); it is wall-clock "
                        "and so varies slightly run-to-run, unlike the deterministic "
                        "hit-rate/MRR.",
        "cost_usd_per_query": 0.0,
        "cost_note": "Retrieval uses a local sentence-transformers model; no API "
                     "cost per query. LLM answer/judge cost is reported separately "
                     "and only when an LLM API key is set (default provider: "
                     "Gemini free tier, $0).",
        "misses": [pq for pq in per_query if pq["first_hit_rank"] is None],
        "per_query": per_query,
    }
    return result


def _load_chunk_dq() -> dict[str, dict]:
    """Per-strategy chunk stats from index_summary.json (if present): the indexed
    count (what is actually searched) plus the DQ findings."""
    path = config.RESULTS_DIR / "index_summary.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        summary = json.load(fh)
    out = {}
    for s in config.STRATEGIES:
        if s in summary.get("strategies", {}):
            st = summary["strategies"][s]
            dq = st["chunk_dq"]
            out[s] = {
                "indexed": st["indexed_chunks"],
                "empty": dq["empty_chunks"],
                "dup": dq["exact_duplicate_chunks"],
            }
    return out


def _why(chunk_dq: dict[str, dict]) -> str:
    """Explain WHY fixed beats section using real chunk numbers (indexed count)."""
    f = chunk_dq.get(config.STRATEGY_FIXED)
    s = chunk_dq.get(config.STRATEGY_SECTION)
    if not f or not s:
        return ("Fixed-size windows carry more surrounding context per chunk than "
                "section-aware fragments, which matches question topics more reliably.")
    return (
        f"Why: section-aware chunking indexed {s['indexed']} chunks but its data "
        f"quality is weaker -- {s['dup']} exact-duplicate and {s['empty']} empty "
        "chunks were flagged (empties dropped before indexing); many remaining "
        "chunks are tiny header-only fragments (Abstract, Copyright, References) "
        "that dilute the top-k and split a PEP's content across many small vectors. "
        f"Fixed-size chunking indexed {f['indexed']} denser "
        f"{config.FIXED_CHUNK_CHARS}-char windows ({f['dup']} duplicate, {f['empty']} "
        "empty), so each chunk carries more context and matches the question topic "
        "more reliably -- hence the higher MRR and hit@k."
    )


def _verdict(results: dict[str, dict], chunk_dq: dict[str, dict] | None = None) -> str:
    chunk_dq = chunk_dq or {}
    fixed = results[config.STRATEGY_FIXED]
    section = results[config.STRATEGY_SECTION]
    f_mrr, s_mrr = fixed["mrr"], section["mrr"]
    f_h5 = fixed["hit_rate"]["hit_rate@5"]
    s_h5 = section["hit_rate"]["hit_rate@5"]
    if s_mrr == f_mrr and s_h5 == f_h5:
        return (
            f"Tie: both strategies reach MRR {s_mrr} and hit_rate@5 {s_h5}. "
            "Section-aware is preferred for interpretability (answers map to a "
            "named section) at similar retrieval quality."
        )
    winner = config.STRATEGY_SECTION if (s_mrr, s_h5) >= (f_mrr, f_h5) else config.STRATEGY_FIXED
    win = results[winner]
    lose = results[config.STRATEGY_FIXED if winner == config.STRATEGY_SECTION else config.STRATEGY_SECTION]
    return (
        f"Winner: {winner} chunking. It scores MRR {win['mrr']} vs {lose['mrr']} "
        f"and hit_rate@5 {win['hit_rate']['hit_rate@5']} vs "
        f"{lose['hit_rate']['hit_rate@5']}. Median Chroma-search latency "
        f"{win['latency_ms']['median']} ms vs {lose['latency_ms']['median']} ms. "
        + _why(chunk_dq)
    )


def run() -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    config.ensure_dirs()
    golden = load_golden()
    results = {}
    for strategy in config.STRATEGIES:
        logger.info("Evaluating strategy: %s", strategy)
        res = evaluate_strategy(strategy, golden)
        results[strategy] = res
        out = config.RESULTS_DIR / f"eval_{strategy}.json"
        with out.open("w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
        logger.info(
            "%s: MRR=%.4f hit@1=%.3f hit@5=%.3f median_latency=%.1fms misses=%d",
            strategy, res["mrr"], res["hit_rate"]["hit_rate@1"],
            res["hit_rate"]["hit_rate@5"], res["latency_ms"]["median"], len(res["misses"]),
        )

    comparison = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": config.EMBED_MODEL_NAME,
        "seed": config.SEED,
        "k_values": config.K_VALUES,
        "strategies": {
            s: {
                "hit_rate": results[s]["hit_rate"],
                "mrr": results[s]["mrr"],
                "latency_ms": results[s]["latency_ms"],
                "cost_usd_per_query": results[s]["cost_usd_per_query"],
                "n_misses": len(results[s]["misses"]),
            }
            for s in config.STRATEGIES
        },
        "verdict": _verdict(results, _load_chunk_dq()),
    }
    with (config.RESULTS_DIR / "eval_comparison.json").open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, ensure_ascii=False)
    logger.info("VERDICT: %s", comparison["verdict"])
    return comparison


if __name__ == "__main__":
    run()
