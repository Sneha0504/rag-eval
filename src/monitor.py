"""Lightweight query monitoring: append one JSON line per query to
results/query_log.jsonl with timestamp, query, strategy, latency and the
retrieved chunk IDs."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import config


def log_query(
    query: str,
    strategy: str,
    latency_ms: float,
    retrieved_ids: list[str],
    k: int,
    extra: dict | None = None,
) -> None:
    config.ensure_dirs()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "strategy": strategy,
        "k": k,
        "latency_ms": round(latency_ms, 3),
        "retrieved_ids": retrieved_ids,
    }
    if extra:
        record.update(extra)
    with config.QUERY_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
