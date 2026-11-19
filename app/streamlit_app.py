"""Streamlit dashboard for the RAG evaluation pipeline.

Reads results/*.json and results/query_log.jsonl and shows:
  * the chunking-strategy comparison and verdict,
  * ingestion data-quality findings,
  * per-query retrieval detail,
  * query volume / latency over time from the monitoring log.

Run:  .venv/bin/streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402


def _load_json(path: Path):
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    return None


st.set_page_config(page_title="PEP RAG Evaluation", layout="wide")
st.title("PEP Retrieval & Evaluation Dashboard")

comparison = _load_json(config.RESULTS_DIR / "eval_comparison.json")
ingestion = _load_json(config.INGESTION_REPORT_PATH)
groundedness = _load_json(config.RESULTS_DIR / "eval_groundedness.json")

if comparison is None:
    st.warning("No eval results yet. Run `make reproduce` (or ./run.sh) first.")
    st.stop()

st.caption(
    f"Embedding model: {comparison['embedding_model']} | seed {comparison['seed']} | "
    f"generated {comparison['generated_at']}"
)

# ---- Strategy comparison -------------------------------------------------
st.header("Chunking-strategy comparison")
rows = []
for strat, m in comparison["strategies"].items():
    row = {"strategy": strat, **m["hit_rate"], "MRR": m["mrr"],
           "median_latency_ms": m["latency_ms"]["median"],
           "p95_latency_ms": m["latency_ms"]["p95"],
           "cost/query_usd": m["cost_usd_per_query"], "misses": m["n_misses"]}
    rows.append(row)
st.dataframe(pd.DataFrame(rows).set_index("strategy"), use_container_width=True)
st.success(f"Verdict: {comparison['verdict']}")

hr_df = pd.DataFrame({s: m["hit_rate"] for s, m in comparison["strategies"].items()})
st.subheader("Hit rate @ k")
st.bar_chart(hr_df)

# ---- Ingestion DQ --------------------------------------------------------
if ingestion:
    st.header("Ingestion data quality")
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents parsed", ingestion["documents_parsed"])
    c2.metric("Sections extracted", ingestion["total_sections"])
    c3.metric("Total DQ failures", ingestion["total_failures"])
    st.write("Failures by check:", ingestion["failures_by_check"])
    if ingestion["failures"]:
        st.dataframe(pd.DataFrame(ingestion["failures"]), use_container_width=True)

# ---- Groundedness --------------------------------------------------------
st.header("Groundedness (LLM-as-judge)")
if groundedness and groundedness.get("status") == "ran":
    c1, c2, c3 = st.columns(3)
    c1.metric("Grounded rate", groundedness["grounded_rate"])
    c2.metric("Mean score", groundedness["mean_groundedness_score"])
    c3.metric("LLM cost/query ($)", groundedness["llm_cost_usd_per_query"])
else:
    st.info((groundedness or {}).get("message", "Groundedness requires ANTHROPIC_API_KEY."))

# ---- Per-query detail ----------------------------------------------------
st.header("Per-query retrieval detail")
strat = st.selectbox("Strategy", config.STRATEGIES)
detail = _load_json(config.RESULTS_DIR / f"eval_{strat}.json")
if detail:
    st.dataframe(
        pd.DataFrame(detail["per_query"])[
            ["id", "category", "question", "gold_pep", "first_hit_rank", "latency_ms"]
        ],
        use_container_width=True,
    )

# ---- Monitoring log over time -------------------------------------------
st.header("Query monitoring log")
log_path = config.QUERY_LOG_PATH
if log_path.exists():
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if records:
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.caption(f"{len(df)} logged queries")
        df = df.sort_values("timestamp")
        st.line_chart(df.set_index("timestamp")[["latency_ms"]])
        st.dataframe(df.tail(50), use_container_width=True)
else:
    st.info("No query log yet.")
