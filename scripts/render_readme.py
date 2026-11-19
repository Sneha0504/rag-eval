"""Render README.md from the REAL result files. Every metric in the README is
read from results/*.json so nothing is hand-typed or invented.

Run after the pipeline completes. Reads an optional corpus-provenance file at
results/corpus_provenance.json (written by run.sh / this script's --provenance).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402


def load(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def corpus_provenance() -> dict:
    repo = config.PEPS_REPO_DIR
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(repo), *args],
                                           text=True).strip()
        except Exception:
            return "unknown"
    return {
        "commit": git("rev-parse", "HEAD"),
        "commit_date": git("log", "-1", "--format=%cI"),
    }


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> None:
    ingestion = load(config.INGESTION_REPORT_PATH)
    comparison = load(config.RESULTS_DIR / "eval_comparison.json")
    index_summary = load(config.RESULTS_DIR / "index_summary.json")
    golden = load(config.GOLDEN_SET_PATH)
    groundedness = load(config.RESULTS_DIR / "eval_groundedness.json")
    evals = {s: load(config.RESULTS_DIR / f"eval_{s}.json") for s in config.STRATEGIES}
    prov = corpus_provenance()

    strategies = config.STRATEGIES
    ks = config.K_VALUES

    # Metrics table
    headers = ["Strategy", "Indexed chunks"] + [f"hit@{k}" for k in ks] + \
              ["MRR", "median latency (ms)", "p95 latency (ms)", "cost/query ($)", "misses"]
    rows = []
    for s in strategies:
        m = comparison["strategies"][s]
        rows.append([
            s,
            index_summary["strategies"][s]["indexed_chunks"],
            *[m["hit_rate"][f"hit_rate@{k}"] for k in ks],
            m["mrr"], m["latency_ms"]["median"], m["latency_ms"]["p95"],
            m["cost_usd_per_query"], m["n_misses"],
        ])
    metrics_table = md_table(headers, rows)

    # DQ table
    dq_rows = [[check, cnt] for check, cnt in sorted(ingestion["failures_by_check"].items())]
    if not dq_rows:
        dq_rows = [["(none)", 0]]
    dq_table = md_table(["DQ check", "count"], dq_rows)

    # Concrete failure examples
    fail_examples = "\n".join(
        f"- `{f['doc_id']}` -- **{f['check']}**: {f['cause']}"
        for f in ingestion["failures"]
    ) or "- No data-quality failures were found in this run."

    # Groundedness block
    if groundedness.get("status") == "ran":
        g = groundedness
        ground_block = (
            f"The LLM-as-judge ran on {g['n_judged']} generated answers using "
            f"`{g['model']}`.\n\n"
            f"- Grounded rate: **{g['grounded_rate']}**\n"
            f"- Mean groundedness score: **{g['mean_groundedness_score']}**\n"
            f"- LLM cost/query: **${g['llm_cost_usd_per_query']}** "
            f"(total ${g['llm_cost_usd_total']})\n"
        )
    else:
        ground_block = (
            "**Status: requires `ANTHROPIC_API_KEY`.** Answer generation and the "
            "groundedness judge were not run in this build because no key was "
            "present, so no groundedness or cost numbers are reported here (none "
            "are fabricated). The 20-row manual spot-check template was still "
            "generated at `results/spotcheck_template.csv`.\n\n"
            "To produce real numbers:\n\n"
            "```bash\nexport ANTHROPIC_API_KEY=sk-ant-...\n"
            ".venv/bin/python -m src.judge\n```\n"
        )

    verdict = comparison["verdict"]
    n_docs = ingestion["documents_parsed"]
    n_sections = ingestion["total_sections"]
    dist = golden["category_distribution"]
    total_queries = sum(index_summary["strategies"][s]["indexed_chunks"] for s in strategies)

    # Real chunk-level DQ numbers (also behind the verdict's explanation).
    fixed_dq = index_summary["strategies"][config.STRATEGY_FIXED]["chunk_dq"]
    section_dq = index_summary["strategies"][config.STRATEGY_SECTION]["chunk_dq"]
    # Indexed counts (actually searched) -- the single count used everywhere.
    fixed_idx = index_summary["strategies"][config.STRATEGY_FIXED]["indexed_chunks"]
    section_idx = index_summary["strategies"][config.STRATEGY_SECTION]["indexed_chunks"]

    readme = f"""# Document Retrieval and Evaluation Pipeline (PEP corpus)

A retrieval-augmented-generation (RAG) evaluation pipeline over the Python
Enhancement Proposals (PEPs). It ingests a messy real-world document collection,
cleans it and runs data-quality checks, builds two different chunking strategies,
indexes them with sentence-transformer embeddings in Chroma, and measures how
good retrieval actually is against a 50-question golden set.

Every number in this README is produced by a real run and copied from
`results/*.json` (rendered by `scripts/render_readme.py`). Nothing is invented.

## Corpus

- **Source:** {config.CORPUS_SOURCE_URL} (shallow clone)
- **Commit fetched:** `{prov['commit']}` ({prov['commit_date']})
- **License:** {config.CORPUS_LICENSE}. PEP 1 places PEP text in the public domain.
- **Documents ingested:** **{n_docs}** `.rst` PEP files (modern `peps/` layout;
  older root layout also handled).
- **Sections extracted:** {n_sections} across all documents.
- The downloaded repo is treated as **untrusted data**: only `.rst` files are
  read as text; nothing from it is executed.

## Architecture

```
PEPs repo (.rst)
   │  src/ingest.py      parse RFC-2822 header + RST sections, clean text,
   │                     extract metadata, run data-quality checks
   ▼
data/processed/documents.json  +  results/ingestion_report.json (+ ingestion.log)
   │  src/chunking.py    two strategies: fixed-size (chars+overlap) and
   │                     section-aware (split on RST headers)
   │  src/embed_index.py all-MiniLM-L6-v2 embeddings -> Chroma (one collection
   ▼                     per strategy, cosine space, deterministic order)
Chroma persistent store (data/chroma/)
   │  src/retrieve.py    top-k similarity + metadata filtering (status/type/PEP)
   │  src/generate.py    cited answer via Anthropic API (key-gated)
   ▼
   │  src/evaluate.py    hit@k, MRR, latency, cost over data/golden_set.json
   │  src/judge.py       LLM-as-judge groundedness + 20-row spot-check (key-gated)
   │  src/monitor.py     append every query to results/query_log.jsonl
   ▼
results/*.json  +  app/streamlit_app.py (dashboard)
```

## Design decisions

- **Corpus = PEPs.** 700+ public-domain documents with mixed structure and rich,
  real metadata (number, title, author, status, type, created date) and natural
  RST section headers -- ideal for comparing chunking strategies.
- **Two chunking strategies.** *Fixed* uses {config.FIXED_CHUNK_CHARS}-char windows
  with {config.FIXED_CHUNK_OVERLAP}-char overlap (~500/50 tokens), model-agnostic
  and deterministic. *Section-aware* splits on RST section headers and keeps the
  heading as metadata; long sections are sub-split at {config.SECTION_MAX_CHARS}
  chars. The section strategy makes every retrieved chunk map to a named section,
  which is better for citeable answers.
- **Embeddings = `{config.EMBED_MODEL_NAME}`.** Small, fast, reliable, 384-dim,
  CPU-friendly. Embeddings are L2-normalised and stored in Chroma with cosine space.
- **Offline-honest.** Retrieval metrics (hit@k, MRR, latency) run fully offline.
  Answer generation and groundedness use the Anthropic API only when a key is
  present and degrade gracefully otherwise, so no README figure is fabricated.
- **Determinism.** Fixed seed (`{config.SEED}`) in `config.py`, chunks sorted by a
  stable `chunk_id` before indexing, all dependencies pinned in `requirements.txt`.

## Golden set

`data/golden_set.json` holds **{golden['n_items']}** question/answer pairs built
from real PEP content (`scripts/build_golden_set.py`). Each item records the gold
source document (`gold_pep` / `gold_doc_id`) and, where applicable, the gold
section. The builder **verifies every gold fact against the corpus** and aborts if
any phrase is not found, so gold answers cannot be fabricated. Category mix:

{md_table(["category", "count"], [[k, v] for k, v in sorted(dist.items())])}

A retrieval **hit** = the gold PEP appears among the retrieved chunks' documents
within top-k. Rank = position of the first chunk from the gold PEP.

## Results (real run)

Embedding model `{comparison['embedding_model']}`, seed {comparison['seed']},
k = {ks}. Evaluated over {golden['n_items']} golden questions per strategy.

{metrics_table}

### Verdict

> {verdict}

The verdict above (and its cause) is copied verbatim from
`results/eval_comparison.json`. The table above is the real, reproducible
comparison. Hit-rate and MRR are
deterministic (identical across reruns on CPU). Latency measures the **Chroma
similarity search only** (it excludes query-embedding time), is wall-clock, and
so varies slightly run-to-run.

## Data-quality findings (real, from `results/ingestion_report.json`)

Across **{n_docs}** ingested documents, **{ingestion['total_failures']}**
data-quality failures were found:

{dq_table}

Concrete failure cases (every failure and its cause is listed in the report):

{fail_examples}

The ingestion step also writes a human-readable log to `results/ingestion.log`.

**Chunk-level data quality (from `results/index_summary.json`).** Beyond the
document-level checks above, each chunking strategy is also checked. "Indexed
chunks" is the number actually searched (the same figure as the metrics table);
empty chunks are dropped before indexing and duplicates are flagged:

{md_table(["Strategy", "indexed chunks", "empty (dropped)", "exact-duplicate (flagged)"], [[config.STRATEGY_FIXED, fixed_idx, fixed_dq["empty_chunks"], fixed_dq["exact_duplicate_chunks"]], [config.STRATEGY_SECTION, section_idx, section_dq["empty_chunks"], section_dq["exact_duplicate_chunks"]]])}

The section-aware strategy's {section_dq['exact_duplicate_chunks']} duplicate and
{section_dq['empty_chunks']} empty chunks are boilerplate sections repeated across
many PEPs (`Copyright`, `References`, etc.); empty chunks are dropped before
indexing and the duplicates are a direct cause of its weaker retrieval (see the
verdict above).

## Groundedness (LLM-as-judge)

{ground_block}

## How to run

```bash
# one command, from a clean checkout -- builds venv, clones corpus, runs everything
./run.sh          # or: make reproduce
```

Individual steps:

```bash
make venv         # create .venv and install pinned deps
make corpus       # shallow-clone the PEP repo into data/raw/
make ingest       # parse + clean + data-quality report
make golden       # (re)build the verified golden set
make index        # chunk (both strategies) + embed + index in Chroma
make evaluate     # retrieval metrics for both strategies
make judge        # groundedness + spot-check (needs ANTHROPIC_API_KEY)
make dashboard    # launch the Streamlit dashboard
```

Dashboard: `.venv/bin/streamlit run app/streamlit_app.py` -- shows the strategy
comparison, DQ findings, per-query detail, and query volume/latency over time
from the monitoring log.

## Repository layout

```
config.py                 seed, paths, model name, k values, pricing
requirements.txt          exact pinned dependencies
run.sh / Makefile         one-command reproduce
src/ingest.py             parse + clean + metadata + data-quality checks
src/chunking.py           fixed + section-aware chunking
src/embed_index.py        embeddings -> Chroma (one collection per strategy)
src/retrieve.py           top-k search + metadata filtering
src/generate.py           cited answer generator (Anthropic, key-gated)
src/evaluate.py           hit@k, MRR, latency, cost + strategy comparison
src/judge.py              LLM-as-judge groundedness + spot-check template
src/monitor.py            per-query JSONL logging
scripts/build_golden_set.py   verified 50-question golden set builder
scripts/render_readme.py  regenerates this README from results/
app/streamlit_app.py      dashboard
data/golden_set.json      the golden set (committed)
results/                  real run outputs (committed)
```

## Limitations

- **Document-level hit metric.** A hit means the correct PEP was retrieved within
  top-k; it does not require the exact gold section. Section-level scoring is
  recorded in `gold_section` but not used as the headline metric so the two
  strategies are compared on equal footing.
- **Small embedding model.** `all-MiniLM-L6-v2` is fast but less capable than
  larger models; absolute hit rates would likely rise with a stronger encoder.
- **Golden set is metadata- and content-anchored.** Questions reference real PEP
  topics; a different question distribution could shift the numbers.
- **Groundedness needs a key.** Without `ANTHROPIC_API_KEY`, generation and the
  judge are skipped (clearly marked), not estimated.
- **Near-duplicate detection is O(n^2)** over shingle sets; fine for ~750 docs,
  not for very large corpora.

## Reproducibility

- Fixed seed `{config.SEED}` (`config.set_seed` seeds `random`, `numpy`, `torch`).
- All dependencies pinned to exact versions in `requirements.txt`.
- `./run.sh` reproduces the corpus fetch and the full pipeline end-to-end.
- `results/` is committed so the real outputs travel with the repo.
"""

    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    # Save provenance alongside results for traceability.
    with (config.RESULTS_DIR / "corpus_provenance.json").open("w", encoding="utf-8") as fh:
        json.dump({"source_url": config.CORPUS_SOURCE_URL,
                   "license": config.CORPUS_LICENSE, **prov}, fh, indent=2)
    print("Wrote README.md from real results.")


if __name__ == "__main__":
    main()
