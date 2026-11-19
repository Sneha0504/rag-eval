# Document Retrieval and Evaluation Pipeline (PEP corpus)

A retrieval-augmented-generation (RAG) evaluation pipeline over the Python
Enhancement Proposals (PEPs). It ingests a messy real-world document collection,
cleans it and runs data-quality checks, builds two different chunking strategies,
indexes them with sentence-transformer embeddings in Chroma, and measures how
good retrieval actually is against a 50-question golden set.

Every number in this README is produced by a real run and copied from
`results/*.json` (rendered by `scripts/render_readme.py`). Nothing is invented.

## Corpus

- **Source:** https://github.com/python/peps (shallow clone)
- **Commit fetched:** `3ab1034e406efe7fd5e2295b83847b6f2d9f6f49` (2026-11-18T15:37:38Z)
- **License:** Public domain (per PEP 1). PEP 1 places PEP text in the public domain.
- **Documents ingested:** **748** `.rst` PEP files (modern `peps/` layout;
  older root layout also handled).
- **Sections extracted:** 14787 across all documents.
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
   │  src/generate.py    cited answer via a pluggable LLM provider (key-gated)
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
- **Two chunking strategies.** *Fixed* uses 2000-char windows
  with 200-char overlap (~500/50 tokens), model-agnostic
  and deterministic. *Section-aware* splits on RST section headers and keeps the
  heading as metadata; long sections are sub-split at 2600
  chars. The section strategy makes every retrieved chunk map to a named section,
  which is better for citeable answers.
- **Embeddings = `sentence-transformers/all-MiniLM-L6-v2`.** Small, fast, reliable, 384-dim,
  CPU-friendly. Embeddings are L2-normalised and stored in Chroma with cosine space.
- **Offline-honest.** Retrieval metrics (hit@k, MRR, latency) run fully offline.
  Answer generation and groundedness call a pluggable LLM provider (default:
  Google Gemini's **free** tier via its OpenAI-compatible endpoint; Anthropic
  and other OpenAI-compatible back ends are selectable) only when a key is
  present, and degrade gracefully otherwise, so no README figure is fabricated.
- **Determinism.** Fixed seed (`42`) in `config.py`, chunks sorted by a
  stable `chunk_id` before indexing, all dependencies pinned in `requirements.txt`.

## Golden set

`data/golden_set.json` holds **50** question/answer pairs built
from real PEP content (`scripts/build_golden_set.py`). Each item records the gold
source document (`gold_pep` / `gold_doc_id`) and, where applicable, the gold
section. The builder **verifies every gold fact against the corpus** and aborts if
any phrase is not found, so gold answers cannot be fabricated. Category mix:

| category | count |
| --- | --- |
| author | 8 |
| content | 15 |
| created | 7 |
| status | 12 |
| type | 8 |

A retrieval **hit** = the gold PEP appears among the retrieved chunks' documents
within top-k. Rank = position of the first chunk from the gold PEP.

## Results (real run)

Embedding model `sentence-transformers/all-MiniLM-L6-v2`, seed 42,
k = [1, 3, 5, 10]. Evaluated over 50 golden questions per strategy.

| Strategy | Indexed chunks | hit@1 | hit@3 | hit@5 | hit@10 | MRR | median latency (ms) | p95 latency (ms) | cost/query ($) | misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 8301 | 0.54 | 0.7 | 0.74 | 0.84 | 0.6288 | 7.347 | 19.177 | 0.0 | 8 |
| section | 15119 | 0.4 | 0.62 | 0.7 | 0.74 | 0.5136 | 7.545 | 12.209 | 0.0 | 13 |

### Verdict

> Winner: fixed chunking. It scores MRR 0.6288 vs 0.5136 and hit_rate@5 0.74 vs 0.7. Median Chroma-search latency 7.347 ms vs 7.545 ms. Why: section-aware chunking indexed 15119 chunks but its data quality is weaker -- 844 exact-duplicate and 112 empty chunks were flagged (empties dropped before indexing); many remaining chunks are tiny header-only fragments (Abstract, Copyright, References) that dilute the top-k and split a PEP's content across many small vectors. Fixed-size chunking indexed 8301 denser 2000-char windows (0 duplicate, 1 empty), so each chunk carries more context and matches the question topic more reliably -- hence the higher MRR and hit@k.

The verdict above (and its cause) is copied verbatim from
`results/eval_comparison.json`. The table above is the real, reproducible
comparison. Hit-rate and MRR are
deterministic (identical across reruns on CPU). Latency measures the **Chroma
similarity search only** (it excludes query-embedding time), is wall-clock, and
so varies slightly run-to-run.

## Data-quality findings (real, from `results/ingestion_report.json`)

Across **748** ingested documents, **4**
data-quality failures were found:

| DQ check | count |
| --- | --- |
| empty_chunk | 1 |
| encoding | 2 |
| near_duplicate | 1 |

Concrete failure cases (every failure and its cause is listed in the report):

- `pep-0538` -- **encoding**: contains 25 literal U+FFFD replacement character(s) (valid UTF-8, but the glyph is embedded in the source)
- `pep-3117` -- **encoding**: contains 2 literal U+FFFD replacement character(s) (valid UTF-8, but the glyph is embedded in the source)
- `pep-0210` -- **empty_chunk**: body has 13 chars (< 40)
- `pep-3134` -- **near_duplicate**: shingle Jaccard 0.957 vs pep-0344 (>=0.85)

The ingestion step also writes a human-readable log to `results/ingestion.log`.

**Chunk-level data quality (from `results/index_summary.json`).** Beyond the
document-level checks above, each chunking strategy is also checked. "Indexed
chunks" is the number actually searched (the same figure as the metrics table);
empty chunks are dropped before indexing and duplicates are flagged:

| Strategy | indexed chunks | empty (dropped) | exact-duplicate (flagged) |
| --- | --- | --- | --- |
| fixed | 8301 | 1 | 0 |
| section | 15119 | 112 | 844 |

The section-aware strategy's 844 duplicate and
112 empty chunks are boilerplate sections repeated across
many PEPs (`Copyright`, `References`, etc.); empty chunks are dropped before
indexing and the duplicates are a direct cause of its weaker retrieval (see the
verdict above).

## Groundedness (LLM-as-judge)

The LLM-as-judge ran on 20 generated answers using `openai/gpt-oss-20b` via provider `groq` (free tier -- $0 cost, real token usage still recorded).

- Grounded rate: **0.75**
- Mean groundedness score: **0.7425**
- LLM cost/query: **$0.0** (total $0.0)


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
make judge        # groundedness + spot-check (needs an LLM API key)
make dashboard    # launch the Streamlit dashboard
```

### LLM provider for answer generation + groundedness

Generation and the judge use a pluggable provider, selected with env vars. The
default is **Google Gemini's free tier** (OpenAI-compatible endpoint), so a full
keyed run costs **$0**:

```bash
export GEMINI_API_KEY=...        # free key from Google AI Studio
./run.sh                         # or: .venv/bin/python -m src.judge
```

| env var | purpose | default |
| --- | --- | --- |
| `LLM_PROVIDER` | `gemini`, `anthropic`, `openai`, or `custom` | `gemini` |
| `GEMINI_API_KEY` | key for the default Gemini free tier | (unset) |
| `ANTHROPIC_API_KEY` | key when `LLM_PROVIDER=anthropic` | (unset) |
| `OPENAI_API_KEY` | key when `LLM_PROVIDER=openai` | (unset) |
| `LLM_API_KEY` | generic key override (any provider) | (unset) |
| `LLM_MODEL` | override the model | per provider |
| `LLM_BASE_URL` | override the OpenAI-compatible endpoint | per provider |
| `LLM_PRICE_INPUT_PER_MTOK` / `LLM_PRICE_OUTPUT_PER_MTOK` | price (USD / 1M tokens) for real cost | per provider |

Gemini's free tier is priced at $0, so cost/query is reported as $0 while the
real token usage returned by the API is still recorded. To keep the original
Anthropic path: `export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-...`.

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
src/generate.py           cited answer generator (pluggable LLM, key-gated)
src/llm_client.py         provider-agnostic LLM call layer (Gemini/Anthropic/OpenAI)
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
- **Groundedness needs a key.** Without an LLM API key (default provider:
  Gemini's free tier via `GEMINI_API_KEY`), generation and the judge are skipped
  (clearly marked), not estimated.
- **Near-duplicate detection is O(n^2)** over shingle sets; fine for ~750 docs,
  not for very large corpora.

## Reproducibility

- Fixed seed `42` (`config.set_seed` seeds `random`, `numpy`, `torch`).
- All dependencies pinned to exact versions in `requirements.txt`.
- `./run.sh` reproduces the corpus fetch and the full pipeline end-to-end.
- `results/` is committed so the real outputs travel with the repo.
