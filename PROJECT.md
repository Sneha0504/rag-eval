# Document Retrieval and Evaluation Pipeline for GenAI

**Timeline:** Oct 2026 – Nov 2026
**Language:** Python
**Folder:** `/rag-eval`

## Goal
Prepare a messy document collection for an AI assistant, then measure how good the
retrieval actually is.

## Corpus
A public set of at least 300 documents with mixed structure (for example public
government policy PDFs or open software documentation). Record the source and
license in the README.

## Build

1. **Ingestion:** parse documents, clean them, extract metadata (source, section,
   date, document type), and run data quality checks (empty chunks, duplicates,
   encoding problems) with a logged report of every failure and its cause.
2. **Chunking and embeddings:** implement two chunking strategies (fixed size and
   section-aware) and embed with a sentence-transformers model. Store vectors plus
   metadata in Chroma or FAISS.
3. **Retrieval:** top-k search with metadata filtering, and a simple answer
   generator using an LLM API that cites the retrieved chunks.
4. **Evaluation:** write a golden set of 50 question and answer pairs with the
   source chunk for each. Report hit rate at k, mean reciprocal rank, groundedness
   (LLM-as-judge, spot-checked by hand on 20 answers), median latency, and cost per
   query. Compare the two chunking strategies and say which wins and why.
5. **Monitoring and docs:** log every query with latency and retrieved IDs, a small
   Streamlit or notebook view of eval results over time, and a README with design
   decisions, limitations and failure cases found.

## Constraints
- Use a fixed random seed.
- Pin dependencies.
- Include a one-command script to reproduce the evaluation.
- Do not invent numbers in the README; every figure must come from a real run.

## Decisions (this build)
- **Corpus:** Python Enhancement Proposals (PEPs) — 400+ public-domain `.rst`
  documents, mixed structure, rich metadata (number, title, author, status, type,
  created date), natural section headers. Public domain per PEP 1.
- **Python:** 3.12. **Embeddings:** sentence-transformers. **Vector store:** Chroma.
- **Offline-honest:** retrieval metrics (hit rate@k, MRR, latency) run fully offline.
  LLM answer-generation + groundedness judge use the Anthropic API only if a key is
  present, and degrade gracefully otherwise so no README figure is fabricated.
