#!/usr/bin/env bash
# One-command reproduce: corpus -> ingest -> index -> golden set -> evaluate.
# Paths are quoted throughout because this project lives under a path with a
# space ("Coding Projects").
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Deterministic hashing for any set/dict-order-sensitive code path.
export PYTHONHASHSEED=0

PY="$HERE/.venv/bin/python"
PIP="$HERE/.venv/bin/pip"

# 1. Virtual environment + pinned dependencies -----------------------------
if [ ! -x "$PY" ]; then
  echo ">> Creating virtual environment (.venv)"
  /usr/local/bin/python3.12 -m venv "$HERE/.venv"
  "$PIP" install --quiet --upgrade pip
  "$PIP" install --quiet -r "$HERE/requirements.txt"
fi

# 2. Corpus: shallow-clone the PEPs repo if absent -------------------------
REPO_DIR="$HERE/data/raw/peps-repo"
if [ ! -d "$REPO_DIR/peps" ] && [ ! -e "$REPO_DIR/pep-0001.rst" ]; then
  echo ">> Cloning PEP corpus (shallow)"
  mkdir -p "$HERE/data/raw"
  git clone --depth 1 https://github.com/python/peps.git "$REPO_DIR"
fi
echo ">> Corpus commit: $(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"

# 3. Pipeline --------------------------------------------------------------
echo ">> [1/5] Ingestion + data-quality checks"
"$PY" -m src.ingest

echo ">> [2/5] Building golden set (verified against corpus)"
"$PY" scripts/build_golden_set.py

echo ">> [3/5] Chunking + embedding + indexing (both strategies)"
"$PY" -m src.embed_index

echo ">> [4/5] Retrieval evaluation (both strategies)"
"$PY" -m src.evaluate

echo ">> Groundedness / spot-check (LLM judge runs only if ANTHROPIC_API_KEY is set)"
"$PY" -m src.judge

echo ">> [5/5] Rendering README.md + corpus provenance from the real results"
"$PY" scripts/render_readme.py

echo ""
echo ">> Done. Real outputs are in results/:"
ls -1 "$HERE/results"
