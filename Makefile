# RAG evaluation pipeline. The project path contains a space, so recipes quote
# paths and use the venv interpreter directly.
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Deterministic hashing for any set/dict-order-sensitive code path.
export PYTHONHASHSEED := 0

.DEFAULT_GOAL := reproduce

.PHONY: reproduce venv corpus ingest golden index evaluate judge readme dashboard clean

reproduce:
	./run.sh

venv:
	/usr/local/bin/python3.12 -m venv "$(VENV)"
	"$(PIP)" install --quiet --upgrade pip
	"$(PIP)" install --quiet -r requirements.txt

corpus:
	mkdir -p data/raw
	test -d data/raw/peps-repo || git clone --depth 1 https://github.com/python/peps.git data/raw/peps-repo

ingest:
	"$(PY)" -m src.ingest

golden:
	"$(PY)" scripts/build_golden_set.py

index:
	"$(PY)" -m src.embed_index

evaluate:
	"$(PY)" -m src.evaluate

judge:
	"$(PY)" -m src.judge

readme:
	"$(PY)" scripts/render_readme.py

dashboard:
	"$(VENV)/bin/streamlit" run app/streamlit_app.py

clean:
	rm -rf data/processed data/chroma results/query_log.jsonl
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
