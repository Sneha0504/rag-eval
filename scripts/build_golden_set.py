"""Build the 50-question golden set from REAL PEP content.

Every item is verified against data/processed/documents.json:
  * gold_doc_id must exist,
  * gold_section (when given) must be a real section heading of that PEP,
  * each `verify` phrase must actually appear in that PEP's text (case-insensitive).

If any check fails the script raises -- so no gold fact can be fabricated.
Run after ingestion. Output: data/golden_set.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402


def load_docs() -> dict[int, dict]:
    with config.DOCUMENTS_PATH.open(encoding="utf-8") as fh:
        docs = json.load(fh)
    return {d["pep_number"]: d for d in docs if d["pep_number"] is not None}


# Each entry: (pep, category, question, gold_section, gold_answer, [verify phrases])
# gold_answer for metadata uses the real header value (pulled from the doc at
# build time via the "@field" marker). verify phrases must appear in the PEP.
RAW: list[tuple] = [
    # ---- status (12) ----
    (8, "status", "What is the current status of the PEP that defines the style guide for Python code?", "", "@status", ["style guide"]),
    (20, "status", "Is the Zen of Python PEP active, final, or withdrawn?", "", "@status", ["Beautiful is better than ugly"]),
    (484, "status", "What is the status of the Type Hints PEP?", "", "@status", ["type hints"]),
    (572, "status", "What is the status of the PEP introducing assignment expressions (the walrus operator)?", "", "@status", ["assignment expression"]),
    (1, "status", "What is the status of the PEP describing PEP purpose and guidelines?", "", "@status", ["PEP"]),
    (257, "status", "What is the status of the Docstring Conventions PEP?", "", "@status", ["docstring"]),
    (343, "status", "What is the status of the PEP that introduced the with statement?", "", "@status", ["with statement"]),
    (585, "status", "What is the status of the PEP for type hinting generics in standard collections?", "", "@status", ["generic"]),
    (604, "status", "What is the status of the PEP allowing union types written with the pipe operator?", "", "@status", ["union"]),
    (492, "status", "What is the status of the PEP adding async and await coroutine syntax?", "", "@status", ["coroutine"]),
    (13, "status", "What is the status of the Python Language Governance PEP?", "", "@status", ["governance"]),
    (249, "status", "What is the status of the Python Database API Specification v2.0 PEP?", "", "@status", ["database"]),
    # ---- type (8) ----
    (20, "type", "What PEP type is the Zen of Python classified as?", "", "@type", ["Zen of Python"]),
    (255, "type", "What is the PEP type of the Simple Generators proposal?", "", "@type", ["generator"]),
    (3333, "type", "What PEP type is the Python Web Server Gateway Interface (WSGI) specification?", "", "@type", ["WSGI"]),
    (318, "type", "What PEP type is the proposal for decorators for functions and methods?", "", "@type", ["decorator"]),
    (544, "type", "What is the PEP type of the Protocols structural subtyping proposal?", "", "@type", ["Protocol"]),
    (561, "type", "What PEP type covers distributing and packaging type information?", "", "@type", ["type information"]),
    (8, "type", "What PEP type is the Style Guide for Python Code?", "", "@type", ["style guide"]),
    (517, "type", "What PEP type is the build-system independent format for source trees?", "", "@type", ["build"]),
    # ---- author (8) ----
    (8, "author", "Who authored the Style Guide for Python Code PEP?", "", "@author", ["Guido"]),
    (572, "author", "Who are the authors of the assignment expressions (walrus operator) PEP?", "", "@author", ["assignment expression"]),
    (484, "author", "Who authored the Type Hints PEP?", "", "@author", ["type hints"]),
    (343, "author", "Who wrote the PEP introducing the with statement?", "", "@author", ["with statement"]),
    (289, "author", "Who is the author of the Generator Expressions PEP?", "", "@author", ["generator expression"]),
    (380, "author", "Who wrote the PEP for syntax delegating to a subgenerator (yield from)?", "", "@author", ["yield from"]),
    (498, "author", "Who authored the PEP for literal string interpolation (f-strings)?", "", "@author", ["f-string"]),
    (20, "author", "Who is the author of the Zen of Python PEP?", "", "@author", ["Zen"]),
    # ---- created (7) ----
    (8, "created", "When was the Style Guide for Python Code PEP created?", "", "@created", ["style guide"]),
    (20, "created", "When was the Zen of Python PEP created?", "", "@created", ["Zen of Python"]),
    (257, "created", "When was the Docstring Conventions PEP created?", "", "@created", ["docstring"]),
    (318, "created", "On what date was the decorators for functions and methods PEP created?", "", "@created", ["decorator"]),
    (498, "created", "When was the f-strings (literal string interpolation) PEP created?", "", "@created", ["f-string"]),
    (517, "created", "When was the build-system independent source tree format PEP created?", "", "@created", ["pyproject"]),
    (484, "created", "When was the Type Hints PEP created?", "", "@created", ["type hint"]),
    # ---- content (15) ----
    (604, "content", "How can you write a union type using the vertical bar operator instead of typing.Union?", "Specification", None, ["int | str", "Union"]),
    (572, "content", "What operator does the walrus assignment expression use to assign within an expression?", "Abstract", None, [":="]),
    (8, "content", "How many spaces of indentation does the Python style guide recommend per level?", "Code Lay-out", None, ["4 spaces", "indentation"]),
    (8, "content", "What is the recommended maximum line length in the Python style guide?", "Maximum Line Length", None, ["79 characters"]),
    (20, "content", "According to the Zen of Python, what is better than implicit?", "", None, ["Explicit is better than implicit"]),
    (20, "content", "According to the Zen of Python, what should there preferably be one obvious way to do?", "", None, ["one-- and preferably only one --obvious way"]),
    (343, "content", "What pair of methods must a context manager object implement for the with statement?", "", None, ["__enter__", "__exit__"]),
    (498, "content", "What character prefix introduces a formatted string literal (f-string)?", "Abstract", None, ["f-string", "f'"]),
    (484, "content", "What syntax does PEP 484 use to annotate the types of function arguments and return values?", "Abstract", None, ["type hints", "annotation"]),
    (257, "content", "What kind of quotes does the docstring conventions PEP recommend for docstrings?", "", None, ['"""', "docstring"]),
    (255, "content", "What keyword does a generator function use to produce successive values?", "", None, ["yield"]),
    (380, "content", "What new expression does PEP 380 add for delegating to a subgenerator?", "", None, ["yield from"]),
    (492, "content", "What two keywords does PEP 492 add to declare and await coroutines?", "", None, ["async", "await"]),
    (585, "content", "What does PEP 585 let you use in place of typing.List and typing.Dict for generics?", "", None, ["list[", "dict["]),
    (3333, "content", "What callable interface does WSGI define between web servers and Python applications?", "", None, ["application", "environ", "start_response"]),
]


def main() -> None:
    docs = load_docs()
    items = []
    for idx, (pep, category, question, section, answer, verify) in enumerate(RAW, start=1):
        if pep not in docs:
            raise SystemExit(f"PEP {pep} not found in corpus")
        doc = docs[pep]
        body_lc = doc["body"].lower()
        headings = {s["heading"] for s in doc["sections"]}
        if section and section not in headings:
            raise SystemExit(
                f"q{idx:03d}: section '{section}' not a real heading of PEP {pep}. "
                f"Available: {sorted(h for h in headings if h)[:8]}"
            )
        # Resolve @field metadata answers to the real header value.
        if isinstance(answer, str) and answer.startswith("@"):
            field = answer[1:]
            answer = doc.get(field, "")
            if not answer:
                raise SystemExit(f"q{idx:03d}: PEP {pep} has no '{field}' value")
        # Verify every phrase actually appears in the PEP (title+body).
        hay = (doc["title"] + "\n" + doc["body"] + "\n" + doc["author"]).lower()
        for phrase in verify:
            if phrase.lower() not in hay:
                raise SystemExit(
                    f"q{idx:03d}: verify phrase {phrase!r} NOT found in PEP {pep} -- refusing to fabricate"
                )
        # For content questions with no explicit answer, use the first verify phrase
        # context as the gold answer anchor (kept short, drawn from real text).
        if answer is None:
            answer = "; ".join(verify)
        items.append({
            "id": f"q{idx:03d}",
            "question": question,
            "category": category,
            "gold_pep": pep,
            "gold_doc_id": doc["doc_id"],
            "gold_section": section,
            "gold_answer": answer,
            "gold_answer_phrases": verify,
        })

    if len(items) != 50:
        raise SystemExit(f"expected 50 items, built {len(items)}")

    dist: dict[str, int] = {}
    for it in items:
        dist[it["category"]] = dist.get(it["category"], 0) + 1

    payload = {
        "description": "50 question/answer pairs derived from real PEP content. "
                       "gold_pep/gold_doc_id is the source document; gold_section the "
                       "source section where applicable. Every fact verified against "
                       "the corpus at build time.",
        "corpus_source_url": config.CORPUS_SOURCE_URL,
        "n_items": len(items),
        "category_distribution": dist,
        "items": items,
    }
    with config.GOLDEN_SET_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(items)} golden items to {config.GOLDEN_SET_PATH}")
    print("Category distribution:", dist)


if __name__ == "__main__":
    main()
