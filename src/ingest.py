"""Ingestion: parse PEP .rst files, clean text, extract metadata, and run
data-quality checks.

Every data-quality failure is logged (human-readable via the logging module)
and written to results/ingestion_report.json with its cause. Nothing from the
downloaded repo is executed -- the .rst files are read as plain text only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import config

logger = logging.getLogger("ingest")

# RST section-adornment characters (a heading underline/overline is >=3 of one).
_ADORNMENT_CHARS = set("=-~^\"#*+`:._'<>")
_ADORNMENT_RE = re.compile(r"^([=\-~^\"#*+`:._'<>])\1{2,}\s*$")

# Header fields we extract from the RFC-2822-style PEP preamble.
_HEADER_FIELDS = {
    "pep": "pep",
    "title": "title",
    "author": "author",
    "status": "status",
    "type": "type",
    "created": "created",
    "python-version": "python_version",
}

# Mojibake signatures: UTF-8 bytes mis-decoded as latin-1/cp1252 leave these.
_MOJIBAKE_RE = re.compile(r"Ã[\x80-\xbf]|Â[\x80-\xbf]|â€|â€™|â€œ|â€\x9d|Ã©|Ã¨|Ã¼|Ã¶|Ã¤|Ãñ")
# Control chars except tab/newline/carriage-return.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class Document:
    doc_id: str
    source_path: str
    pep_number: int | None
    title: str
    author: str
    status: str
    type: str
    created: str
    python_version: str
    body: str
    sections: list[dict] = field(default_factory=list)
    raw_char_len: int = 0

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "pep_number": self.pep_number,
            "title": self.title,
            "author": self.author,
            "status": self.status,
            "type": self.type,
            "created": self.created,
            "python_version": self.python_version,
            "body": self.body,
            "sections": self.sections,
            "raw_char_len": self.raw_char_len,
        }


def _find_pep_files() -> list[Path]:
    for d in config.PEPS_SOURCE_DIRS:
        files = sorted(d.glob("pep-*.rst")) if d.exists() else []
        if files:
            return files
    return []


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _decode(raw: bytes) -> tuple[str, list[str]]:
    """Decode bytes, recording encoding problems. Returns (text, problems)."""
    problems: list[str] = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        problems.append(f"non-UTF8 bytes: {exc}")
        text = raw.decode("utf-8", errors="replace")
    if "�" in text:
        n_rep = text.count("�")
        problems.append(
            f"contains {n_rep} literal U+FFFD replacement character(s) "
            "(valid UTF-8, but the glyph is embedded in the source)"
        )
    if _MOJIBAKE_RE.search(text):
        problems.append("mojibake signature detected (double-encoded UTF-8)")
    ctrl = _CONTROL_RE.findall(text)
    if ctrl:
        problems.append(f"{len(ctrl)} disallowed control character(s)")
    return text, problems


def _clean_text(text: str) -> str:
    """Normalise unicode, strip control chars, collapse blank runs."""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ blank lines to 2.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_header(lines: list[str]) -> tuple[dict, int]:
    """Parse the RFC-2822 style preamble. Returns (fields, body_start_index)."""
    fields: dict[str, str] = {}
    current_key: str | None = None
    i = 0
    for i, line in enumerate(lines):
        if line.strip() == "" and fields:
            # First blank line after some headers ends the preamble.
            return fields, i + 1
        m = re.match(r"^([A-Za-z][A-Za-z0-9\- ]*?):\s?(.*)$", line)
        if m:
            key = m.group(1).strip().lower()
            val = m.group(2).strip()
            if key in _HEADER_FIELDS:
                current_key = _HEADER_FIELDS[key]
                fields[current_key] = val
            else:
                current_key = None
        elif current_key and (line.startswith(" ") or line.startswith("\t")):
            # Continuation line (e.g. multiple authors).
            fields[current_key] = (fields[current_key] + " " + line.strip()).strip()
        else:
            current_key = None
    return fields, i + 1


def _extract_sections(body: str) -> list[dict]:
    """Extract RST sections as {heading, text}. Handles underline and
    overline+underline adornment styles."""
    lines = body.split("\n")
    headings: list[tuple[int, str]] = []  # (line_index_of_heading_text, heading)
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < n else ""
        prev = lines[i - 1] if i > 0 else ""
        is_adorn = bool(_ADORNMENT_RE.match(line))
        next_adorn = bool(_ADORNMENT_RE.match(nxt))
        prev_adorn = bool(_ADORNMENT_RE.match(prev))
        # overline + text + underline
        if is_adorn and i + 2 < n and _ADORNMENT_RE.match(lines[i + 2]) \
                and lines[i + 1].strip() and not _ADORNMENT_RE.match(lines[i + 1]):
            headings.append((i + 1, lines[i + 1].strip()))
            i += 3
            continue
        # text + underline  (and the text line is itself not an adornment / not already consumed)
        if line.strip() and not is_adorn and next_adorn and not prev_adorn \
                and len(nxt.strip()) >= max(3, len(line.strip()) // 2):
            headings.append((i, line.strip()))
            i += 2
            continue
        i += 1

    if not headings:
        return [{"heading": "", "text": body.strip()}]

    sections: list[dict] = []
    # Preamble text before first heading.
    first_line = headings[0][0]
    # account for possible overline: find the line index just before heading text
    preamble = "\n".join(lines[: max(0, first_line - 1)]).strip()
    if preamble:
        sections.append({"heading": "", "text": preamble})
    for idx, (hline, heading) in enumerate(headings):
        start = hline + 2  # skip heading text + its underline
        end = headings[idx + 1][0] - 1 if idx + 1 < len(headings) else n
        # back off one for a potential overline of the next heading
        if idx + 1 < len(headings):
            nxt_hline = headings[idx + 1][0]
            if nxt_hline - 2 >= 0 and _ADORNMENT_RE.match(lines[nxt_hline - 2]):
                end = nxt_hline - 2
        text = "\n".join(lines[start:end]).strip()
        sections.append({"heading": heading, "text": text})
    return sections


def parse_document(path: Path) -> tuple[Document | None, list[str]]:
    """Parse one .rst file. Returns (Document or None, encoding_problems)."""
    raw = _read_bytes(path)
    text, enc_problems = _decode(raw)
    text = _clean_text(text)
    lines = text.split("\n")
    fields, body_start = _parse_header(lines)
    body = "\n".join(lines[body_start:]).strip()
    sections = _extract_sections(body)

    m = re.search(r"pep-(\d+)", path.stem)
    pep_number: int | None = int(m.group(1)) if m else None
    if not pep_number and fields.get("pep", "").isdigit():
        pep_number = int(fields["pep"])

    doc = Document(
        doc_id=path.stem,
        source_path=str(path.relative_to(config.ROOT)) if config.ROOT in path.parents else str(path),
        pep_number=pep_number,
        title=fields.get("title", ""),
        author=fields.get("author", ""),
        status=fields.get("status", ""),
        type=fields.get("type", ""),
        created=fields.get("created", ""),
        python_version=fields.get("python_version", ""),
        body=body,
        sections=sections,
        raw_char_len=len(raw),
    )
    return doc, enc_problems


def _normalise_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _shingles(text: str, k: int = 5) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def run_data_quality(docs: list[Document]) -> list[dict]:
    """Document-level DQ: empty/trivial, exact duplicates, near duplicates,
    missing metadata. Returns a list of failure records."""
    failures: list[dict] = []

    # Empty / trivial body.
    for d in docs:
        if len(d.body.strip()) < config.MIN_CHUNK_CHARS:
            failures.append({
                "doc_id": d.doc_id, "check": "empty_chunk",
                "cause": f"body has {len(d.body.strip())} chars (< {config.MIN_CHUNK_CHARS})",
            })

    # Exact duplicates (by normalised-body hash).
    seen: dict[str, str] = {}
    for d in docs:
        h = hashlib.sha256(_normalise_for_dedup(d.body).encode("utf-8")).hexdigest()
        if h in seen:
            failures.append({
                "doc_id": d.doc_id, "check": "exact_duplicate",
                "cause": f"identical normalised body to {seen[h]} (sha256={h[:12]})",
            })
        else:
            seen[h] = d.doc_id

    # Near-duplicates via 5-gram shingle Jaccard (>=0.85 and not exact-dup).
    exact_dup_ids = {f["doc_id"] for f in failures if f["check"] == "exact_duplicate"}
    shingle_map = {d.doc_id: _shingles(d.body) for d in docs}
    ordered = [d for d in docs if d.doc_id not in exact_dup_ids]
    for i, d in enumerate(ordered):
        for other in ordered[i + 1 :]:
            sim = _jaccard(shingle_map[d.doc_id], shingle_map[other.doc_id])
            if sim >= 0.85:
                failures.append({
                    "doc_id": other.doc_id, "check": "near_duplicate",
                    "cause": f"shingle Jaccard {sim:.3f} vs {d.doc_id} (>=0.85)",
                })

    # Missing key metadata.
    for d in docs:
        missing = [k for k in ("title", "status", "type") if not getattr(d, k).strip()]
        if missing:
            failures.append({
                "doc_id": d.doc_id, "check": "missing_metadata",
                "cause": f"missing header field(s): {', '.join(missing)}",
            })

    return failures


def ingest() -> dict:
    """Run the full ingestion + DQ pass. Writes documents.json and the report."""
    config.set_seed()
    config.ensure_dirs()

    # Configure logging to file + console.
    config.INGESTION_LOG_PATH.unlink(missing_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(config.INGESTION_LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    paths = _find_pep_files()
    logger.info("Found %d candidate .rst files", len(paths))
    if not paths:
        raise FileNotFoundError(
            f"No pep-*.rst files under {config.PEPS_SOURCE_DIRS}. Run the corpus "
            "clone step in run.sh first."
        )

    docs: list[Document] = []
    encoding_failures: list[dict] = []
    parse_failures: list[dict] = []

    for p in sorted(paths):
        try:
            doc, enc_problems = parse_document(p)
        except Exception as exc:  # pragma: no cover - defensive
            parse_failures.append({"doc_id": p.stem, "check": "parse_error", "cause": str(exc)})
            logger.error("PARSE FAIL %s: %s", p.stem, exc)
            continue
        for prob in enc_problems:
            rec = {"doc_id": doc.doc_id, "check": "encoding", "cause": prob}
            encoding_failures.append(rec)
            logger.warning("ENCODING %s: %s", doc.doc_id, prob)
        if doc is not None:
            docs.append(doc)

    logger.info("Parsed %d documents", len(docs))

    dq_failures = run_data_quality(docs)
    for f in dq_failures:
        logger.warning("DQ[%s] %s: %s", f["check"], f["doc_id"], f["cause"])

    all_failures = encoding_failures + parse_failures + dq_failures
    by_check: dict[str, int] = {}
    for f in all_failures:
        by_check[f["check"]] = by_check.get(f["check"], 0) + 1

    # Persist cleaned documents for downstream steps.
    with config.DOCUMENTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump([d.to_dict() for d in docs], fh, ensure_ascii=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_source_url": config.CORPUS_SOURCE_URL,
        "corpus_license": config.CORPUS_LICENSE,
        "files_found": len(paths),
        "documents_parsed": len(docs),
        "total_sections": sum(len(d.sections) for d in docs),
        "failures_by_check": by_check,
        "total_failures": len(all_failures),
        "failures": all_failures,
    }
    with config.INGESTION_REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Ingestion complete: %d docs, %d failures %s",
        len(docs), len(all_failures), by_check,
    )
    return report


if __name__ == "__main__":
    ingest()
