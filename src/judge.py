"""Groundedness evaluation (LLM-as-judge) over generated answers, plus a
20-row manual spot-check CSV template.

The judge only runs when ANTHROPIC_API_KEY is set. Without a key:
  * the spot-check template is still written,
  * groundedness results are marked as requiring a key (no fabricated scores).
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone

import config
from src import evaluate, generate, retrieve

logger = logging.getLogger("judge")

JUDGE_SAMPLE_SIZE = 20


def _judge_prompt(question: str, answer: str, context: str) -> str:
    return (
        "You are grading whether an ANSWER is grounded in the provided CONTEXT.\n"
        "Grounded means every claim in the answer is supported by the context.\n\n"
        f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
        "Reply with a single JSON object: "
        '{"grounded": true|false, "score": 0-1 float, "reason": "..."}'
    )


def write_spotcheck_template(sample: list[dict]) -> None:
    path = config.RESULTS_DIR / "spotcheck_template.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "eval_id", "question", "gold_pep", "top_retrieved_pep",
            "generated_answer", "human_grounded_yes_no",
            "human_answer_correct_yes_no", "human_notes",
        ])
        for s in sample:
            w.writerow([
                s["id"], s["question"], s["gold_pep"], s.get("top_retrieved_pep", ""),
                s.get("generated_answer", ""), "", "", "",
            ])
    logger.info("Spot-check template (%d rows) -> %s", len(sample), path)


def run() -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    config.ensure_dirs()
    golden = evaluate.load_golden()
    config.set_seed()
    # Deterministic sample of 20 for groundedness + spot-check.
    sample_items = golden[:JUDGE_SAMPLE_SIZE]
    strategy = config.STRATEGY_SECTION  # judge against the interpretable strategy

    spot_rows = []
    judged = []
    key_present = config.has_api_key()
    total_cost = 0.0

    for item in sample_items:
        res = retrieve.retrieve(item["question"], strategy, k=5)
        top_pep = res["hits"][0]["metadata"].get("pep_number") if res["hits"] else None
        row = {
            "id": item["id"], "question": item["question"],
            "gold_pep": item["gold_pep"], "top_retrieved_pep": top_pep,
            "generated_answer": "",
        }
        if key_present:
            try:
                gen = generate.generate_answer(item["question"], res["hits"])
                row["generated_answer"] = gen["answer"]
                total_cost += gen["cost_usd"]
                context = "\n\n".join(f"[{h['chunk_id']}] {h['text']}" for h in res["hits"])
                import anthropic

                client = anthropic.Anthropic()
                jresp = client.messages.create(
                    model=config.LLM_MODEL,
                    max_tokens=300,
                    messages=[{"role": "user",
                               "content": _judge_prompt(item["question"], gen["answer"], context)}],
                )
                jtext = "".join(b.text for b in jresp.content if getattr(b, "type", "") == "text")
                total_cost += (
                    jresp.usage.input_tokens / 1e6 * config.LLM_PRICE_INPUT_PER_MTOK
                    + jresp.usage.output_tokens / 1e6 * config.LLM_PRICE_OUTPUT_PER_MTOK
                )
                try:
                    verdict = json.loads(jtext[jtext.find("{"): jtext.rfind("}") + 1])
                except Exception:
                    verdict = {"grounded": None, "score": None, "reason": jtext[:200]}
                judged.append({"id": item["id"], **verdict})
            except generate.NoApiKeyError as exc:
                key_present = False
                logger.warning("Generation unavailable: %s", exc)
        spot_rows.append(row)

    write_spotcheck_template(spot_rows)

    if key_present and judged:
        scores = [j["score"] for j in judged if isinstance(j.get("score"), (int, float))]
        grounded = [j for j in judged if j.get("grounded") is True]
        summary = {
            "status": "ran",
            "model": config.LLM_MODEL,
            "n_judged": len(judged),
            "grounded_rate": round(len(grounded) / len(judged), 4) if judged else None,
            "mean_groundedness_score": round(sum(scores) / len(scores), 4) if scores else None,
            "llm_cost_usd_total": round(total_cost, 6),
            "llm_cost_usd_per_query": round(total_cost / len(judged), 6) if judged else None,
            "per_item": judged,
        }
    else:
        summary = {
            "status": "requires_api_key",
            "message": (
                "Groundedness (LLM-as-judge) and answer generation require "
                "ANTHROPIC_API_KEY. Set it and re-run: "
                "`.venv/bin/python -m src.judge`. No groundedness numbers are "
                "fabricated. The 20-row spot-check template was still written to "
                "results/spotcheck_template.csv for manual review."
            ),
            "model": config.LLM_MODEL,
            "sample_size": len(sample_items),
        }

    out = config.RESULTS_DIR / "eval_groundedness.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), **summary},
                  fh, indent=2, ensure_ascii=False)
    logger.info("Groundedness summary (%s) -> %s", summary["status"], out)
    return summary


if __name__ == "__main__":
    run()
