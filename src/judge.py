"""Groundedness evaluation (LLM-as-judge) over generated answers, plus a
20-row manual spot-check CSV template.

The judge only runs when an API key is set for the selected LLM provider
(default: Google Gemini's free tier; see config.py). Without a key:
  * the spot-check template is still written,
  * groundedness results are marked as requiring a key (no fabricated scores).
"""
from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime, timezone

import config
from src import evaluate, generate, llm_client, retrieve

logger = logging.getLogger("judge")

JUDGE_SAMPLE_SIZE = 20


def _judge_prompt(question: str, answer: str, context: str) -> str:
    return (
        "You are grading whether an ANSWER is grounded in the provided CONTEXT.\n"
        "Grounded means every claim in the answer is supported by the context.\n\n"
        f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
        "Reply with ONLY a single JSON object and nothing else (no preamble, no "
        'reasoning outside it): {"grounded": true|false, "score": 0-1 float, '
        '"reason": "one short sentence"}'
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
    failures = 0

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
                jres = llm_client.chat(
                    user=_judge_prompt(item["question"], gen["answer"], context),
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )
                jtext = jres["text"]
                total_cost += jres["cost_usd"]
                try:
                    verdict = json.loads(jtext[jtext.find("{"): jtext.rfind("}") + 1])
                except Exception:
                    verdict = {"grounded": None, "score": None, "reason": jtext[:200]}
                judged.append({"id": item["id"], **verdict})
            except generate.NoApiKeyError as exc:
                key_present = False
                logger.warning("Generation unavailable: %s", exc)
            except Exception as exc:  # transient API error (429/503/timeout etc.)
                failures += 1
                logger.warning(
                    "LLM call failed for %s after retries (%s); skipping this "
                    "item and continuing.", item["id"], type(exc).__name__
                )
            if config.LLM_REQUEST_PAUSE_S:
                time.sleep(config.LLM_REQUEST_PAUSE_S)
        spot_rows.append(row)

    write_spotcheck_template(spot_rows)

    if key_present and judged:
        scores = [j["score"] for j in judged if isinstance(j.get("score"), (int, float))]
        grounded = [j for j in judged if j.get("grounded") is True]
        cost_note = (
            f"Provider '{config.LLM_PROVIDER}' is a $0 free tier, so cost/query "
            f"is $0 even though real token usage was recorded."
            if config.LLM_IS_FREE_TIER else
            "Cost is computed from the real token counts returned by the API."
        )
        summary = {
            "status": "ran",
            "provider": config.LLM_PROVIDER,
            "free_tier": config.LLM_IS_FREE_TIER,
            "model": config.LLM_MODEL,
            "n_judged": len(judged),
            "n_failed": failures,
            "sample_size": len(sample_items),
            "grounded_rate": round(len(grounded) / len(judged), 4) if judged else None,
            "mean_groundedness_score": round(sum(scores) / len(scores), 4) if scores else None,
            "llm_cost_usd_total": round(total_cost, 6),
            "llm_cost_usd_per_query": round(total_cost / len(judged), 6) if judged else None,
            "cost_note": cost_note,
            "per_item": judged,
        }
        if failures:
            summary["note"] = (
                f"{failures} of {len(sample_items)} items were skipped after the "
                f"provider returned repeated transient errors (e.g. 429/503); the "
                f"rates above are over the {len(judged)} items that succeeded."
            )
    elif key_present and failures:
        # Key was set but every call failed (provider overloaded / transient).
        # Do NOT claim 'requires_api_key' and do NOT fabricate any scores.
        summary = {
            "status": "error",
            "provider": config.LLM_PROVIDER,
            "model": config.LLM_MODEL,
            "n_judged": 0,
            "n_failed": failures,
            "sample_size": len(sample_items),
            "message": (
                f"All {failures} LLM calls failed after retries (provider "
                f"'{config.LLM_PROVIDER}' returned transient errors such as "
                f"429/503). No groundedness scores are reported (none fabricated). "
                f"Re-run when the provider is available: "
                f"`.venv/bin/python -m src.judge`."
            ),
        }
    else:
        key_hint = config.api_key_env_hint()
        summary = {
            "status": "requires_api_key",
            "provider": config.LLM_PROVIDER,
            "message": (
                f"Groundedness (LLM-as-judge) and answer generation require an "
                f"API key for provider '{config.LLM_PROVIDER}'. Set {key_hint} "
                f"(or LLM_API_KEY) and re-run: `.venv/bin/python -m src.judge`. "
                f"The default provider is Gemini's free tier "
                f"(`export GEMINI_API_KEY=...`). No groundedness numbers are "
                f"fabricated. The 20-row spot-check template was still written to "
                f"results/spotcheck_template.csv for manual review."
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
