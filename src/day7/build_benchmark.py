"""Build the Day 7 benchmark from the Day 6 questions plus 4 unanswerable cases.

    python -m src.day7.build_benchmark

Answerable questions keep the Day 6 split, gold page and verbatim evidence; Day 6
`alternative_gold` becomes extra items in `gold` (any one counts as a hit). Every
evidence string must lie inside one exported chunk of the right source and page.
"""
from __future__ import annotations

import json

from .config import CASES_JSON, CHUNKS_JSON, DAY6_QUESTIONS, QUESTIONS_JSON

# Near-topic questions whose answer is NOT in the report. Each absence was checked by
# reading the whole page(s) on the topic and searching the full corpus for the key terms.
UNANSWERABLE = [
    {"id": "u01", "split": "dev",
     "question": "What coupon rate did the catastrophe bond issued by the World Bank for Jamaica in 2024 pay to investors?",
     "absence_reason": "Page 20 only reports the full $150 million payout after Hurricane Melissa. 'coupon' "
                       "occurs only on page 19 (Ghana Clean Cooking Outcome Bond, no rate given)."},
    {"id": "u02", "split": "dev",
     "question": "What green hydrogen production capacity, in megawatts or tonnes per year, will the Hygenco Green Energies project in India have?",
     "absence_reason": "Page 14 gives IFC's $25 million, blended finance and 1,000 jobs but no capacity; "
                       "'MW'/'megawatt' occur only on page 10 (Ukraine wind), 'hydrogen' only on pages 14 and 36."},
    {"id": "u03", "split": "test",
     "question": "How many passengers did Cabo Verde's airports handle in 2025?",
     "absence_reason": "Page 13 states only a 60 percent surge in passenger volumes since 2022 and 35 new routes; "
                       "no absolute passenger count anywhere in the corpus."},
    {"id": "u04", "split": "test",
     "question": "What was the largest single damages award, in US dollars, issued by an ICSID tribunal in fiscal 2026?",
     "absence_reason": "Page 44 gives only percentage buckets (e.g. 3 percent of cases in the 100–499 million range); "
                       "no individual award amount is reported."},
]


# Day 7 label audit (done for EVERY question by searching the corpus for the answer's key
# figures, independent of any system output): passages that state the same fact on another
# page are added as alternative gold. Test needed no addition. Original Day 6 gold is kept
# first in `gold`; added items carry "origin": "day7_audit".
LABEL_AUDIT = {
    "d01": [(4, "Total private capital mobilized during the fiscal year reached $112 billion"),
            (5, "We mobilized $112 billion in private capital, more than triple the $35 billion we mobilized in fiscal 2022"),
            (32, "The World Bank Group mobilized $112 billion in private capital in fiscal 2026, more than tripling from $35 billion in fiscal 2022")],
    "d02": [(5, "We shortened average project preparation and approval times from 19 months to nine")],
    "d03": [(5, "Those countries are expected to create only about 420 million jobs")],
    "d14": [(19, "In fiscal 2026, IBRD, which is rated triple-A by the major credit rating agencies, raised nearly $50 billion through Sustainable Development Bonds")],
}


def convert(q: dict) -> dict:
    gold = [{"source_id": q["gold_source_id"], "page": q["gold_page"], "evidence": q["evidence_quote"]}]
    gold += [{"source_id": a["gold_source_id"], "page": a["gold_page"], "evidence": a["evidence_quote"]}
             for a in q.get("alternative_gold", [])]
    gold += [{"source_id": q["gold_source_id"], "page": page, "evidence": ev, "origin": "day7_audit"}
             for page, ev in LABEL_AUDIT.get(q["id"], [])]
    out = {"id": q["id"], "split": q["split"], "type": q["type"], "question": q["question"],
           "expected_answer": q["expected_answer"], "gold": gold,
           "gold_day6_count": 1 + len(q.get("alternative_gold", []))}
    if q.get("filter"):
        out["filter"], out["filter_source"] = q["filter"], q["filter_source"]
    return out


def main() -> None:
    chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    raw = [json.loads(line) for line in DAY6_QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = [convert(q) for q in raw]
    problems = []
    for q in questions:
        for g in q["gold"]:
            if not any(c["source_id"] == g["source_id"] and c["page"] == g["page"] and g["evidence"] in c["text"]
                       for c in chunks):
                problems.append(f"{q['id']}: evidence not inside one chunk on p{g['page']}: {g['evidence'][:60]}")
    for u in UNANSWERABLE:
        # Guard: none of the unanswerable cases may reuse an answerable ID.
        assert u["id"] not in {q["id"] for q in questions}
    cases = ([{"id": q["id"], "split": q["split"], "question": q["question"], "answerable": True,
               "expected_behavior": "search_and_answer", **({"filter": q["filter"]} if "filter" in q else {})}
              for q in questions]
             + [{**u, "answerable": False, "expected_behavior": "search_and_abstain"} for u in UNANSWERABLE])
    QUESTIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    QUESTIONS_JSON.write_text(json.dumps(questions, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    CASES_JSON.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    count = {s: {"answerable": sum(c["answerable"] and c["split"] == s for c in cases),
                 "unanswerable": sum(not c["answerable"] and c["split"] == s for c in cases)} for s in ("dev", "test")}
    print(json.dumps(count))
    if problems:
        print("WARNING (evidence split across chunks; such an alternative can never be hit):")
        print("\n".join("  " + p for p in problems))


if __name__ == "__main__":
    main()
