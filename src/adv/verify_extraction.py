"""Check vision transcriptions against the original PDF text layer.

    python -m src.adv.verify_extraction

For every visual element, every number written in the transcription (e.g. 3,194 / 69.3 / 2024) is
looked up among the numbers printed on that PDF page's text layer. A number absent from the page is
either an approximate chart reading ("≈", "about") or a possible hallucination. The report lists,
per element type, the share of numbers confirmed by the original page, and the worst elements so
they can be inspected by eye (manual checks are recorded in advanced/index/extraction_manual_check.json).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from .config import CHUNKS_JSON, CHUNKS_JSON as _C, TEXT_JSONL

NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w])")


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUM.findall(text)}


def main() -> None:
    page_nums = {json.loads(l)["page"]: numbers(json.loads(l)["text"]) for l in TEXT_JSONL.open(encoding="utf-8")}
    chunks = [c for c in json.loads(CHUNKS_JSON.read_text()) if c["kind"] != "text"]
    per_kind = defaultdict(lambda: [0, 0, 0])   # elements, numbers, confirmed
    rows = []
    for c in chunks:
        body = c["content"]
        # numbers explicitly marked as approximate readings are not expected on the page
        exact = numbers(re.sub(r"(≈|~|about|approximately|around|roughly)\s*\S+", " ", body, flags=re.I))
        exact -= {str(c["page"]), str(c["page"] - 32)}
        found = {n for n in exact if n in page_nums[c["page"]]}
        k = per_kind[c["kind"]]
        k[0] += 1; k[1] += len(exact); k[2] += len(found)
        if exact:
            rows.append((len(found) / len(exact), c["id"], c["metadata"].get("label"), len(exact), sorted(exact - found)[:8]))
    report = {"elements_by_kind": {k: v[0] for k, v in per_kind.items()},
              "numbers_confirmed_by_page_text": {k: f"{v[2]}/{v[1]} ({v[2] / v[1]:.0%})" for k, v in per_kind.items() if v[1]},
              "lowest_confirmation": [{"id": i, "label": l, "share": round(s, 2), "numbers": n, "unconfirmed": u}
                                      for s, i, l, n, u in sorted(rows)[:15]]}
    out = CHUNKS_JSON.parent / "extraction_check.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
