"""Check that verbatim evidence quotes occur on their PDF page (text layer of the original PDF).
Evidence for charts/diagrams/maps is a human reading of the page image and is checked by eye instead."""
import json, re, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
norm = lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).replace("­", "").replace("​", "")).strip().lower()
pages = {json.loads(l)["page"]: norm(json.loads(l)["text"]) for l in (ROOT / "advanced/work/pages_text.jsonl").open()}
qs = json.loads((ROOT / "advanced/benchmark/questions.json").read_text())
ok = bad = skipped = 0
for q in qs:
    for e in q.get("evidence", []):
        if "text" not in e["element"] and not e["element"].endswith(" note"):
            skipped += 1; continue
        if all(norm(part) in pages[e["page"]] for part in e["evidence"].split(" ... ")):
            ok += 1
        else:
            bad += 1; print("NOT FOUND", q["id"], e["page"], e["evidence"][:90])
print(f"verbatim text evidence found: {ok}, missing: {bad}, visual/table evidence checked by eye: {skipped}")
