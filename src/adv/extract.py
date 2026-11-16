"""Step 1: per-page text layer + page renders + vision extraction of visual elements.

    python -m src.adv.extract text            # pdftotext per page -> work/pages_text.jsonl
    python -m src.adv.extract vision [--pages 1-408]   # render + Gemini vision, cached per page

Vision sees the rendered page AND its text layer, and returns every table/chart/diagram/map
/photo on the page with a bounding box (0-1000 page coordinates) and a faithful
transcription. Pages with no visual element return an empty list. Results are cached in
work/visual/pNNN.json, so re-runs only process missing pages. Errors are stored per page.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import time

from google.genai import types

from src.day7 import llm
from .config import PAGES_DIR, RENDER_DPI, SOURCE, SOURCE_PDF, TEXT_JSONL, VISION_MODEL, VISUAL_DIR

log = logging.getLogger("adv.extract")
_PRINTED = re.compile(r"(?:^|\n)\s*(\d{1,3}|[ivxl]{1,6})\s*(?:\n|$)")

VISION_PROMPT = """You index one page (PDF page {page}) of the World Bank "World Development Report 2025:
Standards for Development". You get the page image and its extracted text layer (may be jumbled).

Find every VISUAL element on the page: table, chart (bar/line/scatter/etc.), diagram/flowchart/
decision tree, map, infographic, or photo. Ignore running headers, page numbers, logos and
purely decorative graphics. Running body text is NOT a visual element.

For each element return:
- "type": one of table|chart|diagram|map|infographic|photo
- "label": printed label such as "Figure 3.1", "Table O.1", "Map 6.2", "Figure B8.2.1" (or "" if none)
- "title": printed title/caption
- "bbox": [x0, y0, x1, y1] integers 0-1000 relative to page width/height, covering the element incl. title
- "content": a faithful, complete transcription that lets someone answer questions WITHOUT the image:
   * table: every row as "col1 | col2 | ..." with the header row first; keep units and footnote marks.
   * chart: chart type, each panel (a, b, ...) with its title, axis titles, units, legend/series names,
     and every printed data label/value; if values are not printed, give readable approximate values
     per category ("≈") and describe trends/rankings. Name the highest and lowest categories.
   * diagram: every box/node text and every arrow as "A -> B: arrow label"; describe the flow/decision logic.
   * map: legend categories (with colors) and which countries/regions are visibly in each category
     when identifiable; state what the map measures and its year.
   * photo/infographic: what it shows and any text in it.
   Copy numbers exactly as printed. Do not invent values. Include the "Source:"/"Note:" lines.

Return JSON only: {{"elements": [ ... ]}}  (empty list if the page has no visual element).

Text layer of the page:
<<<
{text}
>>>"""


def page_text(page: int, layout: bool) -> str:
    args = ["pdftotext", "-f", str(page), "-l", str(page)] + (["-layout"] if layout else []) + [str(SOURCE_PDF), "-"]
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def cmd_text(_args) -> None:
    TEXT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with TEXT_JSONL.open("w", encoding="utf-8") as f:
        for p in range(1, SOURCE["pages"] + 1):
            raw = page_text(p, layout=False)
            f.write(json.dumps({"page": p, "text": raw, "text_layout": page_text(p, layout=True)},
                               ensure_ascii=False) + "\n")
    print(f"wrote {TEXT_JSONL}")


def render(page: int) -> bytes:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    out = PAGES_DIR / f"p{page:03d}.png"
    if not out.exists():
        subprocess.run(["pdftoppm", "-r", str(RENDER_DPI), "-png", "-singlefile", "-f", str(page), "-l", str(page),
                        str(SOURCE_PDF), str(out.with_suffix(""))], check=True)
    return out.read_bytes()


def vision_page(page: int, text: str, usage: llm.Usage) -> dict:
    image = render(page)
    cfg = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
    started = time.perf_counter()
    resp = llm.generate(VISION_MODEL, [types.Part.from_bytes(data=image, mime_type="image/png"),
                                       VISION_PROMPT.format(page=page, text=text[:12000])], cfg, usage)
    data = json.loads(resp.text or "{}")
    elements = data.get("elements", []) if isinstance(data, dict) else []
    meta = resp.usage_metadata
    return {"page": page, "model": VISION_MODEL, "dpi": RENDER_DPI, "elements": elements,
            "seconds": round(time.perf_counter() - started, 2),
            "prompt_tokens": meta.prompt_token_count if meta else None,
            "output_tokens": meta.candidates_token_count if meta else None}


def parse_range(spec: str) -> list[int]:
    pages = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        pages += list(range(int(a), int(b or a) + 1))
    return pages


def cmd_vision(args) -> None:
    texts = {json.loads(l)["page"]: json.loads(l)["text_layout"] for l in TEXT_JSONL.open(encoding="utf-8")}
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    usage = llm.Usage()
    for page in parse_range(args.pages):
        out = VISUAL_DIR / f"p{page:03d}.json"
        if out.exists() and "error" not in json.loads(out.read_text()):
            continue
        try:
            result = vision_page(page, texts[page], usage)
        except Exception as exc:  # keep going; the page is retried on the next run
            result = {"page": page, "error": repr(exc)[:300]}
            log.error("p%d: %s", page, exc)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        log.info("p%d: %d elements %s", page, len(result.get("elements", [])), result.get("seconds"))
        time.sleep(0.5)
    print(json.dumps(usage.as_dict()))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("text").set_defaults(func=cmd_text)
    v = sub.add_parser("vision")
    v.add_argument("--pages", default=f"1-{SOURCE['pages']}")
    v.set_defaults(func=cmd_vision)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
