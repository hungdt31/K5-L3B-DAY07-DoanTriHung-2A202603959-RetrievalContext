"""Step 2: build text + visual chunks for ALL pages, embed them, store in pgvector.

    python -m src.adv.index            # chunks.json + embeddings into table adv_chunks
    python -m src.adv.index --stats    # print index statistics only

Chunk IDs encode the page so a citation always locates the page:
  p153-t01  text chunk 1 of PDF page 153 (reading-order text layer, ≤1200 chars, paragraph-aligned)
  p153-v1   visual element 1 of PDF page 153 (vision transcription, with type/label/bbox/crop path)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from collections import Counter

import psycopg
from psycopg.types.json import Jsonb

from .config import (CHUNKS_JSON, CROPS_DIR, DATABASE_URL, EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, PAGES_DIR,
                     RENDER_DPI, SOURCE, SOURCE_PDF, TABLE, TEXT_JSONL, VISUAL_DIR, embedder)

MAX_CHARS = 1200
PRINTED_OFFSET = 32          # printed page = PDF page - 32 for the main body (PDF page >= 33)
_FOOTER = re.compile(r"^(World Development Report 2025|\d{1,3}|[ivxlc]{1,7})$", re.I)


def printed_page(page: int) -> str:
    return str(page - PRINTED_OFFSET) if page > PRINTED_OFFSET else "front matter"


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("­", "").replace("​", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text


def text_chunks(page: int, raw: str) -> list[dict]:
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", clean(raw))]
    paras = [p for p in paras if p and not _FOOTER.match(p)]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > MAX_CHARS:                      # split very long paragraphs at a sentence end
            cut = p.rfind(". ", 0, MAX_CHARS)
            cut = cut + 1 if cut > 200 else MAX_CHARS
            if cur:
                chunks.append(cur); cur = ""
            chunks.append(p[:cut].strip()); p = p[cut:].strip()
        if cur and len(cur) + 1 + len(p) > MAX_CHARS:
            chunks.append(cur); cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return [{"id": f"p{page:03d}-t{i:02d}", "page": page, "printed_page": printed_page(page), "kind": "text",
             "content": c, "metadata": {}} for i, c in enumerate(chunks, 1) if len(c) >= 40]


def norm_bbox(b) -> list[int] | None:
    while isinstance(b, list) and len(b) == 1 and isinstance(b[0], list):
        b = b[0]
    if isinstance(b, list) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b):
        x0, y0, x1, y1 = (max(0, min(1000, int(v))) for v in b)
        if x1 > x0 and y1 > y0:
            return [x0, y0, x1, y1]
    return None


def crop(page: int, idx: int, bbox: list[int] | None) -> str | None:
    """Save a crop of the element from the page render (evidence for graders)."""
    src = PAGES_DIR / f"p{page:03d}.png"
    if bbox is None or not src.exists():
        return None
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    out = CROPS_DIR / f"p{page:03d}-v{idx}.png"
    if not out.exists():
        dims = subprocess.run(["magick", "identify", "-format", "%w %h", str(src)], capture_output=True, text=True)
        w, h = (int(x) for x in dims.stdout.split())
        x0, y0, x1, y1 = bbox
        geom = f"{(x1 - x0) * w // 1000}x{(y1 - y0) * h // 1000}+{x0 * w // 1000}+{y0 * h // 1000}"
        subprocess.run(["magick", str(src), "-crop", geom, "+repage", str(out)], check=True)
    return str(out.relative_to(CROPS_DIR.parents[2]))


def visual_chunks(page: int) -> tuple[list[dict], str | None]:
    path = VISUAL_DIR / f"p{page:03d}.json"
    if not path.exists():
        return [], "not processed"
    data = json.loads(path.read_text(encoding="utf-8"))
    if "error" in data:
        return [], data["error"]
    out = []
    for i, e in enumerate(data.get("elements", []), 1):
        content = str(e.get("content", "")).strip()
        if not content:
            continue
        bbox = norm_bbox(e.get("bbox"))
        label, title, kind = str(e.get("label") or ""), str(e.get("title") or ""), str(e.get("type") or "visual")
        out.append({"id": f"p{page:03d}-v{i}", "page": page, "printed_page": printed_page(page), "kind": kind,
                    "content": f"[{kind}] {label}: {title}\n{content}".strip(),
                    "metadata": {"label": label, "title": title, "bbox_0_1000": bbox, "crop": crop(page, i, bbox),
                                 "extracted_by": data.get("model"), "render_dpi": data.get("dpi", RENDER_DPI)}})
    return out, None


def build() -> tuple[list[dict], dict]:
    pages = [json.loads(l) for l in TEXT_JSONL.open(encoding="utf-8")]
    chunks, issues = [], {}
    for p in pages:
        t = text_chunks(p["page"], p["text"])
        v, err = visual_chunks(p["page"])
        if err:
            issues[p["page"]] = f"vision: {err}"
        if not t and not v:
            issues.setdefault(p["page"], "no extractable text and no visual element (blank/decorative page)")
        chunks += t + v
    return chunks, issues


def embed_all(chunks: list[dict]) -> list[list[float]]:
    """Local bge-small-en-v1.5 passage embeddings (normalized, 384 dims)."""
    vectors = [list(map(float, v)) for v in embedder().embed([c["content"][:4000] for c in chunks], batch_size=16)]
    assert all(len(v) == EMBEDDING_DIMENSIONS for v in vectors)
    print(f"embedded {len(vectors)} chunks with {EMBEDDING_MODEL}")
    return vectors


def store(chunks: list[dict], vectors: list[list[float]]) -> None:
    with psycopg.connect(DATABASE_URL) as conn, conn.transaction():
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"DROP TABLE IF EXISTS {TABLE}")  # full rebuild: the index is derived data
        conn.execute(f"""CREATE TABLE {TABLE} (
            chunk_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, page INT NOT NULL, printed_page TEXT,
            kind TEXT NOT NULL, content TEXT NOT NULL, metadata JSONB NOT NULL DEFAULT '{{}}',
            embedding_model TEXT NOT NULL, embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL)""")
        conn.execute(f"DELETE FROM {TABLE} WHERE source_id = %s", (SOURCE["source_id"],))
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {TABLE} VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)",
                [(c["id"], SOURCE["source_id"], c["page"], c["printed_page"], c["kind"], c["content"],
                  Jsonb(c["metadata"]), EMBEDDING_MODEL, "[" + ",".join(map(repr, v)) + "]")
                 for c, v in zip(chunks, vectors)])


def stats(chunks: list[dict], issues: dict) -> dict:
    kinds = Counter(c["kind"] for c in chunks)
    visual_pages = sorted({c["page"] for c in chunks if c["kind"] != "text"})
    return {"source": SOURCE, "pdf_sha256": subprocess.run(["sha256sum", str(SOURCE_PDF)], capture_output=True,
                                                           text=True).stdout.split()[0],
            "chunks": len(chunks), "by_kind": dict(kinds),
            "pages_with_text_chunks": len({c["page"] for c in chunks if c["kind"] == "text"}),
            "pages_with_visual_elements": len(visual_pages),
            "visual_labels": sorted({c["metadata"].get("label") for c in chunks if c["kind"] != "text"} - {""}),
            "pages_not_indexed_or_partial": issues}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()
    chunks, issues = build()
    info = stats(chunks, issues)
    CHUNKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    (CHUNKS_JSON.parent / "index_stats.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k not in ("visual_labels", "source")}, ensure_ascii=False)[:1500])
    if args.stats:
        return
    CHUNKS_JSON.write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    store(chunks, embed_all(chunks))
    print(f"stored {len(chunks)} chunks in {TABLE}")


if __name__ == "__main__":
    main()
