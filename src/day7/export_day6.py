"""Export the Day 6 pgvector rows (strategy B) to the Day 7 chunk JSON contract.

    python -m src.day7.export_day6

The JSON copy feeds the lexical retriever and the evidence checks; dense search still
queries pgvector directly. Vectors are not exported.
"""
from __future__ import annotations

import json

import psycopg

from .config import CHUNKS_JSON, DATABASE_URL, DAY6_STRATEGY, EMBEDDING_MODEL


def main() -> None:
    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL missing in .env")
    with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
        rows = conn.execute(
            "SELECT chunk_id, document_id, source, page, content, metadata, embedding_model "
            "FROM chunks WHERE strategy = %s ORDER BY chunk_id", (DAY6_STRATEGY,)).fetchall()
    if not rows:
        raise SystemExit(f"no rows for strategy {DAY6_STRATEGY}; run Day 6 ingest first")
    models = {r[6] for r in rows}
    if models != {EMBEDDING_MODEL}:
        raise SystemExit(f"index built with {models}, config expects {EMBEDDING_MODEL}")
    chunks = [{"id": cid, "source_id": doc, "source": src, "page": page, "text": text, "metadata": meta}
              for cid, doc, src, page, text, meta, _ in rows]
    CHUNKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_JSON.write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"exported {len(chunks)} chunks from {len({c['page'] for c in chunks})} pages -> {CHUNKS_JSON}")


if __name__ == "__main__":
    main()
