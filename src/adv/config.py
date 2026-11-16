"""Advanced challenge: World Development Report 2025 (408 PDF pages), multimodal index."""
from __future__ import annotations

import os
from pathlib import Path

from src.day7.config import DATABASE_URL, REPO  # noqa: F401 (loads .env)

ADV = REPO / "advanced"
SOURCE_PDF = ADV / "source" / "WDR-2025.pdf"
SOURCE = {
    "source_id": "wdr2025",
    "title": "World Development Report 2025: Standards for Development",
    "publisher": "World Bank",
    "handle": "https://hdl.handle.net/10986/43949",
    "download_url": "https://openknowledge.worldbank.org/server/api/core/bitstreams/24107611-5057-4d2c-a300-7e9e03cc979a/content",
    "license": "CC BY 3.0 IGO",
    "pages": 408,
}
WORK = ADV / "work"
PAGES_DIR = WORK / "pages"          # page renders (re-creatable, git-ignored)
CROPS_DIR = WORK / "crops"          # crops of visual elements (re-creatable, git-ignored)
VISUAL_DIR = WORK / "visual"        # one JSON per page: vision extraction (cached, committed)
TEXT_JSONL = WORK / "pages_text.jsonl"
CHUNKS_JSON = ADV / "index" / "chunks.json"
BENCH = ADV / "benchmark"
RESULTS = ADV / "results"

RENDER_DPI = 120
# Free tier = 500 generate requests/day PER MODEL, so the work is spread over models:
#   index vision: gemini-3.5-flash-lite for PDF pages 1-252, gemini-3.6-flash for part of 253-382; the rest hit the daily quota and are indexed from the text layer only (recorded per page)
#   agent and look_at_page (vision at query time): gemini-3.1-flash-lite (chosen by the student;
#   earlier attempts hit 503 "overloaded" errors, which the client retries)
VISION_MODEL = os.getenv("ADV_VISION_MODEL", "gemini-3.5-flash-lite")
AGENT_MODEL = os.getenv("ADV_AGENT_MODEL", "gemini-3.1-flash-lite")
LOOK_MODEL = os.getenv("ADV_LOOK_MODEL", "gemini-3.1-flash-lite")
TABLE = "adv_chunks"
# Local embedding model (ONNX via fastembed): no API quota/cost. The Gemini free tier allows only
# 1000 embedding requests/day, fewer than the ~1.5k chunks of this 408-page document.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
MODELS_DIR = WORK / "models"


_embedder = None


def embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(EMBEDDING_MODEL, cache_dir=str(MODELS_DIR))
    return _embedder
