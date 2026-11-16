"""Paths, model names and named experiment configurations for Day 7."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")

DAY6_REPO_URL = "https://github.com/hungdt31/K5-L3B-DAY06-DoanTriHung-2A202603959-DataIngestion"
DAY6_DIR = Path(os.getenv("DAY6_DIR", REPO.parent / "K5-L3B-DAY06-DoanTriHung-2A202603959-DataIngestion"))
DAY6_QUESTIONS = DAY6_DIR / "benchmark" / "questions.jsonl"

DATA = REPO / "data"
CHUNKS_JSON = DATA / "day6" / "chunks.json"            # export of the Day 6 index (strategy B rows)
QUESTIONS_JSON = DATA / "benchmark" / "questions.json"  # answerable questions, Day 7 schema
CASES_JSON = DATA / "benchmark" / "cases.json"          # answerable + unanswerable cases
RESULTS = REPO / "results"

DATABASE_URL = os.getenv("DATABASE_URL", "")
# Day 6 strategy B = paragraph chunks (max 1000 chars). Day 6 strategy C = same rows + filter,
# so Day 7 searches the B rows and applies filters itself.
DAY6_STRATEGY = "B"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "gemini-embedding-001"
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS") or 768)
AGENT_MODEL = os.getenv("AGENT_MODEL") or "gemini-3.5-flash-lite"
FILTERABLE_FIELDS = {"institution", "section"}

# Retrieval-only experiments, compared on the same corpus, index, questions and labels.
# `budget` = characters of final context (chunk header included). `candidates` = pool size
# fed to fusion / rerank before cutting to top_k.
RETRIEVAL_CONFIGS: dict[str, dict] = {
    "dense":         {"method": "dense",  "top_k": 3, "budget": 3000, "use_filter": False},
    "bm25":          {"method": "bm25",   "top_k": 3, "budget": 3000, "use_filter": False},
    "hybrid":        {"method": "hybrid", "top_k": 3, "budget": 3000, "use_filter": False, "candidates": 20},
    # Improvement 1: user-provided metadata filter (from question/user context, never from gold).
    "hybrid_filter": {"method": "hybrid", "top_k": 3, "budget": 3000, "use_filter": True, "candidates": 20},
    # Improvement 2: LLM rerank of the hybrid candidate pool.
    "hybrid_rerank": {"method": "hybrid_rerank", "top_k": 3, "budget": 3000, "use_filter": True, "candidates": 12},
}

# Agent experiments: same model, prompt and stop rules; only retrieval tool / limits differ.
AGENT_CONFIGS: dict[str, dict] = {
    # Baseline: dense tool, a single search (classic retrieve-then-answer behaviour).
    "agent_dense_1search": {"method": "dense", "top_k": 3, "budget": 3000, "use_filter": False, "max_searches": 1},
    # Improvement A: allow up to 3 searches so the model can rewrite / narrow the query.
    "agent_dense": {"method": "dense", "top_k": 3, "budget": 3000, "use_filter": False, "max_searches": 3},
    # Improvement B: hybrid (dense+BM25, RRF) + user-provided filter, wider context (top-5, 5000 chars).
    "agent_hybrid_filter_k5": {"method": "hybrid", "top_k": 5, "budget": 5000, "use_filter": True,
                               "candidates": 20, "max_searches": 3},
}
