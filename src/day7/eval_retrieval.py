"""Retrieval/context benchmark on the ORIGINAL question text (no rewriting, no gold).

    python -m src.day7.eval_retrieval --split dev  --configs dense bm25 hybrid hybrid_filter hybrid_rerank
    python -m src.day7.eval_retrieval --split test --configs dense hybrid_filter

Writes results/retrieval/<split>/<config>/{traces.json,summary.json} and prints a table.
Hit@1 / Evidence Hit@3: a ranked chunk counts when it is on the gold source+page and
contains one gold evidence string. Context Evidence Rate: same test on the packed context.
Unanswerable cases are excluded from these denominators.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

from .config import QUESTIONS_JSON, RESULTS, RETRIEVAL_CONFIGS
from .retrieval import QueryEmbedder, build, is_hit, load_chunks, pack_context

REPORT_K = 3  # always report top-3 so configs with other top_k stay comparable


def run_config(name: str, cfg: dict, questions: list[dict], chunks, embedder) -> tuple[list, dict]:
    retriever = build(cfg["method"], chunks, candidates=cfg.get("candidates", 20), embedder=embedder)
    emb_calls_before = embedder.calls
    traces = []
    for q in questions:
        applied = q.get("filter") if cfg["use_filter"] else None
        started = time.perf_counter()
        results = retriever.search(q["question"], max(cfg["top_k"], REPORT_K), applied)
        elapsed = time.perf_counter() - started
        context, ids = pack_context(results[:cfg["top_k"]], cfg["budget"])
        gold_day6 = [g for g in q["gold"] if g.get("origin") != "day7_audit"]
        marked = [{"rank": r["rank"], "id": r["id"], "page": r["page"], "score": round(r["score"], 5),
                   "hit": is_hit(r, q["gold"]), "hit_day6_labels": is_hit(r, gold_day6),
                   **{k: r[k] for k in ("dense_rank", "bm25_rank", "hybrid_rank", "llm_grade") if k in r}}
                  for r in results]
        traces.append({
            "query_id": q["id"], "split": q["split"], "type": q["type"], "question": q["question"],
            "config": name, "applied_filter": applied, "results": marked,
            "context_ids": ids, "context_chars": len(context), "seconds": round(elapsed, 3),
            "hit1": bool(marked and marked[0]["hit"]),
            "hit3": any(r["hit"] for r in marked[:REPORT_K]),
            "context_hit": any(r["hit"] for r in marked if r["id"] in ids),
            "day6_labels": {"hit1": bool(marked and marked[0]["hit_day6_labels"]),
                            "hit3": any(r["hit_day6_labels"] for r in marked[:REPORT_K]),
                            "context_hit": any(r["hit_day6_labels"] for r in marked if r["id"] in ids)},
        })
    n = len(traces)
    usage = getattr(retriever, "usage", None)
    summary = {
        "config": name, **cfg, "split": questions[0]["split"], "N": n,
        "hit1": f"{sum(t['hit1'] for t in traces)}/{n}", "hit1_rate": round(sum(t['hit1'] for t in traces) / n, 4),
        "evidence_hit3": f"{sum(t['hit3'] for t in traces)}/{n}",
        "evidence_hit3_rate": round(sum(t['hit3'] for t in traces) / n, 4),
        "context_evidence": f"{sum(t['context_hit'] for t in traces)}/{n}",
        "context_evidence_rate": round(sum(t['context_hit'] for t in traces) / n, 4),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
        "median_seconds": round(statistics.median(t["seconds"] for t in traces), 3),
        "total_seconds": round(sum(t["seconds"] for t in traces), 2),
        "new_query_embeddings": embedder.calls - emb_calls_before,
        "llm_usage": usage.as_dict() if usage else None,
        "day6_labels": {k: f"{sum(t['day6_labels'][k] for t in traces)}/{n}" for k in ("hit1", "hit3", "context_hit")},
        "misses_hit3": [t["query_id"] for t in traces if not t["hit3"]],
        "misses_context": [t["query_id"] for t in traces if not t["context_hit"]],
    }
    return traces, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--configs", nargs="+", default=list(RETRIEVAL_CONFIGS))
    args = parser.parse_args()
    # Only the requested split is loaded, so dev work never touches test evidence.
    questions = [q for q in json.loads(QUESTIONS_JSON.read_text(encoding="utf-8")) if q["split"] == args.split]
    chunks, embedder = load_chunks(), QueryEmbedder()
    print(f"{'config':<15}{'Hit@1':>8}{'EvHit@3':>9}{'CtxEv':>8}{'ctx chars':>10}{'med s':>8}  day6-labels(H1/H3/Ctx)  misses@3")
    for name in args.configs:
        traces, summary = run_config(name, RETRIEVAL_CONFIGS[name], questions, chunks, embedder)
        out = RESULTS / "retrieval" / args.split / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{name:<15}{summary['hit1']:>8}{summary['evidence_hit3']:>9}{summary['context_evidence']:>8}"
              f"{summary['avg_context_chars']:>10}{summary['median_seconds']:>8}  "
              f"{'/'.join(summary['day6_labels'].values()):<23}{summary['misses_hit3']}")


if __name__ == "__main__":
    main()
