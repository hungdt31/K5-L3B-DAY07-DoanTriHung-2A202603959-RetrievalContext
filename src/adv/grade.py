"""Review sheet + pass-rate aggregation for the advanced benchmark (labels are written by a human).

    python -m src.adv.grade sheet --split dev --config mm_hybrid_look     # print what to read for each case
    python -m src.adv.grade score --split test --config mm_hybrid_look    # aggregate labels.json -> metrics.json

labels.json (next to traces.json): [{"query_id": ..., "pass": true/false, "reason": "..."}]
Pass rule (ADVANCED.md): answerable -> correct AND complete (units/conditions/calculation), citations locate the
evidence page/region and support the whole answer; unanswerable -> clear refusal without unsupported claims.
Errors/timeouts count as fail and stay in the denominator.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from .config import BENCH, CHUNKS_JSON, RESULTS

GROUPS = ["table", "image", "chart", "multihop", "text", "unanswerable"]


def load(split: str, config: str):
    qs = {q["id"]: q for q in json.loads((BENCH / "questions.json").read_text()) if q["split"] == split}
    traces = {t["query_id"]: t for t in json.loads((RESULTS / split / config / "traces.json").read_text())}
    return qs, traces


def sheet(args) -> None:
    qs, traces = load(args.split, args.config)
    chunks = {c["id"]: c for c in json.loads(CHUNKS_JSON.read_text())}
    for qid, q in qs.items():
        t = traces[qid]
        gold_pages = sorted({e["page"] for e in q.get("evidence", [])})
        print("=" * 100)
        print(f"{qid} [{q['group']}] {q['question']}")
        print(f"  EXPECTED: {q.get('expected_answer') or q.get('expected_behavior')}  | gold pages {gold_pages}")
        print(f"  STATUS: {t['status']}  tools={t['tool_counts']}  cited pages={t['cited_pages']}  {t['seconds']}s")
        print(f"  ANSWER: {t['answer']}")
        for c in t["citations"]:
            ch = chunks.get(c)
            if ch:
                print(f"    [{c}] {ch['kind']} {ch['metadata'].get('label', '')} :: {ch['content'][:args.chars]!r}")
        for e in t["events"]:
            if e["event"] == "look_at_page":
                print(f"    look_at_page(p{e['page']}, {e['question'][:80]!r}) -> {str(e.get('observation'))[:args.chars]!r}")


def score(args) -> None:
    qs, traces = load(args.split, args.config)
    out_dir = RESULTS / args.split / args.config
    labels = {l["query_id"]: l for l in json.loads((out_dir / "labels.json").read_text())}
    missing = set(qs) - set(labels)
    if missing:
        raise SystemExit(f"missing labels for {sorted(missing)}")
    per = defaultdict(lambda: [0, 0])
    for qid, q in qs.items():
        ok = bool(labels[qid]["pass"]) and traces[qid]["status"] != "error"
        per[q["group"]][0] += ok
        per[q["group"]][1] += 1
    n, N = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    tr = list(traces.values())
    metrics = {
        "split": args.split, "config": args.config, "pass": f"{n}/{N}", "pass_rate": round(n / N, 4),
        "meets_90_percent": n / N >= 0.9,
        "by_group": {g: f"{per[g][0]}/{per[g][1]}" for g in GROUPS if per[g][1]},
        "errors": sum(t["status"] == "error" for t in tr),
        "status": {s: sum(t["status"] == s for t in tr) for s in ("answered", "abstain", "error")},
        "agent_model": tr[0]["model"],
        "agent_tokens": {"prompt": sum(t["usage"]["prompt_tokens"] for t in tr),
                         "output": sum(t["usage"]["output_tokens"] + t["usage"]["thought_tokens"] for t in tr)},
        "look_tokens": {"prompt": sum(t["vision_usage"]["prompt_tokens"] for t in tr),
                        "output": sum(t["vision_usage"]["output_tokens"] + t["vision_usage"]["thought_tokens"] for t in tr)},
        "model_calls": sum(t["usage"]["calls"] + t["vision_usage"]["calls"] for t in tr),
        "seconds": {"median": sorted(t["seconds"] for t in tr)[len(tr) // 2], "total": round(sum(t["seconds"] for t in tr), 1)},
        "failed": [{"id": q, "group": qs[q]["group"], "reason": labels[q]["reason"]} for q in qs if not labels[q]["pass"]],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("sheet", sheet), ("score", score)):
        s = sub.add_parser(name)
        s.add_argument("--split", required=True, choices=("dev", "test"))
        s.add_argument("--config", required=True)
        s.add_argument("--chars", type=int, default=260)
        s.set_defaults(func=fn)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
