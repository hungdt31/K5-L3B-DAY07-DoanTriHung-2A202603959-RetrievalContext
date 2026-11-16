"""Ask the agent one ad-hoc question (not part of the benchmark).

    python -m src.day7.ask "How much did IFC invest in Hygenco?"
    python -m src.day7.ask "What record guarantees were issued?" --config agent_hybrid_filter_k5 -f institution=miga
    python -m src.day7.ask "..." --trace          # also print every tool call
"""
from __future__ import annotations

import argparse
import json

from .agent import Agent
from .config import AGENT_CONFIGS
from .retrieval import QueryEmbedder, load_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--config", default="agent_dense", choices=list(AGENT_CONFIGS))
    parser.add_argument("-f", "--filter", action="append", help="key=value, e.g. institution=miga "
                        "(only used by configs with use_filter)")
    parser.add_argument("--trace", action="store_true", help="print tool calls and retrieved chunks")
    args = parser.parse_args()
    filters = dict(item.split("=", 1) for item in args.filter or []) or None
    case = {"id": "adhoc", "split": "adhoc", "question": args.question, "answerable": None}
    if filters:
        case["filter"] = filters
    result = Agent(AGENT_CONFIGS[args.config], load_chunks(), QueryEmbedder()).run(case)

    if args.trace:
        for e in result["events"]:
            if e["event"] == "search":
                print(f"search({e['query']!r}) -> " + ", ".join(f"{r['id']} ({r['score']:.3f})" for r in e["results"]))
            else:
                print(e["event"], {k: v for k, v in e.items() if k != "event"} or "")
        print()
    print(f"[{result['status']}] {result['answer']}")
    for cid in result["citations"]:
        print("  -", cid)
    u = result["usage"]
    print(f"\n{result['n_searches']} search, {u['calls']} model calls, "
          f"{u['prompt_tokens']}+{u['output_tokens']} tokens, {result['seconds']} s")
    if result["status"] == "error":
        print(json.dumps(result["events"][-1], ensure_ascii=False))


if __name__ == "__main__":
    main()
