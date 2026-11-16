"""Collect retrieval + agent results into results/summary.md (no API calls).

    python -m src.day7.summarize
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import CHUNKS_JSON, QUESTIONS_JSON, RESULTS
from .retrieval import is_hit


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else "0/0 (không áp dụng)"


def retrieval_rows(split: str) -> list[str]:
    rows = []
    for d in sorted((RESULTS / "retrieval" / split).glob("*/summary.json")):
        s = load(d)
        n = s["N"]
        llm = s.get("llm_usage") or {}
        cost = f"{llm.get('calls', 0)} lần gọi LLM, {llm.get('prompt_tokens', 0) + llm.get('output_tokens', 0)} token" \
            if llm else "0 lần gọi LLM"
        k = f"top-{s['top_k']}" + (f", pool {s['candidates']}" if "candidates" in s else "")
        rows.append(f"| {s['config']} | {s['method']}, {k}, filter={'có' if s['use_filter'] else 'không'}, "
                    f"budget {s['budget']} ký tự | {split} | {pct(int(s['hit1'].split('/')[0]), n)} | "
                    f"{pct(int(s['evidence_hit3'].split('/')[0]), n)} | {pct(int(s['context_evidence'].split('/')[0]), n)} | "
                    f"{s['day6_labels']['hit1']} / {s['day6_labels']['hit3']} | {s['avg_context_chars']:.0f} | "
                    f"{s['median_seconds']} s | {cost} |")
    return rows


def agent_rows(split: str, questions: dict, chunks: dict) -> list[str]:
    rows = []
    for d in sorted((RESULTS / "agent" / split).glob("*/metrics.json")):
        m, s = load(d), load(d.parent / "summary.json")
        traces = load(d.parent / "traces.json")
        ans = [t for t in traces if t["answerable"] and t["status"] != "error"]
        ctx = sum(any(is_hit(chunks[i], questions[t["query_id"]]["gold"]) for i in t["context_ids"]) for t in ans)
        g = m["grounded_answer"]
        rows.append(
            f"| {s['config']} / {split} | {pct(m['tool_use']['n'], m['tool_use']['N'])} | "
            f"{s['searches_distribution']} (TB {s['mean_searches']}) | {g['0']}/{g['1']}/{g['2']}, TB {g['mean']:.2f}/2 | "
            f"{pct(m['citation_validity']['n'], m['citation_validity']['N'])} | "
            f"{pct(m['answer_citation_coverage']['n'], m['answer_citation_coverage']['N'])} | "
            f"{pct(m['abstention']['n'], m['abstention']['N'])} | {pct(m['false_abstention']['n'], m['false_abstention']['N'])} | "
            f"{pct(ctx, len(ans))} | {s['model_calls']} lần gọi, {s['prompt_tokens']}+{s['output_tokens']} token | "
            f"trung vị {s['median_seconds']} s, tổng {s['total_seconds']} s | {s['errors']} (chạy lại: {len(s.get('rerun_after_error', []))}) |")
    return rows


def main() -> None:
    questions = {q["id"]: q for q in load(QUESTIONS_JSON)}
    chunks = {c["id"]: c for c in load(CHUNKS_JSON)}
    out = ["# Kết quả Day 7 (tự sinh bởi `python -m src.day7.summarize`)", "",
           "## Retrieval với query gốc", "",
           "| Cấu hình | Phương pháp | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Hit@1 / Hit@3 theo nhãn Day 6 gốc | "
           "Ký tự context TB | Thời gian trung vị | Chi phí thêm |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    out += retrieval_rows("dev") + retrieval_rows("test")
    out += ["", "Query embedding: `gemini-embedding-001` 768 chiều, 1 lần gọi/câu hỏi mới (cache theo nội dung query, "
            "lần gọi đầu ~0.4–0.5 s). Thời gian trên đo khi cache đã có, trừ khi ghi khác.", "",
            "## Agent (model thật, nhãn đọc tay)", "",
            "| Cấu hình / split | Tool use | Số lượt search | Grounded 0/1/2 | Citation hợp lệ | Câu trả lời có citation | "
            "Abstention đúng | Từ chối nhầm | Context Evidence Rate | Token | Thời gian | Lỗi |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out += agent_rows("dev", questions, chunks) + agent_rows("test", questions, chunks)
    out += ["", "## Từng câu test — cấu hình đã chọn (`agent_dense`)", "",
            "| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |", "|---|---|---|---|---|"]
    sel = RESULTS / "agent" / "test" / "agent_dense"
    reviews = {r["query_id"]: r for r in load(sel / "reviews.json")}
    for t in load(sel / "traces.json"):
        r = reviews[t["query_id"]]
        calls = "; ".join(f"search(\"{e['query']}\") → " + ", ".join(f"p{x['page']}" for x in e["results"])
                          for e in t["events"] if e["event"] == "search")
        label = f"grade {r['answer_grade']}" if t["answerable"] else f"abstention_correct={r['abstention_correct']}"
        answer = t["answer"].replace("|", "/")
        out.append(f"| {t['query_id']} | {'có đáp án' if t['answerable'] else 'thiếu evidence'} | {calls} | "
                   f"{t['status']}: {answer} | {label}: {r['reason']} |")
    (RESULTS / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
