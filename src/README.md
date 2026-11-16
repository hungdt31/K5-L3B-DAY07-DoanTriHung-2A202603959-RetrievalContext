# Code Day 7 — Agentic RAG trên index Day 6

| File | Việc |
|---|---|
| `day7/config.py` | Đường dẫn, model, các cấu hình retrieval và agent được so sánh |
| `day7/export_day6.py` | Xuất 245 chunk (strategy B) từ pgvector Day 6 → `data/day6/chunks.json` |
| `day7/build_benchmark.py` | Chuyển 24 câu Day 6 + 4 câu thiếu evidence → `data/benchmark/` và kiểm tra evidence |
| `day7/retrieval.py` | dense (pgvector), BM25, hybrid RRF, hybrid + LLM rerank; pack context; định nghĩa hit |
| `day7/eval_retrieval.py` | Benchmark retrieval/context trên query gốc |
| `day7/agent.py` | Agent Gemini function calling: tool `search_report`, điều kiện dừng, citation, trace |
| `day7/ask.py` | Hỏi agent một câu tùy ý: `python -m src.day7.ask "câu hỏi" [--trace]` |
| `day7/summarize.py` | Tổng hợp mọi kết quả → `results/summary.md` |
| `day7/llm.py` | Gọi Gemini có retry 429/5xx, đếm token/thời gian |

## Chạy lại từ đầu

Cần Day 6 đã ingest ([repo Day 6](https://github.com/hungdt31/K5-L3B-DAY06-DoanTriHung-2A202603959-DataIngestion)) và Docker.

```bash
# 1. Bật pgvector Day 6 (dữ liệu nằm trong volume Docker của Day 6)
cd ../K5-L3B-DAY06-DoanTriHung-2A202603959-DataIngestion/starter
docker compose -p day6_student_starter up -d --wait
cd -

# 2. Môi trường Day 7
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # điền GEMINI_API_KEY (cùng key Day 6)

# 3. Dữ liệu + benchmark (đã commit sẵn; chạy lại để kiểm tra)
.venv/bin/python -m src.day7.export_day6
.venv/bin/python -m src.day7.build_benchmark

# 4. Dev: so sánh retrieval rồi agent
.venv/bin/python -m src.day7.eval_retrieval --split dev --configs dense bm25 hybrid hybrid_filter hybrid_rerank
.venv/bin/python -m src.day7.agent --split dev --configs agent_dense_1search agent_dense agent_hybrid_filter_k5
#    (lỗi mạng: thêm --retry-errors để chạy lại riêng câu lỗi, lỗi cũ vẫn giữ trong trace)

# 5. Chấm tay: results/agent/<split>/<config>/reviews.json, rồi tổng hợp
.venv/bin/python starter/agent_metrics.py --cases data/benchmark/cases.json \
  --traces results/agent/dev/agent_dense/traces.json --reviews results/agent/dev/agent_dense/reviews.json \
  --split dev --output results/agent/dev/agent_dense/metrics.json

# 6. Chốt cấu hình (results/selection.json), rồi mới chạy test
.venv/bin/python -m src.day7.eval_retrieval --split test --configs dense hybrid_filter
.venv/bin/python -m src.day7.agent --split test --configs agent_dense_1search agent_dense

# 7. Bảng tổng hợp + test code
.venv/bin/python -m src.day7.summarize
.venv/bin/python -m unittest discover -s tests -v
```

Kết quả LLM có thể khác nhẹ giữa các lần chạy dù `temperature=0`.
