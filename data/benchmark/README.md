# Benchmark Day 7 — World Bank Group Annual Report 2026

Tạo bằng `python -m src.day7.build_benchmark` từ `benchmark/questions.jsonl` của
[repo Day 6](https://github.com/hungdt31/K5-L3B-DAY06-DoanTriHung-2A202603959-DataIngestion).

| Loại câu | Dev | Test |
|---|---:|---:|
| Có đáp án (`questions.json`) | 18 | 6 |
| Thiếu bằng chứng (`cases.json`, `answerable: false`) | 2 (u01, u02) | 2 (u03, u04) |
| **Tổng** | **20** | **8** |

- `questions.json`: `id`, `split`, `type`, `question`, `expected_answer`, `gold` (danh sách
  `{source_id, page, evidence}`; mỗi phần tử là một evidence thay thế hợp lệ), tùy chọn
  `filter` + `filter_source`. Evidence là trích nguyên văn, nằm trọn trong một chunk của
  index Day 6 (script kiểm tra tự động).
- `cases.json`: toàn bộ 28 case cho agent; câu thiếu evidence có `absence_reason`.
- Split giữ nguyên từ Day 6 (chia theo trang gold, chốt trước mọi lần chạy). Câu thiếu
  evidence của dev bám chủ đề trang dev (p14, p20), của test bám trang test (p13, p44).

## Kiểm tra câu thiếu evidence

Đọc toàn bộ trang cùng chủ đề và tìm từ khóa trên toàn corpus (245 chunk):

| ID | Câu hỏi (rút gọn) | Vì sao không có |
|---|---|---|
| u01 | Lãi coupon của cat bond Jamaica | p20 chỉ nêu khoản chi trả $150 triệu; "coupon" chỉ xuất hiện ở p19 (trái phiếu Ghana, không có lãi suất) |
| u02 | Công suất hydro xanh của Hygenco | p14 không có công suất; "MW" chỉ có ở p10 (điện gió Ukraine) |
| u03 | Số hành khách sân bay Cabo Verde 2025 | p13 chỉ nêu tăng 60% từ 2022 |
| u04 | Khoản bồi thường ICSID lớn nhất | p44 chỉ có tỷ lệ theo khoảng giá trị |

## Tiêu chí chọn (chốt trước khi chạy dev)

**Retrieval** (query gốc, chỉ dùng dev):
1. Evidence Hit@3 cao hơn;
2. hòa → Context Evidence Rate cao hơn;
3. hòa → Hit@1 cao hơn;
4. hòa → ít lần gọi model hơn, rồi thời gian trung vị thấp hơn.

**Agent** (chỉ dùng dev, nhãn đọc tay):
1. Grounded answer trung bình /2 cao hơn;
2. hòa → abstention đúng nhiều hơn, rồi ít từ chối nhầm hơn;
3. hòa → ít token/thời gian hơn.

Sau khi chốt, chỉ chạy test cho baseline (dense) và cấu hình đã chọn, không chỉnh sau khi xem test.

## Filter

`filter` đến từ câu hỏi hoặc ngữ cảnh người dùng đưa trước (ghi trong `filter_source`), không suy
ra từ gold. Cấu hình `use_filter: true` áp filter này; các câu không có filter chạy không filter.

## Giới hạn

- Một tài liệu, một văn phong; câu hỏi do AI hỗ trợ soạn ở Day 6 và đã được kiểm tra evidence tự động.
- Kết quả retrieval của test đã được báo trong Day 6 (A/B/C). Day 7 không dùng test để chỉnh,
  nhưng test không hoàn toàn "chưa từng thấy".
- d01 và d17 có evidence thay thế nằm trên trang test (p18, p39) vì cùng sự kiện được in hai lần.
