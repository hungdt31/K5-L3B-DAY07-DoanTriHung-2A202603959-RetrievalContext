# Thử thách nâng cao: đánh giá hiện trạng và việc cần làm

Cập nhật: 16/11/2026, 23:20. Tài liệu chọn: *World Development Report 2025: Standards for Development* (World Bank).

## 1. Kết luận

**Hiện chưa đủ điều kiện nhận 15 điểm bonus.** Tài liệu, benchmark và toàn bộ code của pipeline đã xong. Phần còn thiếu nằm ở bước chạy và chấm: chưa chạy hết dev, chưa chốt cấu hình, chưa chạy test, chưa có nhãn và tỷ lệ đạt. Theo `ADVANCED.md`, bonus chỉ được cộng khi đạt **≥ 27/30 câu test** và nộp đủ bằng chứng tái lập. Không có điểm từng phần.

Lý do dừng: các model Gemini ở free tier chỉ cho 500 request/ngày cho mỗi model (embedding: 1000/ngày). Quota của `gemini-3.5-flash-lite`, `gemini-3.6-flash` và `gemini-3.5-flash` đã hết trong ngày. `gemini-3.1-flash-lite` thì thường trả lỗi 503 do server quá tải.

## 2. Đối chiếu với từng yêu cầu của `ADVANCED.md`

| Yêu cầu | Trạng thái | Bằng chứng / còn thiếu |
|---|---|---|
| Tài liệu thực tế ≥ 100 trang, có bảng, ảnh, biểu đồ; ghi nguồn, phiên bản, quyền | ✅ Xong | 408 trang PDF, CC BY 3.0 IGO, `hdl.handle.net/10986/43949`, sha256 `9613c40c…be536` (`advanced/index/index_stats.json`, `src/adv/config.py`). PDF không commit; tải lại qua `download_url`. |
| Index toàn bộ nội dung hữu ích | 🟡 Một phần | Text layer: 398/408 trang có chunk (1445 chunk). 10 trang không có chunk text: 2, 24, 32, 98 (trống hoặc trang trí), 268, 298, 340, 374, 406, 407 (rất ít chữ, nhiều khả năng là trang mở phần hoặc bìa; cần xác nhận bằng mắt và chạy vision cho các trang này). Vision: **267/408 trang** đã trích (250 trang bằng `gemini-3.5-flash-lite`, 17 bằng `gemini-3.6-flash`) → 90 phần tử trực quan trên 88 trang. **141 trang chưa có vision** (42 lỗi 429, 99 chưa chạy, phần lớn trong khoảng p250–408) nên hiện chỉ được index bằng text layer. |
| Liên kết về trang gốc; lưu vùng hoặc crop cho bảng/ảnh/biểu đồ | ✅ Xong | ID chunk chứa số trang PDF (`p185-v1`); metadata gồm trang in, nhãn, `bbox_0_1000` và đường dẫn crop. Số trang in = trang PDF − 32. |
| Pipeline lấy evidence từ cả text lẫn nội dung trực quan | ✅ Code xong | Text layer, vision transcription và embedding cục bộ `bge-small-en-v1.5` (pgvector `adv_chunks`), cộng hybrid BM25 + RRF. Agent có 3 tool: `search_document`, `read_page`, `look_at_page` (xem lại ảnh trang gốc khi trả lời). |
| Kiểm tra nội dung trích xuất với bản gốc | 🟡 Một phần | `advanced/index/extraction_manual_check.json`: đã soát tay 7 phần tử; 4 đúng, 1 đúng nhưng thiếu, **2 sai một phần** (sai hướng mũi tên ở Figure 3.1, sai một ô ở Figure 3.13). Script `python -m src.adv.verify_extraction` đối chiếu số liệu với text layer đã viết nhưng **chưa chạy**. |
| Benchmark riêng, chia dev/test trước khi thử | ✅ Xong | `advanced/benchmark/questions.json`. Test có 30 câu: 6 bảng, 5 ảnh/sơ đồ, 5 biểu đồ, 4 multi-hop, 4 text, 6 thiếu bằng chứng. Dev có 20 câu: 16 có đáp án và 4 thiếu bằng chứng. Đáp án đọc từ bản gốc; `check.py` xác nhận 11/11 trích dẫn văn bản; tiêu chí chọn đã chốt trong `benchmark/README.md`. |
| Thử cấu hình trên dev, chốt pipeline | ❌ Chưa | 3 cấu hình đã định nghĩa (`text_dense`, `mm_hybrid`, `mm_hybrid_look`). Dev mới chạy được 10/20 câu của `text_dense` (bằng `gemini-3.5-flash-lite`, rồi bị dừng để đổi model) và chưa chấm. |
| Chạy test, lưu trace, chấm từng câu, metrics theo nhóm | ❌ Chưa | `src/adv/agent.py` lưu trace đầy đủ; `src/adv/grade.py` tạo bảng review và tính tỷ lệ đạt theo nhóm. |
| Báo cáo: lỗi, thời gian, chi phí, giới hạn | ❌ Chưa | Mục "Thử thách nâng cao" trong báo cáo cá nhân cần điền sau khi có kết quả test. |

## 3. Quan sát sơ bộ (chưa chấm, chỉ mang tính định tính)

Trong 10 câu dev mà baseline `text_dense` (chỉ dùng text) đã chạy:
- **Từ chối cả 3 câu cần đọc hình:** d-chart1 (Figure 4.7), d-chart2 (Figure 4.1) và d-img4 (Map 6.2). Các giá trị trên biểu đồ và màu bản đồ không có trong text layer, nên đây là lý do cần cấu hình đa phương thức.
- **Từ chối d-tab2** (Bảng 2.2). Bảng chữ nhiều cột bị text layer xé vụn, tìm không ra.
- Trả lời được d-tab1, d-tab3, d-tab4, d-img1, d-img2, d-img3, vì chữ trong các bảng và sơ đồ này có trong text layer. Độ đúng của các câu này chưa được chấm.

## 4. Rủi ro với ngưỡng 27/30

1. **6/30 câu test có trang evidence chưa được trích vision**: t-tab1 (p326), t-img4 (p385), t-img5 (p391), t-chart3 (p320), t-chart4 (p347), t-txt4 (p287). Với các câu này, agent chỉ có text layer, hoặc phải tự gọi `look_at_page`.
2. **Vision lúc index có thể sai** ở sơ đồ và biểu đồ (2/7 mẫu soát tay sai một phần). Câu hỏi về hướng mũi tên hoặc ô trong ma trận cần được xem lại bằng `look_at_page`.
3. **Biểu đồ không in số** (Figure 4.10, 4.6, 6.7, 7.2) đòi hỏi đọc xấp xỉ. Đáp án đã ghi khoảng chấp nhận được, nhưng model vẫn có thể đọc lệch.
4. **Multi-hop** cần trích dẫn đủ cả hai trang; thiếu một trang là không đạt.
5. **Quota và lỗi 503:** chạy dev và test cần khoảng 350–400 request. Vừa đủ quota một ngày của một model, nếu không phải chạy lại.
6. Nếu test dưới 27/30, **không được chỉnh theo test**; phải tạo bộ test mới rồi mới được báo điểm.

## 5. Việc cần làm (theo thứ tự)

Mỗi bước ghi số request ước tính. Nên dùng **một model duy nhất** cho dev và test (trong `src/adv/config.py`: `AGENT_MODEL`, `LOOK_MODEL`, hoặc biến môi trường `ADV_AGENT_MODEL`/`ADV_LOOK_MODEL`).

| # | Việc | Lệnh | Request |
|---|---|---|---:|
| 1 | Bật pgvector Day 6 | `cd ../K5-L3B-DAY06-…/starter && docker compose -p day6_student_starter up -d --wait` | 0 |
| 2 | *(Nên làm)* Trích vision cho 141 trang còn thiếu (trang lỗi được tự chạy lại) | `ADV_VISION_MODEL=<model còn quota> .venv/bin/python -m src.adv.extract vision` | ~141 |
| 3 | Build lại index (embedding cục bộ, không tốn quota) | `.venv/bin/python -m src.adv.index` | 0 |
| 4 | Kiểm tra trích xuất với bản gốc, soát tay thêm vài phần tử mới | `.venv/bin/python -m src.adv.verify_extraction` | 0 |
| 5 | Chạy dev cả 3 cấu hình | `.venv/bin/python -m src.adv.agent --split dev --configs text_dense mm_hybrid mm_hybrid_look` (lỗi mạng: thêm `--retry-errors`) | ~200–250 |
| 6 | Đọc và chấm từng câu dev, viết `labels.json` (`query_id`, `pass`, `reason`) | `.venv/bin/python -m src.adv.grade sheet --split dev --config <cfg>` → `… score …` | 0 |
| 7 | Phân tích lỗi dev; nếu cần, sửa prompt, top-k hoặc tool **trên dev**, chạy lại dev | như bước 5–6 | ~70/lần |
| 8 | Chốt cấu hình theo tiêu chí trong `benchmark/README.md`, ghi `advanced/results/selection.json` (kèm thời điểm) | — | 0 |
| 9 | Chạy **một lần** cấu hình đã chọn trên test | `.venv/bin/python -m src.adv.agent --split test --configs <cfg>` | ~120–150 |
| 10 | Đọc và chấm từng câu test, đối chiếu evidence với bản gốc, rồi tính metrics | `grade sheet` → `labels.json` → `grade score` (ra `pass n/N`, theo nhóm, lỗi, token, thời gian) | 0 |
| 11 | Điền mục "Thử thách nâng cao" trong báo cáo cá nhân | — | 0 |

Kế hoạch quota gợi ý: ngày 1 làm bước 2–7 với một model (≈ 400–450 request); ngày 2 làm bước 9 sau khi quota reset (khoảng 15:00 giờ VN). Nếu bật billing cho key, có thể chạy hết trong một lần với chi phí rất nhỏ (flash-lite tính theo token).

### Cải tiến nên thử trên dev để tăng khả năng đạt
- Yêu cầu agent **luôn gọi `look_at_page`** cho câu nhắc tới Figure/Map hoặc hỏi về hướng, màu, giá trị trên biểu đồ.
- Với bảng dài hoặc bảng nối trang (Bảng 4.2, S2.3): gợi ý agent `read_page` trang tiếp theo.
- Với multi-hop: kiểm tra trước khi trả lời rằng mỗi ý đều có citation, và citation trải ra từ 2 trang trở lên.
- Giữ ngưỡng từ chối chặt với câu hỏi "năm/quốc gia/hạng mục không có trong hình" (các câu kiểu t-u2, t-u3, t-u4).

## 6. File liên quan

| Đường dẫn | Nội dung |
|---|---|
| `src/adv/config.py` | Nguồn tài liệu, model, đường dẫn |
| `src/adv/extract.py` | Text layer theo trang, render ảnh trang, vision transcription (cache theo trang trong `advanced/work/visual/`) |
| `src/adv/index.py` | Chunk text và phần tử trực quan, bbox/crop, embedding cục bộ, pgvector `adv_chunks`, thống kê |
| `src/adv/verify_extraction.py` | Đối chiếu số liệu trong transcription với text layer gốc |
| `src/adv/agent.py` | Agent 3 tool, 3 cấu hình, trace từng câu |
| `src/adv/grade.py` | Bảng review và tổng hợp tỷ lệ đạt theo nhóm từ nhãn chấm tay |
| `advanced/benchmark/` | `build.py`, `questions.json`, `check.py`, `README.md` (tiêu chí chọn) |
| `advanced/index/` | `chunks.json`, `index_stats.json`, `extraction_manual_check.json` |
| `advanced/work/` | `pages_text.jsonl`, `visual/pNNN.json`, log chạy. Ảnh render, crop và model là dữ liệu tái tạo được nên đã gitignore. |
