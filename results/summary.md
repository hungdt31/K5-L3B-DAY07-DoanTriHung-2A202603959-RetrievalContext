# Kết quả Day 7 (tự sinh bởi `python -m src.day7.summarize`)

## Retrieval với query gốc

| Cấu hình | Phương pháp | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Hit@1 / Hit@3 theo nhãn Day 6 gốc | Ký tự context TB | Thời gian trung vị | Chi phí thêm |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | bm25, top-3, filter=không, budget 3000 ký tự | dev | 14/18 (78%) | 18/18 (100%) | 18/18 (100%) | 13/18 / 17/18 | 2539 | 0.002 s | 0 lần gọi LLM |
| dense | dense, top-3, filter=không, budget 3000 ký tự | dev | 18/18 (100%) | 18/18 (100%) | 18/18 (100%) | 17/18 / 17/18 | 2454 | 0.05 s | 0 lần gọi LLM |
| hybrid | hybrid, top-3, pool 20, filter=không, budget 3000 ký tự | dev | 17/18 (94%) | 18/18 (100%) | 18/18 (100%) | 15/18 / 18/18 | 2469 | 0.046 s | 0 lần gọi LLM |
| hybrid_filter | hybrid, top-3, pool 20, filter=có, budget 3000 ký tự | dev | 17/18 (94%) | 18/18 (100%) | 18/18 (100%) | 16/18 / 18/18 | 2528 | 0.049 s | 0 lần gọi LLM |
| hybrid_rerank | hybrid_rerank, top-3, pool 12, filter=có, budget 3000 ký tự | dev | 18/18 (100%) | 18/18 (100%) | 18/18 (100%) | 17/18 / 18/18 | 2442 | 1.744 s | 18 lần gọi LLM, 49202 token |
| dense | dense, top-3, filter=không, budget 3000 ký tự | test | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) | 6/6 / 6/6 | 2656 | 0.504 s | 0 lần gọi LLM |
| hybrid_filter | hybrid, top-3, pool 20, filter=có, budget 3000 ký tự | test | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) | 6/6 / 6/6 | 2456 | 0.06 s | 0 lần gọi LLM |

Query embedding: `gemini-embedding-001` 768 chiều, 1 lần gọi/câu hỏi mới (cache theo nội dung query, lần gọi đầu ~0.4–0.5 s). Thời gian trên đo khi cache đã có, trừ khi ghi khác.

## Agent (model thật, nhãn đọc tay)

| Cấu hình / split | Tool use | Số lượt search | Grounded 0/1/2 | Citation hợp lệ | Câu trả lời có citation | Abstention đúng | Từ chối nhầm | Context Evidence Rate | Token | Thời gian | Lỗi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agent_dense / dev | 20/20 (100%) | {'1': 16, '2': 3, '3': 1} (TB 1.25) | 0/0/18, TB 2.00/2 | 21/21 (100%) | 18/18 (100%) | 2/2 (100%) | 0/18 (0%) | 18/18 (100%) | 45 lần gọi, 35998+2946 token | trung vị 2.47 s, tổng 135.3 s | 0 (chạy lại: 0) |
| agent_dense_1search / dev | 20/20 (100%) | {'1': 20} (TB 1) | 1/0/17, TB 1.89/2 | 19/19 (100%) | 17/17 (100%) | 2/2 (100%) | 1/18 (6%) | 17/18 (94%) | 44 lần gọi, 34386+2853 token | trung vị 2.55 s, tổng 289.6 s | 0 (chạy lại: 3) |
| agent_hybrid_filter_k5 / dev | 20/20 (100%) | {'1': 18, '2': 1, '3': 1} (TB 1.15) | 0/0/18, TB 2.00/2 | 22/22 (100%) | 18/18 (100%) | 2/2 (100%) | 0/18 (0%) | 18/18 (100%) | 46 lần gọi, 51555+3019 token | trung vị 2.66 s, tổng 174.0 s | 0 (chạy lại: 0) |
| agent_dense / test | 8/8 (100%) | {'1': 6, '2': 1, '3': 1} (TB 1.38) | 0/0/6, TB 2.00/2 | 6/6 (100%) | 6/6 (100%) | 2/2 (100%) | 0/6 (0%) | 6/6 (100%) | 20 lần gọi, 19000+1252 token | trung vị 2.53 s, tổng 41.7 s | 0 (chạy lại: 0) |
| agent_dense_1search / test | 8/8 (100%) | {'1': 8} (TB 1) | 0/0/6, TB 2.00/2 | 6/6 (100%) | 6/6 (100%) | 2/2 (100%) | 0/6 (0%) | 6/6 (100%) | 17 lần gọi, 13925+1129 token | trung vị 2.18 s, tổng 70.1 s | 0 (chạy lại: 0) |

## Từng câu test — cấu hình đã chọn (`agent_dense`)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t01 | có đáp án | search("Cabo Verde airports passenger volumes") → p13, p13, p13 | answered: Passenger volumes have experienced a 60 percent surge since 2022, driven by the financing that modernized and greened Cabo Verde's seven airports and established 35 new air routes [wbg-ar-2026:p013:para1000:002]. | grade 2: Đúng: tăng 60% lượng hành khách từ 2022 (câu hỏi chỉ hỏi mức tăng); citation p13 chứa nguyên văn. |
| t02 | có đáp án | search("Country Partnership Framework usually last maximum") → p18, p18, p31 | answered: Country Partnership Frameworks (CPFs) typically span four to six years, with a maximum duration of 10 years [wbg-ar-2026:p018:para1000:003]. | grade 2: Đúng và đủ cả hai điều kiện: thường 4–6 năm, tối đa 10 năm; citation p18. |
| t03 | có đáp án | search("sanctioned settlement agreements fiscal 2026 entities individuals") → p24, p40, p42 | answered: In fiscal 2026, the World Bank Group sanctioned eight entities and individuals via settlement agreements [wbg-ar-2026:p024:para1000:001]. | grade 2: Đúng (8) và đúng loại xử phạt (settlement agreements), không nhầm với 16/3 của hai hình thức khác; citation p24. |
| t04 | có đáp án | search("IDA issue capital markets fiscal 2026 currencies") → p31, p19, p31; search("IDA issued currencies fiscal 2026 equivalent") → p31, p31, p29 | answered: In fiscal 2026, IDA issued $13.9 billion equivalent in the capital markets across six currencies [wbg-ar-2026:p031:para1000:001]. | grade 2: Đúng và đủ: $13.9 tỷ, 6 loại tiền; citation p31. |
| t05 | có đáp án | search("gross outstanding guarantee portfolio June 30 2026") → p39, p33, p39 | answered: As of June 30, 2026, MIGA's gross outstanding guarantee portfolio reached a record high of $41.4 billion [wbg-ar-2026:p039:para1000:000]. | grade 2: Đúng giá trị được hỏi ($41.4 tỷ, kỷ lục, MIGA); không nêu 'tăng 13%' nhưng câu hỏi chỉ hỏi giá trị danh mục; citation p39. |
| t06 | có đáp án | search("ICSID arbitration cases decided awarded no damages fiscal 2026") → p44, p44, p43 | answered: In fiscal 2026, 70 percent of arbitration cases decided by tribunals resulted in no damages awarded to claimants, whether due to jurisdictional dismissals, findings of no liability, or liability without damages [wbg-ar-2026:p044:para1000:000]. | grade 2: Đúng (70%) kèm điều kiện (bác thẩm quyền, không có trách nhiệm, có trách nhiệm nhưng không bồi thường); citation p44. |
| u03 | thiếu evidence | search("Cabo Verde airports passengers 2025") → p13, p13, p13 | abstain: The World Bank Group Annual Report 2026 does not state the exact number of passengers Cabo Verde's airports handled in 2025, though it mentions a 60 percent surge in passenger volumes since 2022 [wbg-ar-2026:p013:para1000:002]. | abstention_correct=True: Từ chối đúng: nói rõ tài liệu không có số hành khách tuyệt đối, chỉ nêu mức tăng 60% có trong tài liệu, không bịa số. |
| u04 | thiếu evidence | search("ICSID tribunal damages award fiscal 2026 US dollars largest") → p44, p44, p43; search("ICSID damages award USD fiscal 2026 largest") → p44, p44, p43; search("million billion damages ICSID fiscal 2026") → p44, p44, p43 | abstain: The World Bank Group Annual Report 2026 notes that 8 percent of ICSID arbitration cases decided by tribunals in fiscal 2026 resulted in awards exceeding $50 million, but it does not specify the exact amount of the largest single damages award [wbg-ar-2026:p044:para1000:000]. | abstention_correct=True: Từ chối đúng: chỉ nêu điều tài liệu có (8% vụ > $50 triệu), không suy ra con số cụ thể. |
