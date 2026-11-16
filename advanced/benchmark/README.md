# Benchmark nâng cao — World Development Report 2025

`questions.json` được tạo bằng `python advanced/benchmark/build.py`; `python advanced/benchmark/check.py`
kiểm tra mọi trích dẫn văn bản nằm nguyên văn trên đúng trang PDF.

| Nhóm chính | Dev | Test |
|---|---:|---:|
| Bảng | 4 | 6 |
| Ảnh, sơ đồ, bản đồ | 4 | 5 |
| Biểu đồ | 3 | 5 |
| Multi-hop (≥ 2 trang) | 1 | 4 |
| Text | 4 | 4 |
| **Có đáp án** | **16** | **24** |
| Thiếu bằng chứng | 4 | 6 |
| **Tổng** | **20** | **30** |

- Mỗi câu có `id`, `group` (một nhóm chính), câu hỏi, `expected_answer` + `answer_key` (các ý bắt buộc), và
  `evidence` gồm trang PDF, trang in (= trang PDF − 32), phần tử (bảng/hàng/cột, hình/nhánh, đoạn text) và
  trích dẫn hoặc mô tả vùng. Câu tính toán có `calculation`. Câu thiếu bằng chứng có `absence_check`.
- Đáp án được đọc từ **bản gốc** (ảnh trang render từ PDF và text layer), không lấy từ nội dung model trích xuất.
- Dev và test dùng phần tử khác nhau, trừ Bảng 4.2 và Bảng S2.1, nơi dev và test hỏi các hàng khác nhau.
- Split chốt trước mọi lần chạy. Agent chỉ nhận `id` và câu hỏi, không nhận đáp án hay evidence.

## Tiêu chí chọn cấu hình (chốt trước khi chạy dev)

1. Tỷ lệ đạt trên dev (chấm đạt/không đạt theo `ADVANCED.md`) cao hơn;
2. hòa → ít câu thiếu bằng chứng bị trả lời bừa hơn;
3. hòa → ít lần gọi model hơn.

Sau khi chốt, chỉ chạy **cấu hình đã chọn** trên toàn bộ test một lần; không sửa code, prompt hay
benchmark sau khi xem kết quả test. Nếu buộc phải sửa, cần tạo bộ test mới.

## Kiểm tra câu thiếu bằng chứng

Mỗi câu ghi trong `absence_check` trang đã đọc và từ khóa đã tìm trên toàn bộ text layer (408 trang),
ví dụ: Bảng 4.2 không có Ấn Độ hay Kenya; Bảng B6.5.1 chỉ có quạt và tủ lạnh; Figure 7.2 không có Nhật Bản.
