# Báo cáo Cập nhật Phương pháp Dóng hàng (Changelog)

**Đề tài:** HVB18 - Đồ án xây dựng tập dữ liệu (corpus) song ngữ Hán–Việt từ các tác phẩm lịch sử, sử dụng Google AI Studio để tự động dóng hàng (Sentence Alignment).

**Nhóm thực hiện:**
- 25C11027 -- Bùi Quốc Việt
- 25C11006 -- Nguyễn Tất Hưng
- 25C11011 -- Nguyễn Ngọc Hồng Lĩnh
- 25C11065 -- Trần Quốc Thịnh

**Mã nguồn (Source Code):** [https://github.com/james2072/nlp-k35-midterm](https://github.com/james2072/nlp-k35-midterm)
**Báo cáo chi tiết:** Cập nhật phương pháp mới tương ứng trong [report.pdf](./report.pdf) đính kèm trong cùng thư mục.

Tài liệu này báo cáo quá trình tái cấu trúc quy trình dóng hàng câu (Sentence Alignment) của nhóm, chuyển đổi từ mô hình tiếp cận lai (Hybrid: Heuristic + LLM) sang phương pháp tự động hóa hoàn toàn bằng Mô hình Ngôn ngữ lớn (LLM).

## 1. Phương pháp Cũ: Ensemble Embedding + DeepSeek Refinement (Đã loại bỏ)

**Quy trình:**
1. **Ensemble Embedding:** Mã hóa (Encode) câu Hán và Việt qua cả LaBSE lẫn BERT-base-multilingual, sau đó kết hợp với length-based similarity (Gale-Church) theo trọng số 0.50 / 0.30 / 0.20.
2. **Greedy Best-Match:** Với mỗi câu Hán, thuật toán chọn câu Việt có ensemble similarity cao nhất. Các xung đột được giải quyết bằng cách ưu tiên câu Hán có score cao hơn; câu còn lại bị đẩy sang lựa chọn tốt tiếp theo.
3. **LLM Refinement (DeepSeek API):** Các cặp có score < 0.40 được gửi theo batch (mỗi lần 3 cặp) cho DeepSeek để kiểm tra tính tương đương ngữ nghĩa và sửa các lỗi OCR còn sót.

**Hạn chế:**
- **Kém bền vững với nhiễu OCR:** Các thuật toán tính khoảng cách vector và độ dài chữ rất dễ bị đánh lừa khi file OCR gốc chứa nhiều rác, gây lệch pha hàng loạt.
- **Hiện tượng Ảo giác (Hallucination):** Việc cắt nhỏ lẻ cho DeepSeek hậu kiểm khiến LLM mất ngữ cảnh toàn cục, dẫn đến tự sinh chữ hoặc lặp câu.

## 2. Phương pháp Mới: LLM Structured Output

**Quy trình:**
Loại bỏ hoàn toàn các bước trung gian (Regex, LaBSE, DP). Trực tiếp sử dụng Google AI Studio xử lý cặp văn bản thô theo từng lô (batch).

**Ưu điểm học thuật & kỹ thuật:**
- **Cơ chế Structured Output:** Ràng buộc LLM xuất dữ liệu theo định dạng JSON nghiêm ngặt, sử dụng bản Hán làm tham chiếu chuẩn (Ground Truth).
- **Phân tách ngữ nghĩa động (Fine-grained Segmentation):** Khắc phục triệt để điểm yếu của phương pháp cũ khi xử lý thể loại thơ. LLM tự động nhận diện cấu trúc thơ/văn xuôi để ngắt vế câu.
- **Khử nhiễu tự động:** Tự động đối chiếu chéo hai bản Hán-Việt để loại bỏ hoàn toàn nhiễu OCR một chiều và chuẩn hóa dấu câu.

---

## 3. Hướng dẫn Thực thi

Chương trình đầu ra trực tiếp tập tin `{work_id}_parallel.tsv` chứa ngữ liệu đã được dóng hàng.

**Lệnh thực thi:**
```bash
python src/sentence_alignment/run.py --work-id HVB_005 --id-start 0 --n 5 --k 1
```

**Các tham số:**
- `--work-id`: Mã định danh tác phẩm (VD: `HVB_005`, `HVB_004`).
- `--id-start`: Chỉ mục ID khởi tạo (VD: `0` sinh ra mã định danh `HVB_005_0000`).
- `--n`: Kích thước phân hệ (chunk batch size) gộp trong mỗi truy vấn API (Mặc định: 5).
- `--k`: Kích thước ngữ cảnh lề (context overlap) hỗ trợ LLM tham khảo (Mặc định: 1).

## 4. Cấu trúc Thư mục

```text
data/
  ocr_output/          ← Thư mục chứa text thô (Hán và Việt) cần dóng hàng
  corpus/              ← Thư mục lưu kết quả song ngữ TSV/XLSX
src/
  ocr/  ← File Notebook chạy OCR
  sentence_alignment/
    run.py  ← Script chính chạy dóng hàng bằng Google AI Studio
```
