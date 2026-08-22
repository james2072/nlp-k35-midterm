# KHMT K35 - NLP - Midterm Project

**Nhóm thực hiện:**
- 25C11027 -- Bùi Quốc Việt
- 25C11006 -- Nguyễn Tất Hưng
- 25C11011 -- Nguyễn Ngọc Hồng Lĩnh
- 25C11065 -- Trần Quốc Thịnh

## Đề tài 18 - HVB — Ngữ liệu Song ngữ Hán–Việt

Đồ án xây dựng tập dữ liệu (corpus) song ngữ Hán–Việt từ các tác phẩm lịch sử, sử dụng Google AI Studio để tự động dóng hàng (Sentence Alignment).

## Danh sách tác phẩm

| ID | Tên tác phẩm |
|---|---|
| HVB_001 | An Nam Chí Lược |
| HVB_002 | An Nam Chí Nguyên |
| HVB_003 | Công Dư Tiệp Ký |
| HVB_004 | Đại Nam Quốc Sử Diễn Ca |
| HVB_005 | Đại Việt Lịch Triều Đăng Khoa Lục |

## Cài đặt

```bash
# Cài đặt thư viện
pip install -r requirements.txt

# Tạo file .env và cấu hình API Key
cp .env.example .env
```

Trong file `.env`, khai báo:
```env
LLM_API_KEY="<ai_studio_api_key>"
LLM_MODEL_NAME="<ai_studio_model>"
```

## Cách chạy Dóng hàng (Alignment)

Pipeline sử dụng tính năng **Structured Output** của Google AI Studio để tự động chẻ câu, nhận diện thơ/văn xuôi, lọc rác OCR và trả về file `{work-id}_parallel.tsv`.

Cú pháp chạy:
```bash
python src/sentence_alignment/run.py --work-id HVB_005 --id-start 0 --n 5 --k 1
```

**Các tham số:**
- `--work-id`: Mã tác phẩm (ví dụ: `HVB_00X`)
- `--id-start`: ID bắt đầu đánh số (ví dụ: `0` sẽ sinh ra `HVB_00X_0000`)
- `--n`: Số lượng chunk gộp lại trong 1 lần gọi API (mặc định: 5)
- `--k`: Số lượng chunk trước đó dùng làm context (ngữ cảnh) để LLM tham khảo (mặc định: 1)

## Cấu trúc thư mục

```text
data/
  ocr_output/          ← Thư mục chứa text thô (Hán và Việt) cần dóng hàng
  corpus/              ← Thư mục lưu kết quả song ngữ TSV/XLSX
src/
  ocr/  ← File Notebook chạy OCR
  sentence_alignment/
    run.py  ← Script chính chạy dóng hàng bằng Google AI Studio
```