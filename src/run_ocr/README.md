# `src/run_ocr` — OCR Pipeline cho Tiếng Việt

Module OCR dành riêng cho Tiếng Việt từ PDF scan / ảnh / text, có bước hậu xử lý sửa lỗi bằng LLM (hoặc chạy mộc).
Các thành phần đã được đơn giản hóa và tập trung gọn gàng tại thư mục con `vie/`.

---

## Cấu trúc thư mục

```
src/run_ocr/
├── vie/
│    ├── run.py            # Entry point chính chạy toàn bộ pipeline OCR + LLM cho tiếng Việt
│    ├── ocr_utils.py      # Bộ tiện ích OCR tiếng Việt (PaddleOCR lang='vi', smart_sort_layout, enhance_image)
│    └── llm_corrector.py  # Bộ sửa lỗi OCR qua LLM (OpenAI-compatible / Gemini) chuyên biệt cho tiếng Việt
└── weights/               # Thư mục chứa model weights của PaddleOCR
```

---

## Luồng xử lý (`src/run_ocr/vie/run.py`)

```
data/raw/vie/{slug}/
         │
         ▼
   load_and_process_input()          [ocr_utils.py]
   (đọc PDF/ảnh/text → list pages)
         │
         ▼  (nếu là ảnh/scan)
   enhance_image()                   [ocr_utils.py]
   ┌──────────────────────────────────────────────┐
   │ 1. Upscale tối đa 2× (LANCZOS4)              │
   │ 2. Grayscale                                 │
   │ 3. CLAHE (tăng tương phản cục bộ)           │
   │ 4. Median Blur (khử nhiễu đốm li ti)         │
   │ 5. Unsharp Masking (làm sắc nét viền chữ)    │
   │ 6. Downscale về tối đa 3500px                │
   └──────────────────────────────────────────────┘
         │
         ▼
   PaddleOCR (lang="vi")             [ocr_utils.py]
   → phát hiện bounding box + nhận diện text tiếng Việt
         │
         ▼
   smart_sort_layout()               [ocr_utils.py]
   (tự động sắp xếp dòng chữ từ trên xuống dưới, trái sang phải)
         │
         ▼
   correct_text_with_llm()           [llm_corrector.py]
   ┌───────────────────────────────────────────────────┐
   │ Chia text thành chunks (100 dòng, overlap 20)    │
   │ Mỗi chunk → LLM API (OpenAI-compatible)          │
   │ Ghép kết quả (trim overlap an toàn 1-to-1)       │
   │ filter_for_alignment() loại bỏ cước chú/số trang │
   └───────────────────────────────────────────────────┘
         │
         ▼
   data/ocr_output/{work_id}_vie_raw.txt
```

---

## Cách chạy pipeline Tiếng Việt

Chạy lệnh duy nhất từ thư mục gốc của project:

```bash
python src/run_ocr/vie/run.py
```

---

## Config (`data/config.json`)

Từng tác phẩm được cấu hình qua file `data/config.json`:

| Trường | Ý nghĩa |
|--------|---------|
| `id` | Định danh tác phẩm (HVB_001 … HVB_005) |
| `vie_file` | Đường dẫn file Tiếng Việt (từ project root) |
| `vie_type` | Loại dữ liệu (`text` / `pdf_text` / `pdf_scan`) |

**Loại file và cách xử lý:**

| Type | Xử lý |
|------|-------|
| `text` | Đọc `.txt` trực tiếp, **không chạy OCR** |
| `pdf_text` | Trích text layer từ PDF (PyMuPDF), **không chạy OCR** |
| `pdf_scan` | Render PDF → ảnh → tiền xử lý (`enhance_image`) → **PaddleOCR (`lang="vi"`)** + **LLM Corrector** |
