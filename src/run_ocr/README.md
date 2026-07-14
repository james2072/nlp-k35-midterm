# `src/run_ocr` — OCR Pipeline

Module OCR chữ Hán/Nôm và Tiếng Việt từ PDF scan / ảnh / text, có bước hậu xử lý bằng LLM.

---

## Luồng xử lý

```
data/raw/sino/{slug}/   hoặc   data/raw/vie/{slug}/
         │
         ▼
   load_and_process_input()          [ocr_utils.py]
   (đọc PDF/ảnh/text → list pages)
         │
         ▼  (nếu là ảnh/scan)
   enhance_image()                   [ocr_utils.py]
   ┌─────────────────────────────┐
   │ 1. Upscale 2× (LANCZOS4)    │
   │ 2. Grayscale                │
   │ 3. CLAHE (tương phản cục bộ)│
   │ 4. Median Blur (khử noise)  │
   │ 5. Adaptive Threshold       │
   │ 6. Downscale → max 3500px   │
   └─────────────────────────────┘
         │
         ▼
   PaddleOCR (lang="ch" / "en")
   → raw bboxes + text
         │
         ▼
   sort_layout()
   ┌──────────────────────────────────────────┐
   │ Hán  → cột dọc, phải → trái             │
   │        (sort_vertical_layout)            │
   │ Việt → tự phát hiện dọc/ngang            │
   │        (smart_sort_layout)               │
   └──────────────────────────────────────────┘
         │
         ▼
   correct_text_with_llm()           [llm_corrector.py]
   ┌───────────────────────────────────────────────────┐
   │ Chia text thành chunks (100 dòng, overlap 20)    │
   │ Mỗi chunk → LLM local (OpenAI-compatible API)    │
   │ Ghép kết quả (trim overlap để không nhân đôi)    │
   │ Retry 3 lần nếu lỗi, fallback giữ text gốc      │
   └───────────────────────────────────────────────────┘
         │
         ▼
   clean_text()
   (xóa ký tự rác, chuẩn hóa khoảng trắng)
         │
         ▼
   data/ocr_output/{work_id}_sino_raw.txt
   data/ocr_output/{work_id}_vie_raw.txt
```

---

## Các file

| File | Vai trò |
|------|---------|
| `ocr_utils.py` | Load file (PDF/ảnh/text), enhance ảnh trước OCR |
| `run_ocr_chinese.py` | Pipeline Hán: PaddleOCR `lang=ch`, sort cột dọc |
| `run_ocr_vietnamese.py` | Pipeline Việt: PaddleOCR `lang=en`, smart layout |
| `llm_corrector.py` | Sửa lỗi OCR qua LLM local (chunked + overlap) |

---

## Cách chạy

```bash
# Chạy OCR toàn bộ 5 tác phẩm
python src/run_ocr/run_ocr_chinese.py     # → *_sino_raw.txt
python src/run_ocr/run_ocr_vietnamese.py  # → *_vie_raw.txt
```

---

## Config (`data/config.json`)

Mỗi tác phẩm có các trường:

| Trường | Ý nghĩa |
|--------|---------|
| `id` | Định danh (HVB_001 … HVB_005) |
| `sino_file` | Đường dẫn file Hán (từ project root) |
| `vie_file` | Đường dẫn file Việt (từ project root) |
| `sino_type` | `text` / `pdf_text` / `pdf_scan` |
| `vie_type` | `text` / `pdf_text` / `pdf_scan` |

**Loại file và cách xử lý:**

| Type | Xử lý |
|------|-------|
| `text` | Đọc `.txt` trực tiếp, **không OCR** |
| `pdf_text` | Trích text layer PDF (PyMuPDF), **không OCR** |
| `pdf_scan` | Render PDF → ảnh → enhance → **OCR** |

---

## LLM Chunking

```
Ví dụ: 250 dòng, CHUNK=100, OVERLAP=20, STRIDE=80

Chunk 0: dòng   0–99   (context: "")
Chunk 1: dòng  80–179  (context: dòng 60–79)
Chunk 2: dòng 160–249  (context: dòng 140–159)

Khi ghép: chunk 1 trim 20 dòng đầu → không bị nhân đôi
```

---

## Ước tính token (toàn bộ 5 tác phẩm)

| Nhóm | LLM Calls | Tokens |
|------|---:|---:|
| Sino | ~308 | ~624K |
| Vie | ~1,628 | ~1,274K |
| **Tổng** | **~1,936** | **~1.9M** |

> Thời gian ước tính với Qwen3-4B local: **13–35 giờ** tùy phần cứng.

**Tối ưu:** File `text` / `pdf_text` đã có text layer sạch — có thể dùng `--no-llm` để bỏ qua LLM correction.
