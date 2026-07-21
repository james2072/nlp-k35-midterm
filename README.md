# KHMT K35 - NLP - Midterm Project

## Đề tài 18 - HVB — Ngữ liệu Song ngữ Hán–Việt

Corpus xây dựng từ các tác phẩm lịch sử Việt Nam song ngữ Hán–Việt.

## Data

```text
https://drive.google.com/drive/folders/1szTHxYqiYGSeg5eutJKjRkRfEayf40Hl?usp=sharing
```

## Danh sách tác phẩm

| ID | Tên tác phẩm | Sino type | Vie type |
|----|---|---|---|
| HVB_001 | An Nam Chí Lược | text | pdf_text |
| HVB_002 | An Nam Chí Nguyên | pdf_scan | pdf_scan |
| HVB_003 | Công Dư Tiệp Ký | pdf_scan | pdf_scan |
| HVB_004 | Đại Nam Quốc Sử Diễn Ca | pdf_scan | pdf_text |
| HVB_005 | Đại Việt Lịch Triều Đăng Khoa Lục | text | pdf_scan |

## Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env
```

## Cấu hình cho LLM (`.env`)
Có thể sử dụng cho `Local` (`Ollama`, `LM Studio`) hoặc thông qua các `LLM API Provider` (`OpenAI`, `Google AI Studio`, ...)


| Biến | Mô tả | Mặc định |
|------|--------|----------|
| `LLM_API_URL` | Endpoint OpenAI-compatible (LM Studio) | `http://localhost:1234/v1/chat/completions` |
| `LLM_MODEL_NAME` | Tên model | `qwen/qwen3-4b-2507` |
| `LLM_API_KEY` | API Key của LLM service | `lm-studio` |
| `LLM_TIMEOUT` | Thời gian timeout cho LLM service | `900` |
| `LLM_MAX_RETRIES` | Số lần retry cho LLM service | `4` |


## Chạy pipeline

**Bước 1 — OCR (Hán / Việt riêng biệt)**
Mở và thực thi các Jupyter Notebook trong thư mục `src/ocr/`:
- `src/ocr/sino_ocr.ipynb` (Dành cho văn bản chữ Hán/Nôm)
- `src/ocr/vie_ocr.ipynb` (Dành cho văn bản chữ Quốc Ngữ)

**Bước 2 — Dóng hàng câu**
```bash
python src/sentence_alignment/run.py
python src/sentence_alignment/run.py --work-id HVB_001
python src/sentence_alignment/run.py --no-llm   # bỏ qua bước LLM refine
```

## Cấu trúc thư mục

```text
data/
  config.json          ← khai báo tác phẩm (id, file paths, type)
  raw/
    sino/              ← file Hán/Nôm gốc (pdf, txt)
    vie/               ← file Việt gốc (pdf, txt)
  ocr_output/          ← text thô sau OCR (*_sino_raw.txt / *_vie_raw.txt)
  processed/           ← câu đã tách (JSON)
  corpus/              ← corpus song ngữ (TSV)
src/
  ocr/                 ← Phần 1: OCR
    sino_ocr.ipynb
    vie_ocr.ipynb
  sentence_alignment/  ← Phần 2: dóng hàng câu
    run.py             ← LaBSE + DP alignment + LLM
```