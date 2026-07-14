# KHMT K35 - NLP

## Đề tài 18 - HVB — Ngữ liệu Song ngữ Hán–Việt

Corpus xây dựng từ các tác phẩm lịch sử Việt Nam song ngữ Hán–Việt.

## Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env
# Chỉnh sửa .env: LLM_API_URL, LLM_MODEL_NAME, ...
```

## Chạy pipeline

```bash
# Bước 1 — OCR (Hán / Việt riêng biệt)
python src/run_ocr/run_ocr_chinese.py       # → data/ocr_output/*_sino_raw.txt
python src/run_ocr/run_ocr_vietnamese.py    # → data/ocr_output/*_vie_raw.txt

# Bước 2 — Dóng hàng câu
python src/sentence_alignment/run.py
python src/sentence_alignment/run.py --work-id HVB_001
python src/sentence_alignment/run.py --no-llm   # bỏ qua bước LLM refine
```

## Test nhanh (N trang đầu)

```bash
# Chạy OCR thử trên N trang đầu của một tác phẩm (default: 5 trang)
python test_ocr_5pages.py --work-id HVB_002 --lang sino
python test_ocr_5pages.py --work-id HVB_003 --lang vie
python test_ocr_5pages.py --work-id HVB_003 --lang vie --pages 10
python test_ocr_5pages.py --work-id HVB_002 --lang sino --no-llm

# Output: data/ocr_output/{work_id}_{lang}_test.txt
```

## Cấu trúc thư mục

```
data/
  config.json          ← khai báo tác phẩm (id, file paths, type)
  raw/
    sino/              ← file Hán/Nôm gốc (pdf, txt)
    vie/               ← file Việt gốc (pdf, txt)
  ocr_output/          ← text thô sau OCR (*_sino_raw.txt / *_vie_raw.txt)
  processed/           ← câu đã tách (JSON)
  corpus/              ← corpus song ngữ (TSV)
src/
  run_ocr/             ← Phần 1: OCR + LLM correction  [README trong thư mục]
    ocr_utils.py
    run_ocr_chinese.py
    run_ocr_vietnamese.py
    llm_corrector.py
  sentence_alignment/  ← Phần 2: dóng hàng câu
    run.py             ← LaBSE + DP alignment + LLM verify
  utils.py
configs/pipeline.yaml
test_ocr_5pages.py     ← test nhanh N trang đầu
```

## Data

```
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

## LLM Configuration (`.env`)

| Biến | Mô tả | Mặc định |
|------|--------|----------|
| `LLM_API_URL` | Endpoint OpenAI-compatible (LM Studio) | `http://localhost:1234/v1/chat/completions` |
| `LLM_MODEL_NAME` | Tên model | `qwen/qwen3-4b-2507` |
| `LLM_CHUNK_LINES` | Số dòng mỗi chunk gửi LLM | `100` |
| `LLM_OVERLAP_LINES` | Overlap giữa các chunk | `20` |
| `LLM_MAX_TOKENS` | Token output tối đa mỗi call | `4096` |
