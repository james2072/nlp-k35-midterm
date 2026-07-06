# KHMT K35 - NLP

## Đề tài 18 - HVB — Ngữ liệu Song ngữ Hán–Việt

Corpus xây dựng từ các tác phẩm lịch sử Việt Nam song ngữ Hán–Việt.

## Cài đặt

```bash
pip install -r requirements.txt
pip install hanlp   # cài riêng (~500MB model)
cp .env.example .env
# Điền GEMINI_API_KEY vào .env
```

## Chạy pipeline

```bash
# Bước 1 — OCR + tách câu
python src/ocr_segmentation/run.py
python src/ocr_segmentation/run.py --work-id an-nam-chi-luoc

# Bước 2 — Dóng hàng câu
python src/sentence_alignment/run.py
python src/sentence_alignment/run.py --work-id an-nam-chi-luoc --method labse
```

## Cấu trúc thư mục

```
data/
  config.json          ← khai báo tác phẩm
  raw/                 ← file gốc (không sửa)
  ocr_output/          ← text thô sau OCR
  processed/           ← câu đã tách (JSON)
  corpus/              ← corpus song ngữ (TSV)
src/
  ocr_segmentation/    ← Phần 1
  sentence_alignment/  ← Phần 2
  utils.py
configs/pipeline.yaml
```
## Data
```link
https://drive.google.com/drive/folders/1szTHxYqiYGSeg5eutJKjRkRfEayf40Hl?usp=sharing
```
## Cấu trúc thư mục Data
```text
data/
├── config.json          # Khai báo metadata, đường dẫn file và định dạng
├── sino/                # Dữ liệu đầu vào phía Hán / Nôm
│   ├── an-nam-chi-luoc/
│   └── ...
├── vie/                 # Dữ liệu đầu vào phía Tiếng Việt
│   ├── an-nam-chi-luoc/
│   └── ...
├── ocr_output/          # Output OCR thô ([matacpham]_sino_raw.txt / vie_raw.txt)
└── corpus/              # Output dóng hàng ([matacpham]_parallel.tsv / .xlsx)

