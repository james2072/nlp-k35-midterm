# HVB Project Implementation Plan

This document outlines the design and implementation steps for building the Hán–Việt bilingual historical corpus pipeline.

## User Review Required

> [!IMPORTANT]
> The current `data/config.json` references files and paths that do not match the actual files present in `data/raw/`. We will update `data/config.json` to point to the actual raw materials under `data/raw/sino/` and `data/raw/vie/`.

### Work Mapping

We have mapped the actual files in `data/raw` to three historical works:

1. **An Nam Chí Nguyên** (`an-nam-chi-nguyen`):
   - Hán: `data/raw/sino/an_nam_chi_nguyen_sino.pdf`
   - Việt: `data/raw/vie/an_nam_chi_nguyen.pdf`
2. **Công Dư Tiệp Ký** (`cong-du-tiep-ky`):
   - Hán: `data/raw/sino/cong_du_tiep_ky_sino/` (Directory of page images)
   - Việt: `data/raw/vie/cong_du_tiep_ky_1.pdf`
3. **Đại Việt Lịch Triều Đăng Khoa Lục** (`dai-viet-lich-trieu-dang-khoa-luc`):
   - Hán: `data/raw/sino/dai_viet_lich_trieu_dang_khoa_luc_1_sino.txt`
   - Việt: `data/raw/vie/dai_viet_lich_trieu_dang_khoa_luc_1.pdf`

---

## Proposed Changes

We will implement all TODOs across the `src` directory to complete the pipeline.

### Component 1: Shared Utilities

#### [MODIFY] [utils.py](file:///d:/workspace/nlp-k35-midterm/src/utils.py)
- Implement `generate_id` to format components into `HVB_fff.ccc.ppp.ss`.
- Implement `read_json` and `write_json` with UTF-8 encoding.
- Implement `read_lines` and `write_lines` for clean text file I/O.

---

### Component 2: Part 1 - OCR & Segmentation

#### [MODIFY] [config_loader.py](file:///d:/workspace/nlp-k35-midterm/src/ocr_segmentation/config_loader.py)
- Implement JSON parsing, validate configuration fields, and resolve paths relative to the repository root.

#### [MODIFY] [pdf_to_images.py](file:///d:/workspace/nlp-k35-midterm/src/ocr_segmentation/pdf_to_images.py)
- Implement PDF-to-image conversion using PyMuPDF (`fitz`).
- Parse page ranges (e.g. `"1-80"`, `"1,3,5"`) into 0-indexed page numbers.

#### [MODIFY] [ocr_engine.py](file:///d:/workspace/nlp-k35-midterm/src/ocr_segmentation/ocr_engine.py)
- Initialize PaddleOCR (Hán) and pytesseract (Việt) with caching/lazy-loading.
- Extract embedded text layers when available, or fall back to OCR.
- For `mixed_page`, implement split-crop logic based on the config.

#### [MODIFY] [bbox_sort.py](file:///d:/workspace/nlp-k35-midterm/src/ocr_segmentation/bbox_sort.py)
- Implement sorting of OCR bounding boxes for Hán reading order (columns right to left, top to bottom within columns) using horizontal projection / clustering.

#### [MODIFY] [segmenter.py](file:///d:/workspace/nlp-k35-midterm/src/ocr_segmentation/segmenter.py)
- Tokenize text blocks into sentences: HanLP (Hán) and Underthesea (Việt). Implement rule-based fallbacks to handle environments without pre-trained model downloads.

#### [MODIFY] [run.py (ocr_segmentation)](file:///d:/workspace/nlp-k35-midterm/src/ocr_segmentation/run.py)
- Implement command-line argument parsing and tie all Part 1 steps together.

---

### Component 3: Part 2 - Sentence Alignment & Character Verification

#### [MODIFY] [aligner.py](file:///d:/workspace/nlp-k35-midterm/src/sentence_alignment/aligner.py)
- Implement three alignment methods:
  1. `labse`: Encoding with SentenceTransformers (`LaBSE`) and computing cosine similarity.
  2. `auto_translate`: Translating Hán to Việt using Gemini API, then comparing via TF-IDF cosine similarity.
  3. `greedy`: Matching sentences based on length-ratio and relative positions.

#### [MODIFY] [char_aligner.py](file:///d:/workspace/nlp-k35-midterm/src/sentence_alignment/char_aligner.py)
- Implement Sino-Nom character alignment and confidence verification using Levenshtein distance:
  - $S_1$ (similar shapes from `SinoNom_Similar.dic`)
  - $S_2$ (readings from `QuocNgu_SinoNom.dic`)
  - Character classification: `"ok"` (black), `"corrected"` (green), or `"error"` (red).

#### [MODIFY] [corpus_builder.py](file:///d:/workspace/nlp-k35-midterm/src/sentence_alignment/corpus_builder.py)
- Build the final aligned TSV with structured corpus IDs.
- Handle merging of individual TSVs into `HVB_corpus_all.tsv`.

#### [MODIFY] [run.py (sentence_alignment)](file:///d:/workspace/nlp-k35-midterm/src/sentence_alignment/run.py)
- Connect all Part 2 steps and parse CLI parameters.

---

## Verification Plan

### Automated Tests
- We will execute the pipeline steps using the test commands:
  ```bash
  python src/ocr_segmentation/run.py
  python src/sentence_alignment/run.py
  ```
- Write unit tests for `bbox_sort.py` and `char_aligner.py` to verify logic.

### Manual Verification
- Verify the structure and alignment scores in the output `corpus/HVB_corpus_all.tsv`.
- Inspect character alignment reports for accuracy.
