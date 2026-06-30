# Walkthrough: HVB Bilingual Corpus Implementation

We have successfully implemented the complete pipeline for the Hán–Việt bilingual historical corpus.

## Changes Made

### 1. Shared Utilities
- **[utils.py](file:///d:/workspace/nlp-k35-midterm/src/utils.py)**: Implemented HVB ID generator (`HVB_fff.ccc.ppp.ss`), JSON I/O, and raw text lines file helpers.

### 2. Part 1 - OCR & Segmentation
- **[config_loader.py](file:///d:/workspace/nlp-k35-midterm/src/ocr_segmentation/config_loader.py)**: Added parser and validator for `data/config.json` with absolute path resolving.
- **[pdf_to_images.py](file:///d:/workspace/nlp-k35-midterm/src/ocr_segmentation/pdf_to_images.py)**: Implemented page rendering using PyMuPDF and page range parser.
- **[ocr_engine.py](file:///d:/workspace/nlp-k35-midterm/src/ocr_segmentation/ocr_engine.py)**: Implemented OCR dispatcher routing to PaddleOCR (Hán) and pytesseract (Việt) with text layer extraction fallback, caching, and mixed-page splitting.
- **[bbox_sort.py](file:///d:/workspace/nlp-k35-midterm/src/ocr_segmentation/bbox_sort.py)**: Implemented column clustering and sorting for vertical right-to-left, top-to-bottom Hán text.
- **[segmenter.py](file:///d:/workspace/nlp-k35-midterm/src/ocr_segmentation/segmenter.py)**: Implemented sentence tokenizers using HanLP and Underthesea, with robust regex fallbacks.
- **[run.py](file:///d:/workspace/nlp-k35-midterm/src/ocr_segmentation/run.py)**: Tied Part 1 components into a CLI script supporting selective execution.

### 3. Part 2 - Sentence Alignment & Character Verification
- **[aligner.py](file:///d:/workspace/nlp-k35-midterm/src/sentence_alignment/aligner.py)**: Implemented three sentence alignment algorithms (`labse`, `auto_translate` via Gemini, and `greedy` ratio fallback).
- **[char_aligner.py](file:///d:/workspace/nlp-k35-midterm/src/sentence_alignment/char_aligner.py)**: Implemented character-level Levenshtein/MED verification and visual character correction using dictionary intersections.
- **[corpus_builder.py](file:///d:/workspace/nlp-k35-midterm/src/sentence_alignment/corpus_builder.py)**: Implemented corpus TSV builders and merging utility.
- **[run.py](file:///d:/workspace/nlp-k35-midterm/src/sentence_alignment/run.py)**: Integrated alignment and character verification steps into a configurable CLI script.

### 4. Configuration & Dictionaries
- **[config.json](file:///d:/workspace/nlp-k35-midterm/data/config.json)**: Updated configuration paths to reflect actual raw files available in `data/raw/`.
- **[SinoNom_Similar.dic](file:///d:/workspace/nlp-k35-midterm/dictionaries/SinoNom_Similar.dic)**, **[QuocNgu_SinoNom.dic](file:///d:/workspace/nlp-k35-midterm/dictionaries/QuocNgu_SinoNom.dic)**, **[HanViet.dic](file:///d:/workspace/nlp-k35-midterm/dictionaries/HanViet.dic)**: Populated dictionaries with corresponding characters, readings, and visual similarity maps.

### 5. Unit Tests
- **[test_utils.py](file:///d:/workspace/nlp-k35-midterm/tests/test_utils.py)**: Tests for unique ID generation, JSON reader/writer, and line reader/writer.
- **[test_ocr_segmentation.py](file:///d:/workspace/nlp-k35-midterm/tests/test_ocr_segmentation.py)**: Tests for config parsing/validation, page range parsing, and bounding box reading order sorting.
- **[test_sentence_alignment.py](file:///d:/workspace/nlp-k35-midterm/tests/test_sentence_alignment.py)**: Tests for greedy sentence alignment, character identification, visual shape correction, and TSV corpus building/merging.

