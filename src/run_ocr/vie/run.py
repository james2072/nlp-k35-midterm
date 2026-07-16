import os
import json
import sys
import logging
import time
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("ppocr").setLevel(logging.ERROR)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ocr_utils import (
    load_and_process_input,
    enhance_image,
    init_paddleocr,
    init_vietocr,
    run_ocr_page,
    smart_sort_layout,
    clean_viet_text,
)
from llm_corrector import correct_text_with_llm

OUT_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "ocr_output"))
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "config.json"))


def _init_engines():
    """
    Khởi tạo PaddleOCR (detection-only) + VietOCR (recognition).
    Nếu VietOCR không khả dụng, trả về paddle full pipeline + None.
    """
    print("  ⏳ Khởi tạo PaddleOCR (detection)...", end=" ", flush=True)
    paddle = init_paddleocr(lang="vi")
    print("OK")

    print("  ⏳ Khởi tạo VietOCR (vgg_seq2seq recognition)...", end=" ", flush=True)
    vietocr = init_vietocr()
    print("OK")

    return paddle, vietocr


def _ocr_scan_pages(pages: list, work_title: str) -> list[str]:
    """
    Chạy pipeline 2 model (PaddleOCR detect + VietOCR recognize) cho danh sách ảnh.
    Sau đó gửi từng trang qua LLM corrector để sửa lỗi OCR còn sót.
    """
    paddle_engine, vietocr_predictor = _init_engines()

    result_pages = []
    for idx, img in enumerate(pages):
        t0 = time.time()
        print(f"  → OCR trang {idx + 1}/{len(pages)}...", end=" ", flush=True)
        try:
            enhanced  = enhance_image(img)
            ocr_lines = run_ocr_page(enhanced, paddle_engine, vietocr_predictor)
            page_text = smart_sort_layout(ocr_lines)
            page_text = correct_text_with_llm(page_text, work_title, language="vie")
            result_pages.append(page_text)
            print(f"OK ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"LỖI: {e}")
    return result_pages


def run_viet_ocr():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    for work in config.get("works", []):
        if "vie_file" not in work or "vie_type" not in work:
            continue

        file_path  = work["vie_file"]
        file_type  = work["vie_type"]
        work_id    = work["id"]
        work_title = work.get("viet", work_id)
        out_path   = os.path.join(OUT_DIR, f"{work_id}_vie_raw.txt")

        print(f"\n{'='*55}")
        print(f"  Việt: {work_title} ({work_id})  [type={file_type}]")
        print(f"{'='*55}")

        if file_type in ("text", "pdf_text"):
            # Đã có text layer sẵn — đọc thẳng, không cần OCR
            pages, _ = load_and_process_input(file_path, file_type, work_id)
            full_text = "\n".join(pages)
            print(f"  → Bỏ qua OCR (text có sẵn), {len(full_text):,} ký tự")
        else:
            # pdf_scan — chạy pipeline Paddle (detect) + VietOCR (recognize) + LLM Corrector
            pages, data_type = load_and_process_input(file_path, file_type, work_id)
            if data_type != "image" or not pages:
                print("  → Không load được ảnh, bỏ qua.")
                continue
            result_pages = _ocr_scan_pages(pages, work_title)
            full_text    = clean_viet_text("\n".join(result_pages))

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"  ✓ Đã lưu: {out_path}  ({len(full_text):,} ký tự)")


if __name__ == "__main__":
    run_viet_ocr()
