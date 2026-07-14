"""
test_ocr_5pages.py — Chạy thử OCR pipeline trên 5 trang đầu của một tác phẩm.

Usage:
    python test_ocr_5pages.py --work-id HVB_002 --lang sino
    python test_ocr_5pages.py --work-id HVB_003 --lang vie
    python test_ocr_5pages.py --work-id HVB_003 --lang vie --pages 10

Output: data/ocr_output/{work_id}_{lang}_test.txt  (same format as production)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import time
import logging
import warnings
import argparse

warnings.filterwarnings("ignore")
logging.getLogger("ppocr").setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "run_ocr"))

import fitz
import cv2
import numpy as np

from ocr_utils import load_and_process_input, enhance_image
from llm_corrector import correct_text_with_llm


# ── helpers ────────────────────────────────────────────────────────────────────

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "data", "config.json")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def find_work(config: dict, work_id: str) -> dict:
    for work in config["works"]:
        if work["id"].upper() == work_id.upper():
            return work
    ids = [w["id"] for w in config["works"]]
    raise ValueError(f"work_id '{work_id}' not found. Available: {ids}")


def load_first_n_pages(file_path: str, file_type: str, work_id: str, n: int):
    """Load tối đa n trang từ file, trả về (pages, data_type)."""
    root = os.path.dirname(__file__)
    abs_path = os.path.normpath(os.path.join(root, file_path))

    if file_type == "text":
        with open(abs_path, encoding="utf-8") as f:
            return [f.read()], "text"

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File không tồn tại: {abs_path}")

    doc = fitz.open(abs_path)
    total = len(doc)
    limit = min(n, total)
    print(f"  → File có {total} trang, lấy {limit} trang đầu")

    if file_type == "pdf_text":
        pages = [doc[i].get_text("text") for i in range(limit)]
        doc.close()
        return pages, "text"

    # pdf_scan → render to images
    images = []
    for i in range(limit):
        mat = fitz.Matrix(250 / 72, 250 / 72)
        pix = doc[i].get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
        images.append(img)
    doc.close()
    return images, "image"


# ── layout sorters (copied from run_ocr_chinese / vietnamese) ─────────────────

def sort_vertical_layout(result) -> str:
    """Hán: cột dọc phải → trái."""
    if not result or not result[0]:
        return ""
    items = []
    for line in result[0]:
        box = line[0]
        text = line[1][0]
        cx = sum(p[0] for p in box) / 4.0
        cy = sum(p[1] for p in box) / 4.0
        h  = max(abs(box[0][1] - box[2][1]), abs(box[1][1] - box[3][1])) or 20
        w  = max(abs(box[0][0] - box[1][0]), abs(box[2][0] - box[3][0])) or 20
        items.append({"cx": cx, "cy": cy, "h": h, "w": w, "text": str(text)})

    items.sort(key=lambda x: x["cx"], reverse=True)
    columns: list[list] = []
    for item in items:
        for col in columns:
            if abs(item["cx"] - sum(i["cx"] for i in col) / len(col)) < item["w"] * 1.0:
                col.append(item)
                break
        else:
            columns.append([item])

    return "\n".join(
        "".join(i["text"] for i in sorted(col, key=lambda x: x["cy"]))
        for col in columns
    )


def smart_sort_layout(result) -> str:
    """Việt: tự phát hiện layout dọc/ngang."""
    if not result or not result[0]:
        return ""
    items = []
    for line in result[0]:
        box = line[0]
        text = line[1][0]
        cx = sum(p[0] for p in box) / 4.0
        cy = sum(p[1] for p in box) / 4.0
        h  = max(abs(box[0][1] - box[2][1]), abs(box[1][1] - box[3][1])) or 20
        w  = max(abs(box[0][0] - box[1][0]), abs(box[2][0] - box[3][0])) or 20
        items.append({"cx": cx, "cy": cy, "h": h, "w": w, "text": str(text)})

    avg_h = sum(i["h"] for i in items) / len(items)
    avg_w = sum(i["w"] for i in items) / len(items)

    if avg_h > avg_w * 1.2:                        # layout dọc
        items.sort(key=lambda x: x["cx"], reverse=True)
        columns: list[list] = []
        for item in items:
            for col in columns:
                if abs(item["cx"] - sum(i["cx"] for i in col) / len(col)) < item["w"] * 1.5:
                    col.append(item)
                    break
            else:
                columns.append([item])
        return "\n".join(
            "".join(i["text"] for i in sorted(col, key=lambda x: x["cy"]))
            for col in columns
        )
    else:                                           # layout ngang
        items.sort(key=lambda x: (x["cy"], x["cx"]))
        lines, current, last_cy = [], [], -1000
        for item in items:
            if abs(item["cy"] - last_cy) < item["h"] * 0.5:
                current.append(item)
            else:
                if current:
                    lines.append(" ".join(i["text"] for i in sorted(current, key=lambda x: x["cx"])))
                current, last_cy = [item], item["cy"]
        if current:
            lines.append(" ".join(i["text"] for i in sorted(current, key=lambda x: x["cx"])))
        return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test OCR pipeline trên N trang đầu")
    parser.add_argument("--work-id", required=True, help="VD: HVB_002")
    parser.add_argument("--lang",    required=True, choices=["sino", "vie"], help="sino hoặc vie")
    parser.add_argument("--pages",   type=int, default=5, help="Số trang đầu cần test (default: 5)")
    parser.add_argument("--no-llm",  action="store_true", help="Bỏ qua bước LLM correction")
    args = parser.parse_args()

    config   = load_config()
    work     = find_work(config, args.work_id)
    lang     = args.lang
    file_key = f"{lang}_file"
    type_key = f"{lang}_type"
    file_path  = work[file_key]
    file_type  = work[type_key]
    work_title = work["viet"]
    work_id    = work["id"]

    print(f"\n{'='*55}")
    print(f"  TEST OCR — {work_title} ({work_id}) [{lang.upper()}]")
    print(f"  File  : {os.path.basename(file_path)}")
    print(f"  Type  : {file_type}  |  Pages: {args.pages}  |  LLM: {'OFF' if args.no_llm else 'ON'}")
    print(f"{'='*55}\n")

    # Load pages
    pages, data_type = load_first_n_pages(file_path, file_type, work_id, args.pages)

    # Init OCR engine
    from paddleocr import PaddleOCR
    paddle_lang = "ch" if lang == "sino" else "en"
    ocr = PaddleOCR(
        use_angle_cls=True, lang=paddle_lang, show_log=False,
        det_db_box_thresh=0.3, det_db_thresh=0.2, drop_score=0.1,
    )
    sort_fn = sort_vertical_layout if lang == "sino" else smart_sort_layout
    print("PaddleOCR khởi tạo xong.\n")

    # Process pages
    result_pages = []
    if data_type == "text":
        result_pages = pages
        print(f"  → Đọc text trực tiếp ({len(pages)} block)\n")
    else:
        for idx, img in enumerate(pages):
            t0 = time.time()
            print(f"  [Trang {idx+1}/{len(pages)}] Đang xử lý...", end=" ", flush=True)
            enhanced = enhance_image(img)
            ocr_result = ocr.ocr(enhanced, cls=True)
            page_text  = sort_fn(ocr_result)

            if not args.no_llm and page_text.strip():
                page_text = correct_text_with_llm(page_text, work_title, language=lang)

            result_pages.append(page_text)
            print(f"OK ({time.time()-t0:.1f}s)")

    # Save output
    out_dir  = os.path.join(os.path.dirname(__file__), "data", "ocr_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{work_id}_{lang}_test.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_pages))

    print(f"\n✓ Đã lưu: {out_path}")
    print(f"  Tổng ký tự: {sum(len(p) for p in result_pages):,}")


if __name__ == "__main__":
    main()
