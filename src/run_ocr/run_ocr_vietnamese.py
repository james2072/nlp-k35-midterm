import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys
import logging
import time
import warnings
import re

warnings.filterwarnings("ignore")
logging.getLogger("ppocr").setLevel(logging.ERROR)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ocr_utils import load_and_process_input, enhance_image, init_paddleocr, normalize_ocr_result
from llm_corrector import correct_text_with_llm

OUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ocr_output")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "config.json")

# Regex giữ lại chữ Latin (full dấu Việt), chữ Hán/Nôm, khoảng trắng
_KEEP_VIET = re.compile(
    r"[^a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"
    r"\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002A6DF\s]"
)


def smart_sort_layout(result) -> str:
    """Tự phát hiện layout dọc (Nôm cổ) hoặc ngang (Quốc ngữ) rồi sắp xếp bbox."""
    lines = normalize_ocr_result(result)
    if not lines:
        return ""
    items = []
    for line in lines:
        box  = line[0]
        text = line[1][0]
        cx   = sum(p[0] for p in box) / 4.0
        cy   = sum(p[1] for p in box) / 4.0
        h    = max(abs(box[0][1] - box[2][1]), abs(box[1][1] - box[3][1])) or 20
        w    = max(abs(box[0][0] - box[1][0]), abs(box[2][0] - box[3][0])) or 20
        items.append({"cx": cx, "cy": cy, "h": h, "w": w, "text": str(text)})

    avg_h = sum(i["h"] for i in items) / len(items)
    avg_w = sum(i["w"] for i in items) / len(items)

    if avg_h > avg_w * 1.2:            # layout dọc
        items.sort(key=lambda x: x["cx"], reverse=True)
        columns: list[list] = []
        for item in items:
            placed = False
            for col in columns:
                if abs(item["cx"] - col[0]["cx"]) < col[0]["w"] * 0.7:
                    col.append(item); placed = True; break
            if not placed: columns.append([item])
        for col in columns: col.sort(key=lambda x: x["cy"])
        ordered = [item for col in columns for item in col]
    else:                              # layout ngang
        items.sort(key=lambda x: x["cy"])
        rows: list[list] = []
        for item in items:
            placed = False
            for row in rows:
                if abs(item["cy"] - row[0]["cy"]) < row[0]["h"] * 0.6:
                    row.append(item); placed = True; break
            if not placed: rows.append([item])
        for row in rows: row.sort(key=lambda x: x["cx"])
        ordered = [item for row in rows for item in row]

    raw_text = "\n".join(i["text"] for i in ordered)
    return _clean_viet_text(raw_text)


def _clean_viet_text(text: str) -> str:
    """Xóa các ký tự rác, giữ chữ Việt/Latin đầy đủ dấu + Hán + khoảng trắng."""
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line or len(line) == 1 and not line.isalnum():
            continue
        clean_line = _KEEP_VIET.sub("", line)
        if clean_line:
            clean_lines.append(clean_line)
    return "\n".join(clean_lines)


def _ocr_scan_pages(pages: list, work_title: str) -> list[str]:
    """Chạy OCR + LLM correction cho danh sách ảnh (pdf_scan)."""
    ocr = init_paddleocr(lang="vi", use_angle_cls=True)
    print("  PaddleOCR (Việt/Latin) đã khởi tạo.")

    result_pages = []
    for idx, img in enumerate(pages):
        t0 = time.time()
        print(f"  → OCR trang {idx + 1}/{len(pages)}...", end=" ", flush=True)
        try:
            enhanced   = enhance_image(img)
            ocr_result = ocr.ocr(enhanced)
            page_text  = smart_sort_layout(ocr_result)
            page_text  = correct_text_with_llm(page_text, work_title, language="vie")
            result_pages.append(page_text)
            print(f"OK ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"LỖI: {e}")
    return result_pages


def run_viet_ocr():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    for work in config["works"]:
        file_path  = work["vie_file"]
        file_type  = work["vie_type"]
        work_id    = work["id"]
        work_title = work["viet"]
        out_path   = os.path.join(OUT_DIR, f"{work_id}_vie_raw.txt")

        print(f"\n{'='*55}")
        print(f"  Việt: {work_title} ({work_id})  [type={file_type}]")
        print(f"{'='*55}")

        if file_type in ("text", "pdf_text"):
            # Đã có text sẵn — đọc thẳng, không OCR, không LLM, không clean
            pages, _ = load_and_process_input(file_path, file_type, work_id)
            full_text = "\n".join(pages)
            print(f"  → Bỏ qua OCR (text có sẵn), {len(full_text):,} ký tự")
        else:
            # pdf_scan — chạy full pipeline OCR
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