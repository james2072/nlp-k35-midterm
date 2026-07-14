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

from ocr_utils import load_and_process_input, enhance_image
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
    if not result or not result[0]:
        return ""
    items = []
    for line in result[0]:
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
    else:                               # layout ngang
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


def clean_viet_text(raw_text: str) -> str:
    """Lọc ký tự rác OCR, giữ lại chữ Việt và Hán/Nôm."""
    clean_lines = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^[\d\s\W_]+$", line):
            continue
        clean_line = re.sub(r"\s+", " ", _KEEP_VIET.sub("", line)).strip()
        if len(clean_line) >= 4:        # bỏ fragment quá ngắn
            clean_lines.append(clean_line)
    return "\n".join(clean_lines)


def _ocr_scan_pages(pages: list, work_title: str) -> list[str]:
    """Chạy OCR + LLM correction cho danh sách ảnh (pdf_scan)."""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=True, lang="en", show_log=False,
        det_db_box_thresh=0.3, det_db_thresh=0.2, drop_score=0.1,
    )
    print("  PaddleOCR (Việt/Latin) đã khởi tạo.")

    result_pages = []
    for idx, img in enumerate(pages):
        t0 = time.time()
        print(f"  → OCR trang {idx + 1}/{len(pages)}...", end=" ", flush=True)
        try:
            enhanced   = enhance_image(img)
            ocr_result = ocr.ocr(enhanced, cls=True)
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