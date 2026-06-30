"""
OCR dispatcher: routes to PaddleOCR (Han), pytesseract (Viet),
text-layer extraction, or direct read depending on lang and format.
"""


import os
import yaml
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

_paddle_ocr_instance = None

def get_paddle_ocr():
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        from paddleocr import PaddleOCR
        # Disable logging for cleaner output
        _paddle_ocr_instance = PaddleOCR(lang="ch", use_angle_cls=True, use_gpu=False, show_log=False)
    return _paddle_ocr_instance


def get_pipeline_config() -> dict:
    config_path = os.path.join(os.getcwd(), "configs", "pipeline.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}


def run(
    image_paths: list[str],
    lang: str,
    fmt: str,
    pdf_path: str | None = None,
    page_indices: list[int] | None = None,
) -> tuple[list[str], list[dict] | None]:
    """
    Run OCR on a list of images or return text directly.

    Args:
        image_paths:  List of image file paths (or empty for text format).
        lang:         "han" or "viet".
        fmt:          "pdf" | "images" | "text" | "mixed_page".
        pdf_path:     Optional path to the source PDF.
        page_indices: Optional list of 0-indexed page numbers.

    Returns:
        Tuple: (list of raw text lines, list of bbox dicts or None)
    """
    # If format is direct text, we shouldn't even have image paths.
    # But if we do, this is handled. (Direct text loading is typically handled by the runner).
    if fmt == "text":
        return [], None

    pipeline_cfg = get_pipeline_config()

    if fmt == "mixed_page":
        strategy = pipeline_cfg.get("ocr", {}).get("mixed_page_split", "top_bottom")
        mixed_data = _ocr_mixed(image_paths, strategy)
        return mixed_data.get(lang, ([], None))

    if lang == "han":
        return _ocr_paddle_han(image_paths)

    if lang == "viet":
        # Try extracting text layer first if PDF path is available
        if pdf_path and page_indices:
            lines = _extract_text_layer(pdf_path, page_indices)
            if lines:
                return lines, None
        return _ocr_tesseract_viet(image_paths)

    return [], None


def _ocr_paddle_han(image_paths: list[str]) -> tuple[list[str], list[dict]]:
    ocr = get_paddle_ocr()
    all_lines = []
    all_bboxes = []

    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
        try:
            # ocr.ocr returns a list of pages. For one image it's a list with one item.
            res = ocr.ocr(img_path, cls=True)
            if not res or res[0] is None:
                continue
                
            for detection in res[0]:
                bbox_coords = detection[0]
                text_info = detection[1]
                
                text = text_info[0]
                # Calculate bounding box bounding rect: x, y, w, h
                xs = [pt[0] for pt in bbox_coords]
                ys = [pt[1] for pt in bbox_coords]
                
                x = min(xs)
                y = min(ys)
                w = max(xs) - x
                h = max(ys) - y
                
                all_lines.append(text)
                all_bboxes.append({"x": x, "y": y, "w": w, "h": h})
        except Exception as e:
            print(f"Error running PaddleOCR on {img_path}: {e}")

    return all_lines, all_bboxes


def _ocr_tesseract_viet(image_paths: list[str]) -> tuple[list[str], list[dict] | None]:
    # Configure tesseract path on Windows if not set
    if os.name == "nt" and not getattr(pytesseract.pytesseract, "tesseract_cmd", None):
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break

    all_lines = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img, lang="vie")
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            all_lines.extend(lines)
        except Exception as e:
            print(f"Error running Tesseract OCR on {img_path}: {e}")

    return all_lines, None


def _ocr_mixed(image_paths: list[str], strategy: str) -> dict[str, tuple[list[str], list[dict] | None]]:
    # Crop each image into Han crop and Viet crop
    han_crops = []
    viet_crops = []
    
    temp_dir = os.path.join(os.getcwd(), "data", "temp_images")
    os.makedirs(temp_dir, exist_ok=True)

    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path)
            w, h = img.size
            base = os.path.splitext(os.path.basename(img_path))[0]
            
            han_path = os.path.join(temp_dir, f"{base}_han_crop.jpg")
            viet_path = os.path.join(temp_dir, f"{base}_viet_crop.jpg")
            
            if strategy == "top_bottom":
                # Han on top half, Viet on bottom half
                han_img = img.crop((0, 0, w, h // 2))
                viet_img = img.crop((0, h // 2, w, h))
            else:
                # Han on left half, Viet on right half
                han_img = img.crop((0, 0, w // 2, h))
                viet_img = img.crop((w // 2, 0, w, h))
                
            han_img.save(han_path)
            viet_img.save(viet_path)
            
            han_crops.append(han_path)
            viet_crops.append(viet_path)
        except Exception as e:
            print(f"Error cropping mixed page {img_path}: {e}")

    # Perform OCR on the cropped regions
    han_lines, han_bboxes = _ocr_paddle_han(han_crops)
    viet_lines, viet_bboxes = _ocr_tesseract_viet(viet_crops)

    return {
        "han": (han_lines, han_bboxes),
        "viet": (viet_lines, viet_bboxes),
    }


def _extract_text_layer(pdf_path: str, pages: list[int]) -> list[str]:
    try:
        doc = fitz.open(pdf_path)
        all_lines = []
        for idx in pages:
            if 0 <= idx < len(doc):
                text = doc[idx].get_text()
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                all_lines.extend(lines)
        doc.close()
        return all_lines
    except Exception as e:
        print(f"Error extracting text layer from {pdf_path}: {e}")
        return []

