import os
import sys
import re
import cv2
import numpy as np
import fitz  # PyMuPDF

# Regex giữ lại chữ Latin (full dấu Việt), chữ Hán/Nôm, khoảng trắng
_KEEP_VIET = re.compile(
    r"[^a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"
    r"\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002A6DF\s]"
)


def get_project_root():
    """Trả về thư mục gốc của project (nlp-k35-midterm)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def find_file(file_path, work_id):
    root = get_project_root()
    abs_path = os.path.normpath(os.path.join(root, file_path))
    if os.path.exists(abs_path):
        return abs_path

    dir_name = os.path.dirname(abs_path)
    if os.path.exists(dir_name):
        search_id = work_id.upper()
        for f in os.listdir(dir_name):
            if search_id in f.upper():
                return os.path.normpath(os.path.join(dir_name, f))
        files_in_dir = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
        if len(files_in_dir) == 1:
            return os.path.normpath(os.path.join(dir_name, files_in_dir[0]))
    return None


def load_and_process_input(file_path, file_type, work_id):
    abs_path = find_file(file_path, work_id)
    if not abs_path or not os.path.exists(abs_path):
        print(f"Không tìm thấy file: {file_path}")
        return [], "unknown"
    print(f"  -> Đang đọc: {os.path.basename(abs_path)}")

    if file_type == "text":
        with open(abs_path, 'r', encoding='utf-8') as f:
            return [f.read()], "text"

    elif file_type in ["pdf_text", "pdf_scan", "image"]:
        ext = os.path.splitext(abs_path)[1].lower()
        if ext == ".pdf":
            doc = fitz.open(abs_path)
            if file_type == "pdf_text":
                text_pages = []
                for page in doc:
                    text_pages.append(page.get_text("text"))
                doc.close()
                return text_pages, "text"

            images = []
            for page in doc:
                mat = fitz.Matrix(250 / 72, 250 / 72)  # Render ~250 DPI
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                images.append(img)
            doc.close()
            return images, "image"
        else:
            img = cv2.imread(abs_path)
            if img is not None:
                return [img], "image"
    return [], "unknown"


def enhance_image(img):
    """
    Pipeline làm sạch và tăng độ tương phản an toàn cho tài liệu chữ Việt:
    - CLAHE + Median Blur + Unsharp Masking để chữ nổi bật, viền sắc nét cho DBNet phân vùng chuẩn xác.
    """
    h, w = img.shape[:2]
    if max(h, w) < 2000:
        scale = min(2.0, 2500 / max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.medianBlur(enhanced, 3)

    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

    max_side = 3500
    h_new, w_new = sharpened.shape[:2]
    if max(h_new, w_new) > max_side:
        scale = max_side / max(h_new, w_new)
        sharpened = cv2.resize(sharpened, (int(w_new * scale), int(h_new * scale)), interpolation=cv2.INTER_AREA)

    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def init_paddleocr(lang="vi", use_angle_cls=True):
    """
    Khởi tạo engine PaddleOCR tối ưu cho tiếng Việt:
    - Tự động thiết lập các cờ môi trường (KMP, PIR, MKLDNN) chống lỗi Windows CPU.
    - Xử lý đường dẫn DLL cho torch/shm.dll để tránh WinError 127 trên Windows.
    """
    import logging
    import warnings

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["FLAGS_enable_pir_api"] = "0"
    os.environ["FLAGS_enable_pir_in_executor"] = "0"
    os.environ["FLAGS_use_mkldnn"] = "0"

    try:
        torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
        if os.path.exists(torch_lib):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(torch_lib)
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    warnings.filterwarnings("ignore")
    logging.getLogger("ppocr").setLevel(logging.ERROR)

    device = "cpu"
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            device = "gpu"
    except Exception:
        device = "cpu"

    try:
        import torch
    except Exception:
        pass

    from paddleocr import PaddleOCR
    ocr_engine = PaddleOCR(
        use_angle_cls=use_angle_cls,
        lang=lang,
        device=device,
        enable_mkldnn=False,
        text_det_box_thresh=0.6,
        text_det_thresh=0.35,
        text_det_unclip_ratio=1.2,
        text_rec_score_thresh=0.1,
    )
    return ocr_engine


def normalize_ocr_result(result):
    """Chuẩn hóa kết quả PaddleOCR v3.x/v2.x về định dạng list tuples [ [box, (text, score)], ... ]."""
    if not result or len(result) == 0:
        return []

    if isinstance(result, dict) and "rec_texts" in result:
        boxes = result.get("dt_polys", result.get("rec_polys", []))
        texts = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])
        return [[box, (txt, score)] for box, txt, score in zip(boxes, texts, scores)]

    if isinstance(result, list):
        first = result[0]
        if first is None:
            return []
        if isinstance(first, dict) and "rec_texts" in first:
            boxes = first.get("dt_polys", first.get("rec_polys", []))
            texts = first.get("rec_texts", [])
            scores = first.get("rec_scores", [])
            return [[box, (txt, score)] for box, txt, score in zip(boxes, texts, scores)]

        if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[1], (tuple, list)) and len(first[1]) == 2 and isinstance(first[1][0], (str, np.str_)):
            return result

        if isinstance(first, list):
            return first

    return []


def clean_viet_text(text: str) -> str:
    """Xóa các ký tự rác, giữ chữ Việt/Latin đầy đủ dấu + khoảng trắng."""
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


def smart_sort_layout(result) -> str:
    """Sắp xếp bounding box theo hàng ngang từ trên xuống dưới (trái qua phải)."""
    lines = normalize_ocr_result(result)
    if not lines:
        return ""
    items = []
    for line in lines:
        box = line[0]
        text = line[1][0]
        cx = sum(p[0] for p in box) / 4.0
        cy = sum(p[1] for p in box) / 4.0
        h = max(abs(box[0][1] - box[2][1]), abs(box[1][1] - box[3][1])) or 20
        w = max(abs(box[0][0] - box[1][0]), abs(box[2][0] - box[3][0])) or 20
        items.append({"cx": cx, "cy": cy, "h": h, "w": w, "text": str(text)})

    items.sort(key=lambda x: x["cy"])
    rows: list[list] = []
    for item in items:
        placed = False
        for row in rows:
            if abs(item["cy"] - row[0]["cy"]) < row[0]["h"] * 0.6:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])
    for row in rows:
        row.sort(key=lambda x: x["cx"])
    ordered = [item for row in rows for item in row]

    raw_text = "\n".join(i["text"] for i in ordered)
    return clean_viet_text(raw_text)
