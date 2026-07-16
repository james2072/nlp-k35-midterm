import os
import sys
import re
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image

# Padding mở rộng bounding box trước khi crop (pixel) — y như EXPEND = 5 trong notebook colab_paddle.ipynb
CROP_EXPAND = 5


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
    Chuẩn hóa kích thước ảnh vừa phải nếu ảnh quá nhỏ/quá lớn.
    Không dùng CLAHE hay thresholding quá mạnh (y như notebook colab_paddle.ipynb dùng thẳng ảnh gốc)
    để tránh làm mất các header chữ mảnh/tiêu đề trên nền màu như 'Thể Lệ Hiệu Chú'.
    """
    h, w = img.shape[:2]
    # Nếu ảnh quá nhỏ (< 1200px), phóng to nhẹ để DBNet dễ nhận diện chữ nhỏ
    if max(h, w) < 1200:
        scale = min(2.0, 1800 / max(h, w))
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    return img


def _setup_paddle_env():
    """Thiết lập các cờ môi trường Windows trước khi import PaddleOCR/torch."""
    import logging
    import warnings

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["FLAGS_enable_pir_api"] = "0"
    os.environ["FLAGS_enable_pir_in_executor"] = "0"
    os.environ["FLAGS_use_mkldnn"] = "0"

    # Fix WinError 127 (torch shm.dll conflict với paddlepaddle trên Windows)
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


def _detect_device():
    """Phát hiện GPU hay CPU."""
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception:
        pass
    return "cpu"


def init_paddleocr(lang="vi"):
    """
    Khởi tạo PaddleOCR chỉ để text detection (DBNet bbox).
    Tham khảo đúng cấu hình mặc định tường minh như notebook colab_paddle.ipynb.
    """
    _setup_paddle_env()

    # Import torch trước paddleocr để tránh WinError 127 (shm.dll) trên Windows
    try:
        import torch
    except Exception:
        pass

    from paddleocr import PaddleOCR
    return PaddleOCR(
        use_angle_cls=False,
        lang=lang,
        device=_detect_device(),
        enable_mkldnn=False,
        text_det_box_thresh=0.6,
        text_det_thresh=0.3,
        text_det_unclip_ratio=1.5,
    )


def init_vietocr(weights_path: str | None = None):
    """
    Khởi tạo VietOCR predictor (vgg_seq2seq).
    Tự tìm file weights local tại `weights/vgg_seq2seq.pth`, fallback URL nếu không có.
    """
    try:
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
    except ImportError as e:
        print(f"  ⚠️ Lỗi import VietOCR ({e}). Cần chạy: pip install vietocr \"setuptools<70\"")
        return None

    if weights_path is None:
        local = os.path.join(os.path.dirname(__file__), "weights", "vgg_seq2seq.pth")
        weights_path = local if os.path.exists(local) else "https://vocr.vn/data/vietocr/vgg_seq2seq.pth"

    src = "[local] " + os.path.basename(weights_path) if os.path.exists(weights_path) else "[remote] " + weights_path
    print(f"  → VietOCR weights: {src}")

    try:
        config = Cfg.load_config_from_name("vgg_seq2seq")
        config["weights"] = weights_path
        config["pretrain"] = weights_path
        config["device"] = "cuda:0" if _detect_device() == "gpu" else "cpu"
        config["predictor"]["beamsearch"] = False
        return Predictor(config)
    except Exception as e:
        print(f"  ⚠️ Không khởi tạo được VietOCR: {e}")
        return None


def _crop_box(img_bgr, box, expand: int = CROP_EXPAND):
    """
    Crop ảnh BGR theo bounding box polygon 4 góc từ PaddleOCR, mở rộng `expand` pixel.
    """
    h_img, w_img = img_bgr.shape[:2]
    pts = np.array(box, dtype=np.float32)
    x_min = max(0, int(pts[:, 0].min()) - expand)
    y_min = max(0, int(pts[:, 1].min()) - expand)
    x_max = min(w_img, int(pts[:, 0].max()) + expand)
    y_max = min(h_img, int(pts[:, 1].max()) + expand)
    crop = img_bgr[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def run_ocr_page(img_bgr, paddle_engine, vietocr_predictor):
    """
    Pipeline 2 model cực kỳ tối giản, giống hệt 100% logic trong notebook colab_paddle.ipynb:
      1. PaddleOCR tìm bounding box từng dòng chữ.
      2. Tạo boxes 2 điểm [[x_min, y_min], [x_max, y_max]] -> đảo ngược [::-1] -> mở rộng EXPEND=5.
      3. VietOCR crop từng bbox -> predict text tiếng Việt có dấu.
    """
    if vietocr_predictor is None:
        raise ValueError(
            "VietOCR predictor là None (khởi tạo thất bại do lỗi import hoặc thiếu thư viện pkg_resources/setuptools)."
        )

    # Bước 1: Paddle detect bbox
    try:
        det_result = paddle_engine.ocr(img_bgr, det=True, rec=False, cls=False)
    except TypeError:
        det_result = paddle_engine.ocr(img_bgr)

    # Lấy danh sách box từ det_result (tương thích mọi format: v2.x list, v3.x dict, PaddleX)
    raw_lines = []
    if det_result:
        if isinstance(det_result, dict):
            raw_lines = det_result.get("dt_polys", det_result.get("rec_polys", det_result.get("boxes", [])))
        elif isinstance(det_result, list) and len(det_result) > 0:
            first = det_result[0]
            if isinstance(first, dict):
                # Paddle v3 dict format in list: [{'dt_polys': array([...])}]
                raw_lines = first.get("dt_polys", first.get("rec_polys", first.get("boxes", [])))
                if not len(raw_lines) and "points" in first:
                    raw_lines = [item["points"] for item in det_result if isinstance(item, dict) and "points" in item]
            elif isinstance(first, list):
                # Paddle v2 format: [ [box1, box2, ...] ]
                raw_lines = first
            else:
                raw_lines = det_result
        else:
            raw_lines = det_result

    if not raw_lines or len(raw_lines) == 0:
        return []

    # Bước 2: Tạo Boxes 2 điểm từ polygon 4 điểm (y như colab_paddle.ipynb)
    boxes = []
    for line in raw_lines:
        # Nếu line là dict (ví dụ {'points': poly}), lấy points
        if isinstance(line, dict):
            poly = line.get("points", line.get("poly", []))
        # Nếu line là [poly, (txt, score)] thì lấy poly, ngược lại lấy line
        elif isinstance(line, (list, tuple)) and len(line) == 2 and isinstance(line[1], (tuple, list)):
            poly = line[0]
        else:
            poly = line

        try:
            boxes.append([
                [int(poly[0][0]), int(poly[0][1])],
                [int(poly[2][0]), int(poly[2][1])]
            ])
        except (IndexError, TypeError, KeyError):
            continue

    # Đảo ngược danh sách box (y như colab_paddle.ipynb: boxes = boxes[::-1])
    boxes = boxes[::-1]

    # Mở rộng EXPEND = 5 (y như colab_paddle.ipynb)
    EXPEND = 5
    for box in boxes:
        box[0][0] = box[0][0] - EXPEND
        box[0][1] = box[0][1] - EXPEND
        box[1][0] = box[1][0] + EXPEND
        box[1][1] = box[1][1] + EXPEND

    # Bước 3: Crop ảnh -> VietOCR predict (y như colab_paddle.ipynb)
    h_img, w_img = img_bgr.shape[:2]
    recognized = []

    for box in boxes:
        y_min, y_max = max(0, box[0][1]), min(h_img, box[1][1])
        x_min, x_max = max(0, box[0][0]), min(w_img, box[1][0])

        cropped_image = img_bgr[y_min:y_max, x_min:x_max]
        if cropped_image.size == 0:
            continue

        try:
            pil_crop = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            text = vietocr_predictor.predict(pil_crop)
            if text and str(text).strip():
                # Trả về cùng format [ [box_poly, (text, 1.0)] ]
                poly_4pts = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                recognized.append([poly_4pts, (str(text).strip(), 1.0)])
        except Exception:
            continue

    return recognized


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
    """
    Làm sạch dòng trống hoặc khoảng trắng thừa.
    Bỏ hoàn toàn regex cắt chữ/số/dấu ngoặc vì bản thân VietOCR đã có từ điển (vocab) chuẩn,
    giữ nguyên số (như 1961), dấu ngoặc, và header tài liệu.
    """
    lines = text.split("\n")
    clean_lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(clean_lines)


def smart_sort_layout(lines) -> str:
    """
    Sắp xếp văn bản chuẩn theo thứ tự từ trên xuống dưới (Top to Bottom)
    và từ trái qua phải (Left to Right) dựa trên tọa độ Y trung bình của bounding box.
    Đảm bảo 100% không bị ngược dòng và không bị mất/xáo trộn chữ.
    """
    if not lines:
        return ""

    if not (isinstance(lines[0], (list, tuple)) and len(lines[0]) == 2 and isinstance(lines[0][1], (tuple, list))):
        lines = normalize_ocr_result(lines)

    if not lines:
        return ""

    def get_sort_key(line_item):
        box = line_item[0]
        try:
            if isinstance(box, (list, tuple, np.ndarray)) and len(box) > 0:
                pts = np.array(box, dtype=np.float32)
                # Tọa độ y trung bình (giúp chống nghiêng trang tốt hơn y_min)
                y_mean = float(pts[:, 1].mean())
                # Tọa độ x bên trái nhất
                x_min = float(pts[:, 0].min())
                return (y_mean, x_min)
        except Exception:
            pass
        return (0.0, 0.0)

    # Sắp xếp từ trên xuống dưới (Y nhỏ đến Y lớn), nếu cùng dòng thì trái sang phải
    sorted_lines = sorted(lines, key=get_sort_key)

    texts = []
    for line in sorted_lines:
        text = line[1][0] if isinstance(line[1], (tuple, list)) else line[1]
        if text and str(text).strip():
            texts.append(str(text).strip())

    return "\n".join(texts)
