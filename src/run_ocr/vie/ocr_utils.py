import os
import sys
import re
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image

# Padding mở rộng bounding box trước khi crop (pixel) — 300 DPI + expand 8 theo khuyến nghị
CROP_EXPAND = 8


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
                mat = fitz.Matrix(300 / 72, 300 / 72)  # Render 300 DPI cho chữ tiếng Việt sắc nét
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


def _setup_paddle_env():
    """Thiết lập các cờ môi trường Windows trước khi import PaddleOCR/torch."""
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
    """Khởi tạo PaddleOCR chỉ để text detection (DBNet bbox)."""
    _setup_paddle_env()

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
        text_det_thresh=0.4,
        text_det_unclip_ratio=1.2,
    )


def init_vietocr(weights_path: str | None = None, model_name: str = "vgg_transformer"):
    """
    Khởi tạo VietOCR predictor. Mặc định sử dụng model `vgg_transformer` (ít bị lặp từ hơn seq2seq).
    Tự tìm file weights local tại `weights/<model_name>.pth`, fallback nếu không có.
    """
    try:
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
    except ImportError as e:
        print(f"  ⚠️ Lỗi import VietOCR ({e}). Cần chạy: pip install vietocr \"setuptools<70\"")
        return None

    if weights_path is None:
        local_trans = os.path.join(os.path.dirname(__file__), "weights", f"{model_name}.pth")
        if os.path.exists(local_trans):
            weights_path = local_trans
        else:
            local_seq = os.path.join(os.path.dirname(__file__), "weights", "vgg_seq2seq.pth")
            if os.path.exists(local_seq) and model_name == "vgg_seq2seq":
                weights_path = local_seq

    try:
        config = Cfg.load_config_from_name(model_name)
        if weights_path is not None:
            config["weights"] = weights_path
            config["pretrain"] = weights_path
        config["device"] = "cuda:0" if _detect_device() == "gpu" else "cpu"
        if "predictor" in config and "beamsearch" in config["predictor"]:
            config["predictor"]["beamsearch"] = False
        print(f"  → Khởi tạo VietOCR ({model_name}): {weights_path or 'remote default'}")
        return Predictor(config)
    except Exception as e:
        print(f"  ⚠️ Không khởi tạo được VietOCR: {e}")
        return None


def _extract_raw_polygons(det_result):
    """Lấy danh sách polygon thô từ kết quả PaddleOCR (det=True, rec=False)."""
    raw_lines = []
    if not det_result:
        return raw_lines

    if isinstance(det_result, dict):
        raw_lines = det_result.get("dt_polys", det_result.get("rec_polys", det_result.get("boxes", [])))
    elif isinstance(det_result, list) and len(det_result) > 0:
        first = det_result[0]
        if isinstance(first, dict):
            raw_lines = first.get("dt_polys", first.get("rec_polys", first.get("boxes", [])))
            if not len(raw_lines) and "points" in first:
                raw_lines = [item["points"] for item in det_result
                             if isinstance(item, dict) and "points" in item]
        elif isinstance(first, list):
            raw_lines = first
        else:
            raw_lines = det_result
    else:
        raw_lines = det_result
    return raw_lines


def _is_duplicate_or_contained(box1, box2, iou_thresh=0.9):
    """
    Kiểm tra 2 box polygon có trùng lặp (IoU > thresh) hoặc box này chứa trong box kia
    (contained ratio > thresh) hay không để loại bỏ detection trùng của DBNet.
    """
    x1_min, y1_min = float(box1[:, 0].min()), float(box1[:, 1].min())
    x1_max, y1_max = float(box1[:, 0].max()), float(box1[:, 1].max())
    x2_min, y2_min = float(box2[:, 0].min()), float(box2[:, 1].min())
    x2_max, y2_max = float(box2[:, 0].max()), float(box2[:, 1].max())

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return False

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = max(1e-5, (x1_max - x1_min) * (y1_max - y1_min))
    area2 = max(1e-5, (x2_max - x2_min) * (y2_max - y2_min))

    iou = inter_area / (area1 + area2 - inter_area)
    contained = inter_area / min(area1, area2)
    return iou > iou_thresh or contained > iou_thresh


def _normalize_polygons(raw_lines, min_w: float = 12.0, min_h: float = 8.0, max_h_ratio: float = 3.0, iou_thresh: float = 0.9):
    """
    Chuẩn hoá raw_lines về list polygon 4 điểm (float32 numpy array).
    Lọc bỏ:
      - Polygon quá nhỏ (nhiễu / ký tự lạc đơn lẻ): w < min_w hoặc h < min_h
      - Polygon quá cao so với trung bình (box bị merge nhiều dòng): h > avg_h * max_h_ratio
      - Polygon trùng lặp hoặc chứa bên trong box khác (duplicate / contained boxes)
    """
    candidates = []
    for line in raw_lines:
        if isinstance(line, dict):
            poly = line.get("points", line.get("poly", []))
        elif (isinstance(line, (list, tuple)) and len(line) == 2
              and isinstance(line[1], (tuple, list))):
            poly = line[0]
        else:
            poly = line

        try:
            pts = np.array(poly, dtype="float32")
            if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 2:
                continue
            w = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
            h = max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))
            if w < min_w or h < min_h:
                continue
            candidates.append((pts, h))
        except (IndexError, TypeError, KeyError, ValueError):
            continue

    if not candidates:
        return []

    avg_h = float(np.mean([h for _, h in candidates]))
    max_h = avg_h * max_h_ratio

    # Lọc theo chiều cao tối đa (box bị merge quá nhiều dòng)
    height_filtered = [(pts, h) for pts, h in candidates if h <= max_h]

    # Loại box trùng lặp hoặc chứa bên trong box khác (IoU/contained > 0.9)
    deduped = []
    for pts, h in height_filtered:
        is_dup = False
        for exist_pts, _ in deduped:
            if _is_duplicate_or_contained(pts, exist_pts, iou_thresh=iou_thresh):
                is_dup = True
                break
        if not is_dup:
            deduped.append((pts, h))

    return [pts for pts, _ in deduped]


def _sort_polygons_reading_order(polygons):
    """
    Gom polygon thành các hàng dựa trên y_center (dung sai = 60% chiều cao trung bình),
    rồi trong mỗi hàng sort trái -> phải.
    """
    if not polygons:
        return []

    heights = [float(np.max(p[:, 1]) - np.min(p[:, 1])) for p in polygons]
    avg_h = float(np.mean(heights)) if heights else 20.0
    row_thresh = avg_h * 0.6

    items = sorted(
        [(p, float(np.mean(p[:, 1]))) for p in polygons],
        key=lambda t: t[1]
    )

    rows: list[dict] = []
    for poly, yc in items:
        placed = False
        for row in rows:
            if abs(row["y"] - yc) <= row_thresh:
                row["items"].append(poly)
                row["y"] = (row["y"] * row["count"] + yc) / (row["count"] + 1)
                row["count"] += 1
                placed = True
                break
        if not placed:
            rows.append({"y": yc, "count": 1, "items": [poly]})

    rows.sort(key=lambda r: r["y"])
    sorted_polygons = []
    for row in rows:
        row["items"].sort(key=lambda p: float(np.min(p[:, 0])))
        sorted_polygons.extend(row["items"])
    return sorted_polygons


def _crop_box_from_poly(img_bgr, pts, expand: int = CROP_EXPAND):
    """
    Crop ảnh theo axis-aligned bounding rectangle của polygon 4 điểm + padding bất đối xứng:
      - Chiều ngang (expand_x): đủ rộng (expand px) để bao trọn chữ đầu/cuối dòng.
      - Chiều dọc (expand_y): vừa phải (max 3-4 px) để không cắt lẹm vào dòng trên/dưới.
    """
    h_img, w_img = img_bgr.shape[:2]
    h_box = float(pts[:, 1].max() - pts[:, 1].min())
    expand_x = expand
    expand_y = min(4, max(2, int(h_box * 0.12)))

    x_min = max(0,       int(pts[:, 0].min()) - expand_x)
    y_min = max(0,       int(pts[:, 1].min()) - expand_y)
    x_max = min(w_img,   int(pts[:, 0].max()) + expand_x)
    y_max = min(h_img,   int(pts[:, 1].max()) + expand_y)
    crop = img_bgr[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None, None
    poly_4pts = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    return crop, poly_4pts


def draw_polygons(img_bgr, polys, color=(0, 255, 0), thickness=2):
    """Vẽ các polygon lên bản sao của ảnh để kiểm tra bounding-box alignment."""
    img = img_bgr.copy()
    for poly in polys:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    return img


def run_ocr_page(img_bgr, paddle_engine, vietocr_predictor):
    """
    Pipeline 2 model tinh gọn:
      1. PaddleOCR detection (DBNet)
      2. Normalize polygon (lọc nhỏ, lọc cao, lọc trùng/chứa)
      3. Sort reading order (trên xuống dưới, trái sang phải)
      4. Crop polygon với padding chuẩn xác
      5. VietOCR recognition
    """
    if vietocr_predictor is None:
        raise ValueError("VietOCR predictor là None (khởi tạo thất bại).")

    try:
        det_result = paddle_engine.ocr(img_bgr, det=True, rec=False, cls=False)
    except TypeError:
        det_result = paddle_engine.ocr(img_bgr)

    raw_lines = _extract_raw_polygons(det_result)
    if not raw_lines:
        return []

    polygons = _normalize_polygons(raw_lines)
    if not polygons:
        return []

    polygons_sorted = _sort_polygons_reading_order(polygons)

    recognized = []
    for pts in polygons_sorted:
        cropped, poly_4pts = _crop_box_from_poly(img_bgr, pts, expand=CROP_EXPAND)
        if cropped is None:
            continue
        try:
            pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            try:
                res = vietocr_predictor.predict(pil_crop, return_prob=True)
                if isinstance(res, (tuple, list)) and len(res) == 2:
                    text, prob = res[0], float(res[1])
                else:
                    text, prob = res, 1.0
            except TypeError:
                text, prob = vietocr_predictor.predict(pil_crop), 1.0

            if not text or not str(text).strip():
                continue

            recognized.append([poly_4pts, (str(text).strip(), prob)])
        except Exception as e:
            print(f"  ⚠️ Lỗi predict VietOCR: {e}")
            continue

    return recognized


def remove_duplicate_words(text: str) -> str:
    """Xóa các từ/cụm từ lặp lại liên tiếp do lỗi decoder loop của VietOCR."""
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\b((?:\w+\s+){1,4}?\w+)\s+(?:\1\s*){1,}',
        r'\1 ',
        text,
        flags=re.IGNORECASE
    )
    return text.strip()


def smart_sort_layout(lines) -> str:
    """
    Sắp xếp văn bản chuẩn theo thứ tự từ trên xuống dưới (Top to Bottom)
    và từ trái qua phải (Left to Right) dựa trên tọa độ Y trung bình của bounding box.
    Chỉ chạy remove_duplicate_words ở bước cuối cùng.
    """
    if not lines:
        return ""

    def get_sort_key(line_item):
        box = line_item[0]
        try:
            if isinstance(box, (list, tuple, np.ndarray)) and len(box) > 0:
                pts = np.array(box, dtype=np.float32)
                y_mean = float(pts[:, 1].mean())
                x_min = float(pts[:, 0].min())
                return (y_mean, x_min)
        except Exception:
            pass
        return (0.0, 0.0)

    sorted_lines = sorted(lines, key=get_sort_key)

    texts = []
    for line in sorted_lines:
        item = line[1]
        text = item[0] if isinstance(item, (tuple, list)) else item
        if text and str(text).strip():
            texts.append(str(text).strip())

    return remove_duplicate_words("\n".join(texts))
