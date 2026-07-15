import os
import cv2
import numpy as np
import fitz  # PyMuPDF

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def find_file(file_path, work_id):
    root = get_project_root()
    abs_path = os.path.normpath(os.path.join(root, file_path))
    if os.path.exists(abs_path): return abs_path
        
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
                mat = fitz.Matrix(250/72, 250/72) # Render 250 DPI (Cân bằng giữa nét và nhẹ)
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                else: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                images.append(img)
            doc.close()
            return images, "image"
        else:
            img = cv2.imread(abs_path)
            if img is not None: return [img], "image"
    return [], "unknown"

def enhance_image(img):
    """
    Pipeline làm sạch và tăng độ tương phản an toàn cho tài liệu chữ Hán/Nôm/Việt:
    - Giữ lại thang độ xám/RGB tự nhiên (không dùng adaptiveThreshold nhị phân thô để tránh mất nét/tạo nhiễu).
    - Dùng CLAHE + Median Blur + Unsharp Masking để chữ nổi bật, viền sắc nét cho DBNet phân vùng chuẩn xác.
    """
    h, w = img.shape[:2]
    # Resize vừa phải nếu ảnh quá nhỏ để DBNet bắt tốt các dấu tiếng Việt li ti
    if max(h, w) < 2000:
        scale = min(2.0, 2500 / max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. CLAHE: Tăng tương phản cục bộ, làm rõ chữ trên nền giấy ố vàng
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 2. Median Blur nhẹ: Khử nhiễu đốm li ti (salt & pepper) trên giấy cổ
    denoised = cv2.medianBlur(enhanced, 3)
    
    # 3. Unsharp Masking: Tăng độ sắc nét viền chữ giúp DBNet không bị gộp dòng khi các chữ nằm sát nhau
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
    
    # Resize về max 3500px để đảm bảo không tràn RAM
    max_side = 3500
    h_new, w_new = sharpened.shape[:2]
    if max(h_new, w_new) > max_side:
        scale = max_side / max(h_new, w_new)
        sharpened = cv2.resize(sharpened, (int(w_new * scale), int(h_new * scale)), interpolation=cv2.INTER_AREA)
        
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def init_paddleocr(lang="vi", use_angle_cls=True):
    """
    Khởi tạo engine PaddleOCR tập trung:
    - Tự động thiết lập các cờ môi trường (KMP, PIR, MKLDNN) để tránh lỗi C++ trên Windows CPU.
    - Xử lý đường dẫn DLL cho torch/shm.dll để tránh WinError 127 trên Windows.
    - Ngưỡng DBNet được tinh chỉnh tối ưu (text_det_thresh=0.3, text_det_box_thresh=0.5)
      để khung bounding box ôm sát từng dòng/chữ, không bị gộp nhầm hay tạo khung rỗng.
    """
    import logging
    import warnings
    import sys

    # 1. Cờ môi trường chống xung đột OpenMP và lỗi C++ PIR/MKL-DNN trên Windows
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["FLAGS_enable_pir_api"] = "0"
    os.environ["FLAGS_enable_pir_in_executor"] = "0"
    os.environ["FLAGS_use_mkldnn"] = "0"

    # Fix WinError 127 khi modelscope/paddlex import torch shm.dll trên Windows
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

    # 2. Tự động kiểm tra thiết bị CUDA vs CPU
    device = "cpu"
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            device = "gpu"
    except Exception:
        device = "cpu"

    # 3. Khởi tạo PaddleOCR với ngưỡng tối ưu cho sách chữ Hán/Nôm/Việt
    # Import torch trước khi import paddleocr để tránh lỗi WinError 127 (shm.dll) trên Windows
    try:
        import torch
    except Exception:
        pass

    from paddleocr import PaddleOCR
    ocr_engine = PaddleOCR(
        use_angle_cls=use_angle_cls,
        lang=lang,
        device=device,
        enable_mkldnn=False,          # Tắt MKLDNN để tránh lỗi onednn_instruction.cc
        text_det_box_thresh=0.6,      # Ngưỡng bbox (0.6): loại bỏ triệt để khung rỗng ở viền giấy/đầu trang
        text_det_thresh=0.35,         # Ngưỡng nhị phân DBNet (0.35): ngăn nhiễu nền và phân tách rõ ràng
        text_det_unclip_ratio=1.2,    # Thu gọn tỷ lệ unclip (1.2 thay vì 1.5): ngăn các dòng sát nhau bị lấn/chồng chéo dọc
        text_rec_score_thresh=0.1,
    )
    return ocr_engine


def normalize_ocr_result(result):
    """
    Chuẩn hóa cấu trúc kết quả từ PaddleOCR v3.x (OCRResult object/dict) hoặc v2.x
    về định dạng list tuples tiêu chuẩn: [ [box, (text, score)], ... ]
    Đảm bảo idempotent (an toàn tuyệt đối khi truyền vào cả result raw hoặc list đã chuẩn hóa).
    """
    if not result or len(result) == 0:
        return []

    # Trường hợp 1: result là 1 dictionary (OCRResult object/dict)
    if isinstance(result, dict) and "rec_texts" in result:
        boxes = result.get("dt_polys", result.get("rec_polys", []))
        texts = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])
        return [[box, (txt, score)] for box, txt, score in zip(boxes, texts, scores)]

    # Trường hợp 2: result là list
    if isinstance(result, list):
        first = result[0]
        if first is None:
            return []
        # Nếu result = [ dict_OCRResult ]
        if isinstance(first, dict) and "rec_texts" in first:
            boxes = first.get("dt_polys", first.get("rec_polys", []))
            texts = first.get("rec_texts", [])
            scores = first.get("rec_scores", [])
            return [[box, (txt, score)] for box, txt, score in zip(boxes, texts, scores)]

        # Nếu first có cấu trúc của 1 dòng đã chuẩn hóa: [box, (txt, score)]
        if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[1], (tuple, list)) and len(first[1]) == 2 and isinstance(first[1][0], (str, np.str_)):
            return result

        # Nếu first là danh sách các dòng (v2.x format: [ [ [box, (txt, score)], ... ] ])
        if isinstance(first, list):
            return first

    return []