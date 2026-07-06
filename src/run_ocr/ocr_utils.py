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
                has_text = False
                for page in doc:
                    text = page.get_text("text")
                    if text.strip(): has_text = True
                    text_pages.append(text)
                if has_text:
                    doc.close()
                    return text_pages, "text"
            
            images = []
            for page in doc:
                mat = fitz.Matrix(200/72, 200/72) # Render 200 DPI
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

def enhance_han_nom_image(img):
    """
    Pipeline làm nét chữ Hán/Nôm cổ trên giấy dó (Không dùng AI/PyTorch)
    1. Upscale Lanczos giữ nguyên nét sổ
    2. Unsharp Masking làm đậm nét mực tàu
    3. Adaptive Threshold khử vết ố vàng, ánh sáng không đều
    4. Morphology khử đốm bẩn
    """
    # 1. Upscale x2 bằng Lanczos (Giữ cạnh sắc nét, không bị mờ như AI)
    h, w = img.shape[:2]
    img_up = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
    
    # 2. Chuyển sang ảnh xám
    gray = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    
    # 3. Unsharp Masking (Làm nổi bật cạnh nét mực)
    blurred = cv2.GaussianBlur(gray, (0,0), 3)
    sharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    
    # 4. Adaptive Threshold (Tách chữ đen trên nền giấy cổ ố vàng)
    binary = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)
    
    # 5. Morphological Operations (Khử noise đốm bẩn, nối các nét đứt)
    kernel = np.ones((2,2), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 6. Resize về max 3500px để PaddleOCR không bị tràn RAM
    max_side = 3500
    h_new, w_new = clean.shape[:2]
    if max(h_new, w_new) > max_side:
        scale = max_side / max(h_new, w_new)
        clean = cv2.resize(clean, (int(w_new*scale), int(h_new*scale)), interpolation=cv2.INTER_AREA)
        
    # PaddleOCR yêu cầu ảnh 3 kênh màu BGR
    return cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)