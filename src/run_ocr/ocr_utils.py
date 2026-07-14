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
    Pipeline an toàn cho cả Sách Khắc Gỗ (HVB_003) và Sách Viết Tay (HVB_004)
    """
    h, w = img.shape[:2]
    img_up = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    
    # 1. CLAHE: Tăng tương phản cục bộ (Giữ nguyên nét khắc gỗ, không làm dính chữ)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 2. Median Blur: Khử các đốm ố vàng, vết bẩn li ti (Salt & Pepper noise) mà không làm mờ nét chữ
    denoised = cv2.medianBlur(enhanced, 3)
    
    # 3. Adaptive Threshold: Nhị phân hóa an toàn
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    
    # Resize về max 3500px để tránh tràn RAM
    max_side = 3500
    h_new, w_new = binary.shape[:2]
    if max(h_new, w_new) > max_side:
        scale = max_side / max(h_new, w_new)
        binary = cv2.resize(binary, (int(w_new*scale), int(h_new*scale)), interpolation=cv2.INTER_AREA)
        
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)