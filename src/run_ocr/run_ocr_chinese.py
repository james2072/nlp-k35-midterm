import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Fix xung đột OpenMP

import json
import sys
import logging
import time
import warnings
import re

warnings.filterwarnings("ignore")
logging.getLogger('ppocr').setLevel(logging.ERROR)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from paddleocr import PaddleOCR
from ocr_utils import load_and_process_input, enhance_han_nom_image

def clean_han_text(raw_text):
    """Lọc noise, bỏ dấu câu, giữ lại chuỗi chữ Hán/Nôm liền mạch để dóng hàng"""
    lines = raw_text.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue
        if re.match(r'^[\d\sIVXivx\-\_\.]+$', line): continue
        han_nom_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002A6DF]', line)
        if len(han_nom_chars) < 2: continue 
        clean_line = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002A6DF\s]', '', line)
        clean_line = re.sub(r'\s+', '', clean_line)
        if clean_line:
            clean_lines.append(clean_line)
    return '\n'.join(clean_lines)

def run_han_ocr():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    # Khởi tạo PaddleOCR 2.8.1 
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
    
    for work in config['works']:
        file_path = work['sino_file']
        file_type = work['sino_type']
        work_id = work['id']
        
        print(f"Đang xử lý Hán: {work['viet']} ({work_id})...")
        pages, data_type = load_and_process_input(file_path, file_type, work_id)
        
        raw_text_pages = []
        if data_type == "text":
            raw_text_pages = pages 
        elif data_type == "image":
            for idx, img in enumerate(pages):
                start_time = time.time()
                print(f"  -> Đang xử lý & OCR trang {idx + 1}/{len(pages)}...")
                
                # Dùng OpenCV Advanced để làm nét chữ cổ
                img_input = enhance_han_nom_image(img)
                
                try:
                    # API chuẩn của PaddleOCR 2.x
                    result = ocr.ocr(img_input, cls=True)
                    
                    page_text = []
                    if result and result[0]:
                        for line in result[0]:
                            page_text.append(line[1][0])
                            
                    raw_text_pages.append("\n".join(page_text))
                    print(f"Trang {idx + 1} hoàn tất ({len(page_text)} dòng) - Tốn {time.time() - start_time:.1f}s")
                except Exception as e:
                    print(f"Lỗi OCR trang {idx + 1}: {e}")
        
        full_raw_text = "\n".join(raw_text_pages)
        clean_text = clean_han_text(full_raw_text)
                    
        out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ocr_output')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{work_id}_sino_raw.txt")
        
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(clean_text)
        print(f"Hoàn tất tác phẩm: {out_path}\n")

if __name__ == "__main__":
    run_han_ocr()