import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
from ocr_utils import load_and_process_input, enhance_image
from llm_corrector import correct_text_with_llm

def smart_sort_layout(result):
    if not result or not result[0]: return ""
    items = []
    for line in result[0]:
        box = line[0]
        text = line[1][0]
        cx = sum(p[0] for p in box) / 4.0
        cy = sum(p[1] for p in box) / 4.0
        h = max(abs(box[0][1] - box[2][1]), abs(box[1][1] - box[3][1]))
        w = max(abs(box[0][0] - box[1][0]), abs(box[2][0] - box[3][0]))
        if h == 0: h = 20
        if w == 0: w = 20
        items.append({'cx': cx, 'cy': cy, 'h': h, 'w': w, 'text': str(text)})
        
    avg_h = sum(i['h'] for i in items) / len(items)
    avg_w = sum(i['w'] for i in items) / len(items)
    
    if avg_h > avg_w * 1.2: 
        items.sort(key=lambda x: x['cx'], reverse=True)
        columns = []
        for item in items:
            placed = False
            for col in columns:
                col_avg_cx = sum(i['cx'] for i in col) / len(col)
                if abs(item['cx'] - col_avg_cx) < item['w'] * 1.5:
                    col.append(item)
                    placed = True
                    break
            if not placed:
                columns.append([item])
        final_text_lines = []
        for col in columns:
            col.sort(key=lambda x: x['cy'])
            col_text = "".join([i['text'] for i in col])
            final_text_lines.append(col_text)
        return "\n".join(final_text_lines)
    else: 
        items.sort(key=lambda x: (x['cy'], x['cx']))
        lines = []
        current_line = []
        last_cy = -1000
        for item in items:
            if abs(item['cy'] - last_cy) < item['h'] * 0.5:
                current_line.append(item)
            else:
                if current_line:
                    current_line.sort(key=lambda x: x['cx'])
                    lines.append(" ".join([i['text'] for i in current_line]))
                current_line = [item]
                last_cy = item['cy']
        if current_line:
            current_line.sort(key=lambda x: x['cx'])
            lines.append(" ".join([i['text'] for i in current_line]))
        return "\n".join(lines)

def clean_viet_text(raw_text):
    """
    Làm sạch text tiếng Việt / Nôm để chuẩn bị cho bước Dóng hàng (Alignment).
    - Loại bỏ toàn bộ dấu câu, ký tự đặc biệt do OCR hallucinate.
    - Loại bỏ số trang, header/footer rác.
    - Giữ lại chữ cái, chữ Hán/Nôm và khoảng trắng (để không bị dính từ).
    """
    lines = raw_text.split('\n')
    clean_lines = []
    
    # Regex giữ lại: 
    # 1. Chữ cái Latinh (kèm full dấu tiếng Việt)
    # 2. Chữ Hán/Nôm
    # 3. Khoảng trắng (\s)
    # -> Mọi thứ khác (dấu chấm, phẩy, ngoặc, ký tự lạ...) sẽ bị xóa sạch
    pattern_keep = re.compile(
        r'[^a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ'
        r'ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ'
        r'\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002A6DF\s]'
    )
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # 1. Bỏ dòng rác: Chỉ chứa số, dấu câu, ký tự đặc biệt (Thường là số trang, vết ố)
        if re.match(r'^[\d\s\W_]+$', line): continue
        
        # 2. Xóa dấu câu và ký tự lạ
        clean_line = pattern_keep.sub('', line)
        
        # 3. Chuẩn hóa khoảng trắng (gộp nhiều space thừa thành 1 space chuẩn)
        clean_line = re.sub(r'\s+', ' ', clean_line).strip()
        
        # 4. Lọc dòng quá ngắn (dưới 2 từ) để tránh noise OCR rải rác
        if len(clean_line.split()) >= 2:
            clean_lines.append(clean_line)
            
    return '\n'.join(clean_lines)

def run_viet_ocr():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    ocr = PaddleOCR(
        use_angle_cls=True, lang='vi', show_log=False,
        det_db_box_thresh=0.3, det_db_thresh=0.2, drop_score=0.1
    )
    
    print("Đã khởi tạo PaddleOCR (Việt) & kết nối LLM API")
    
    for work in config['works']:
        file_path = work['vie_file']
        file_type = work['vie_type']
        work_id = work['id']
        work_title = work['viet']
        
        print(f"Đang xử lý Việt: {work_title} ({work_id})...")
        pages, data_type = load_and_process_input(file_path, file_type, work_id)
        
        raw_text_pages = []
        if data_type == "text":
            raw_text_pages = pages 
        elif data_type == "image":
            for idx, img in enumerate(pages):
                start_time = time.time()
                print(f"  -> Đang xử lý & OCR trang {idx + 1}/{len(pages)}...")
                
                img_input = enhance_image(img)
                
                try:
                    result = ocr.ocr(img_input, cls=True)
                    page_text_sorted = smart_sort_layout(result)
                    
                    # GỌI LLM THÔNG MINH
                    page_text_corrected = correct_text_with_llm(page_text_sorted, work_title, language="việt")
                            
                    raw_text_pages.append(page_text_corrected)
                    print(f"  Trang {idx + 1} hoàn tất - Tốn {time.time() - start_time:.1f}s")
                except Exception as e:
                    print(f"  Lỗi OCR trang {idx + 1}: {e}")
        
        full_raw_text = "\n".join(raw_text_pages)
        clean_text = clean_viet_text(full_raw_text)
                    
        out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ocr_output')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{work_id}_vie_raw.txt")
        
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(clean_text)
        print(f"Hoàn tất tác phẩm: {out_path}\n")

if __name__ == "__main__":
    run_viet_ocr()