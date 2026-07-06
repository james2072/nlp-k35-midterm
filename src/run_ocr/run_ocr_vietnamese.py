import json
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from ocr_utils import load_and_process_input, upscale_image, binarize_image

def run_viet_ocr():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    # VietOCR
    cfg = Cfg.load_config_from_name('vgg_transformer')
    weights_path = os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'transformerocr.pth')
    cfg['weights'] = weights_path
    cfg['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    detector = Predictor(cfg)
    
    for work in config['works']:
        file_path = work['vie_file']
        file_type = work['vie_type']
        work_id = work['id']
        
        print(f"Đang xử lý Việt: {work['viet']} ({work_id})...")
        pages, data_type = load_and_process_input(file_path, file_type)
        
        raw_text = []
        if data_type == "text":
            raw_text = pages 
        elif data_type == "image":
            for img in pages:
                try:
                    img_up = upscale_image(img)
                    img_bin = binarize_image(img_up)
                    page_text = detector.predict(img_bin)
                    raw_text.append(page_text)
                except Exception as e:
                    print(f"Lỗi OCR trang: {e}")
                    
        out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ocr_output')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{work_id}_vie_raw.txt")
        
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(raw_text))
        print(f"Hoàn tất: {out_path}")

if __name__ == "__main__":
    run_viet_ocr()