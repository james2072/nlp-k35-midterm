import json
import os
import csv
import requests
import re

# Cấu hình Local LLM API (OpenAI Compatible Endpoint)
LOCAL_LLM_URL = "http://localhost:8000/v1/chat/completions" # Đổi port nếu cần
MODEL_NAME = "Qwen2.5-7B-Instruct_openthoughts3_300k_annotated_Qwen3-32B"

def split_sentences(text):
    # Tách câu dựa trên dấu chấm câu hoặc xuống dòng
    sentences = re.split(r'(?<=[.!?。！？\n])\s+', text.strip())
    return [s for s in sentences if len(s) > 2]

def align_with_llm(han_chunk, viet_chunk):
    prompt = f"""
    Bạn là chuyên gia Hán Nôm. Hãy dóng hàng (align) các câu chữ Hán và câu tiếng Việt/Nôm sau đây 
    thành các cặp có nghĩa tương đương. Không được bỏ sót câu.
    Trả về kết quả DUY NHẤT dưới dạng JSON Array với format: [{{"han": "...", "viet": "..."}}]
    
    [HÁN]:
    {han_chunk}
    
    [VIỆT]:
    {viet_chunk}
    """
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": { "type": "json_object" }
    }
    
    try:
        response = requests.post(LOCAL_LLM_URL, json=payload).json()
        content = response['choices'][0]['message']['content']
        # Parse JSON
        return json.loads(content)
    except Exception as e:
        print(f"Lỗi gọi LLM: {e}")
        return []

def run_alignment():
    with open('data/config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    for work in config['works']:
        matacpham = work['id']
        han_file = f"data/ocr_output/{matacpham}_han_raw.txt"
        viet_file = f"data/ocr_output/{matacpham}_viet_raw.txt"
        
        if not os.path.exists(han_file) or not os.path.exists(viet_file):
            continue
            
        han_sents = split_sentences(open(han_file, encoding='utf-8').read())
        viet_sents = split_sentences(open(viet_file, encoding='utf-8').read())
        
        # Chunking 10 câu / 1 LLM
        chunk_size = 10
        aligned_pairs = []
        
        for i in range(0, min(len(han_sents), len(viet_sents)), chunk_size):
            h_chunk = "\n".join(han_sents[i:i+chunk_size])
            v_chunk = "\n".join(viet_sents[i:i+chunk_size])
            aligned_pairs.extend(align_with_llm(h_chunk, v_chunk))
            
        # Ghi ra TSV: [pair_id]\t[han_sentence]\t[viet_sentence]
        out_path = f"data/corpus/{matacpham}_parallel.tsv"
        with open(out_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['pair_id', 'han_sentence', 'viet_sentence'])
            for idx, pair in enumerate(aligned_pairs):
                writer.writerow([f"{matacpham}_{idx:04d}", pair.get('han', ''), pair.get('viet', '')])
                
        print(f"Đã dóng hàng: {out_path}")

if __name__ == "__main__":
    run_alignment()