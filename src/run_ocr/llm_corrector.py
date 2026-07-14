import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# GLOBAL CONFIG
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:1234/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen/qwen3-4b-2507")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 300))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 3))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 4096))
LLM_CHUNK_LINES = int(os.getenv("LLM_CHUNK_LINES", 100))
LLM_OVERLAP_LINES = int(os.getenv("LLM_OVERLAP_LINES", 20))

def call_llm_api(system_prompt, user_prompt):
    # call API LLM chuẩn OpenAI
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": LLM_MAX_TOKENS,
        "stop": ["\n\n\n", "User:", "Giải thích:", "Phân tích:", "Dưới đây là", "Kết quả là", "Note:", "Chú thích:"]
    }
    
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=LLM_TIMEOUT)
            if response.status_code == 200:
                res_json = response.json()
                content = res_json['choices'][0]['message']['content'].strip()
                
                # Lọc bỏ markdown code block của LLM
                if content.startswith("```"):
                    content = re.sub(r'^```[a-z]*\n', '', content)
                    content = re.sub(r'\n```$', '', content)
                return content
            else:
                print(f"API Error {response.status_code} (Retry {attempt + 1}/{LLM_MAX_RETRIES})")
        except requests.exceptions.Timeout:
            print(f"Timeout after {LLM_TIMEOUT}s (Retry {attempt + 1}/{LLM_MAX_RETRIES})")
        except Exception as e:
            print(f"Exception: {e} (Retry {attempt + 1}/{LLM_MAX_RETRIES})")
            
        time.sleep(2 ** attempt)
        
    return None

def correct_text_with_llm(full_text, work_title, language="hán"):
    """
    Chunking thông minh: 
    1. Ưu tiên gửi nguyên trang (nếu < LLM_CHUNK_LINES).
    2. Nếu dài quá thì cắt chunk và có OVERLAP để giữ bối cảnh.
    """
    if not full_text.strip(): return full_text
    
    lines = full_text.split('\n')
    
    # Nếu text ngắn hơn ngưỡng chunk -> Gửi luôn 1 cục
    if len(lines) <= LLM_CHUNK_LINES:
        chunks = [full_text]
        contexts = [""] # Không có context trước
    else:
        chunks = []
        contexts = []
        stride = LLM_CHUNK_LINES - LLM_OVERLAP_LINES
        
        for i in range(0, len(lines), stride):
            chunk_lines = lines[i : i + LLM_CHUNK_LINES]
            chunks.append("\n".join(chunk_lines))
            
            # Lấy overlap của chunk trước làm context
            if i == 0:
                contexts.append("")
            else:
                prev_lines = lines[max(0, i - LLM_OVERLAP_LINES) : i]
                contexts.append("\n".join(prev_lines))
                
        if len(chunks) > 1 and len(chunks[-1].split('\n')) < LLM_OVERLAP_LINES:
            chunks.pop()
            contexts.pop()

    # prompt
    if language == "hán":
        system_prompt = "Bạn là chuyên gia Hán Nôm cổ sử Việt Nam. NHIỆM VỤ: Sửa lỗi OCR, điền chữ mờ dựa trên văn ngôn văn, điều chỉnh thành câu có nghĩa, điều chỉnh các từ có thể còn thiếu hoặc orc sai tạo thành câu có nghĩa, có thể dựa vào các từ trong câu để sửa thành một câu hoàn chỉnh có nghĩa hoặc dựa vào các câu hoặc các từ ở câu ở trước hoặc sau để sửa thành một câu hoàn chỉnh. KHÔNG giải thích, KHÔNG markdown. Chỉ trả về text mộc đã sửa."
    else:
        system_prompt = "Bạn là biên tập viên văn bản cổ sử Việt Nam (Quốc ngữ / Hán Nôm). NHIỆM VỤ: Sửa lỗi OCR, điền chữ mờ, sửa lỗi chính tả tiếng Việt cổ có dấu, điều chỉnh thành câu có nghĩa, các từ có thể còn thiếu hoặc orc sai tạo thành câu có nghĩa, có thể dựa vào các từ trong câu để sửa thành một câu hoàn chỉnh có nghĩa hoặc dựa vào các câu hoặc các từ ở câu ở trước hoặc sau để sửa thành một câu hoàn chỉnh. Đảm bảo là tiếng Việt có dấu nếu ORC chưa nhận diện dấu. KHÔNG giải thích. Chỉ trả về text mộc."

    corrected_parts = []
    total_chunks = len(chunks)
    
    for idx, (chunk, context) in enumerate(zip(chunks, contexts)):
        if not chunk.strip():
            corrected_parts.append(chunk)
            continue
            
        print(f"Đang LLM chunk {idx + 1}/{total_chunks}...", end=" ", flush=True)
        
        # Tạo prompt có bối cảnh
        user_prompt = f"""Tác phẩm: "{work_title}"
Bối cảnh (đoạn trước - chỉ để tham khảo, KHÔNG sửa):
{context if context else '(Đây là phần đầu tác phẩm)'}

Đoạn cần sửa (HÃY SỬA LỖI OCR, ĐIỀN DẤU ĐÚNG VỚI Ý NGHĨA, ĐIỀN CHỮ THIẾU, ĐẢM BẢO Ý NGHĨA CÂU PHÙ HỢP, giữ nguyên số dòng):
{chunk}
"""
        
        corrected_chunk = call_llm_api(system_prompt, user_prompt)
        
        if corrected_chunk is None:
            print("FAILED (Giữ nguyên gốc)")
            corrected_parts.append(chunk) # Fallback
        else:
            print("OK!")
            corrected_parts.append(corrected_chunk)
            
    return "\n".join(corrected_parts)