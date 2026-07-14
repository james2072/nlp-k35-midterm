import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

LLM_API_URL    = os.getenv("LLM_API_URL",    "http://localhost:1234/v1/chat/completions")
LLM_API_KEY    = os.getenv("LLM_API_KEY",    "lm-studio")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen/qwen3-4b-2507")
LLM_TIMEOUT    = int(os.getenv("LLM_TIMEOUT",    300))
LLM_MAX_RETRIES= int(os.getenv("LLM_MAX_RETRIES", 3))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 4096))
LLM_CHUNK_LINES= int(os.getenv("LLM_CHUNK_LINES", 100))
LLM_OVERLAP_LINES=int(os.getenv("LLM_OVERLAP_LINES", 20))

SYSTEM_PROMPTS = {
    "sino": (
        "Bạn là chuyên gia Hán Nôm cổ sử Việt Nam. "
        "NHIỆM VỤ: Sửa lỗi OCR, điền chữ mờ dựa trên văn ngôn, "
        "điều chỉnh thành câu có nghĩa dựa vào ngữ cảnh xung quanh. "
        "KHÔNG giải thích, KHÔNG markdown. Chỉ trả về text mộc đã sửa."
    ),
    "vie": (
        "Bạn là biên tập viên văn bản cổ sử Việt Nam (Quốc ngữ / Hán Nôm). "
        "NHIỆM VỤ: Sửa lỗi OCR, điền chữ mờ, sửa chính tả tiếng Việt cổ có dấu, "
        "điều chỉnh thành câu có nghĩa dựa vào ngữ cảnh xung quanh. "
        "KHÔNG giải thích. Chỉ trả về text mộc."
    ),
}

STOP_TOKENS = ["User:", "Giải thích:", "Phân tích:", "Note:", "Chú thích:"]


def call_llm_api(system_prompt: str, user_prompt: str) -> str | None:
    """Gọi OpenAI-compatible API, retry tối đa LLM_MAX_RETRIES lần."""
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": LLM_MAX_TOKENS,
        "stop": STOP_TOKENS,
    }

    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=LLM_TIMEOUT)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                # Strip markdown code fences nếu LLM trả về
                content = re.sub(r"^```[a-z]*\n", "", content)
                content = re.sub(r"\n```$", "", content)
                return content
            print(f"  API {resp.status_code} (retry {attempt + 1}/{LLM_MAX_RETRIES})")
        except requests.exceptions.Timeout:
            print(f"  Timeout {LLM_TIMEOUT}s (retry {attempt + 1}/{LLM_MAX_RETRIES})")
        except Exception as e:
            print(f"  Lỗi: {e} (retry {attempt + 1}/{LLM_MAX_RETRIES})")

        time.sleep(2 ** attempt)

    return None


def _build_user_prompt(work_title: str, chunk: str, context: str) -> str:
    ctx_text = context if context else "(Đây là phần đầu tác phẩm)"
    return (
        f'Tác phẩm: "{work_title}"\n'
        f"Bối cảnh (đoạn trước - chỉ để tham khảo, KHÔNG sửa):\n{ctx_text}\n\n"
        f"Đoạn cần sửa (sửa lỗi OCR, điền dấu, điền chữ thiếu, giữ nguyên số dòng):\n{chunk}\n"
    )


def correct_text_with_llm(full_text: str, work_title: str, language: str = "sino") -> str:
    """
    Chia text thành chunks có overlap, gửi từng chunk cho LLM sửa lỗi OCR,
    rồi ghép lại (trim phần overlap để không bị nhân đôi dòng).
    """
    if not full_text.strip():
        return full_text

    lines = full_text.split("\n")
    stride = max(1, LLM_CHUNK_LINES - LLM_OVERLAP_LINES)
    system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["vie"])

    # Tạo danh sách (chunk_text, context_text) — thống nhất cả trường hợp ngắn và dài
    chunks = [
        (
            "\n".join(lines[i : i + LLM_CHUNK_LINES]),
            "\n".join(lines[max(0, i - LLM_OVERLAP_LINES) : i]) if i > 0 else "",
        )
        for i in range(0, len(lines), stride)
    ]

    result_lines: list[str] = []
    for idx, (chunk, context) in enumerate(chunks):
        if not chunk.strip():
            continue

        print(f"  LLM chunk {idx + 1}/{len(chunks)}...", end=" ", flush=True)
        corrected = call_llm_api(system_prompt, _build_user_prompt(work_title, chunk, context))

        if corrected is None:
            print("FAILED (giữ gốc)")
            corrected = chunk
        else:
            print("OK")

        corrected_lines = corrected.split("\n")
        # Chunk đầu tiên: lấy toàn bộ
        # Chunk tiếp theo: trim LLM_OVERLAP_LINES dòng đầu (là phần overlap đã có ở chunk trước)
        result_lines.extend(corrected_lines if idx == 0 else corrected_lines[LLM_OVERLAP_LINES:])

    return "\n".join(result_lines)