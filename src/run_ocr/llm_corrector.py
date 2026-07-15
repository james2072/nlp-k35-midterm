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
        "NHIỆM VỤ: Sửa lỗi OCR, điền chữ mờ dựa trên văn ngôn, điều chỉnh thành câu có nghĩa. "
        "QUY TẮC CHUẨN HÓA CHO SENTENCE ALIGNMENT: "
        "1. Xóa bỏ các chữ số/ký hiệu cước chú gắn sau từ (ví dụ: 'chữ¹' -> 'chữ'). Nếu dòng là số trang mồ côi, tiêu đề lặp lại hay cước chú hiện đại, hãy để dòng đó TRỐNG (blank line) để loại bỏ mà không làm lệch chỉ số dòng gối đầu. "
        "2. QUY TẮC BẮT BUỘC: Giữ nguyên chính xác số lượng dòng (1-to-1). KHÔNG gộp dòng chính văn, KHÔNG tách dòng, KHÔNG giải thích, KHÔNG markdown. Chỉ trả về text mộc đã sửa."
    ),
    "vie": (
        "Bạn là biên tập viên văn bản cổ sử Việt Nam (Quốc ngữ / Hán Nôm). "
        "NHIỆM VỤ: Sửa lỗi OCR, điền chữ mờ, sửa chính tả tiếng Việt cổ có dấu. "
        "QUY TẮC CHUẨN HÓA CHO SENTENCE ALIGNMENT: "
        "1. Xóa bỏ chỉ số cước chú gắn sau từ (ví dụ: 'phần¹', 'Hoa Bằng²' -> 'phần', 'Hoa Bằng'). Nếu dòng chỉ chứa số trang (ví dụ '9', '12'), rác không chữ, hay cước chú chân trang (ví dụ: '¹ Xem...', 'Tr. 57-75'), hãy để dòng đó TRỐNG (blank line) để loại bỏ mà không làm lệch chỉ số dòng gối đầu. "
        "2. QUY TẮC BẮT BUỘC: Giữ nguyên chính xác số lượng dòng (1-to-1). KHÔNG gộp dòng chính văn, KHÔNG tách dòng, KHÔNG giải thích. Chỉ trả về text mộc đã sửa."
    ),
}

STOP_TOKENS = ["User:", "Giải thích:", "Phân tích:", "Note:", "Chú thích:"]


def filter_for_alignment(text: str, language: str = "vie") -> str:
    """
    Lọc sạch văn bản sau OCR/LLM để phục vụ tối ưu cho Sentence Alignment:
    - Loại bỏ các dòng trống hoặc chỉ chứa khoảng trắng/chấm câu.
    - Loại bỏ dòng chỉ chứa số trang/chữ số mồ côi (ví dụ: '9', '- 12 -', 'Trang 45').
    - Loại bỏ cước chú chân trang, citation sách báo hiện đại không thuộc chính văn lịch sử (ví dụ: '¹ Xem...', 'Tr. 57-75').
    - Xóa chỉ số cước chú gắn liền trong câu (ví dụ: 'phần¹' -> 'phần', 'sách[1]' -> 'sách').
    """
    if not text or not text.strip():
        return ""

    lines = text.split("\n")
    clean_lines = []

    # Regex nhận diện cước chú / citation điển hình ở chân trang sách lịch sử tiếng Việt
    footnote_regex = re.compile(
        r"^[\(\[\{]?([0-9¹²³⁴⁵⁶⁷⁸⁹\*]+|[\*†‡]+)[\)\]\}]?\s*(Xem|Chú|Chú thích|Chú dẫn|Sđd|Tr\.|Trang|Theo|Bản|Nguyên|Tập san|Tạp chí|Nxb|Nhà xuất bản|Tham khảo)",
        re.IGNORECASE
    )

    # Regex nhận diện dòng chỉ chứa số trang Hán tự hoặc tiêu đề trang (quyển/trang mồ côi như '一', '第 二 頁', '卷 一')
    sino_page_number_regex = re.compile(
        r"^([第]?\s*[一二三四五六七八九十百千零〇・]+\s*[頁葉卷篇上下]?|[一二三四五六七八九十百千零〇・\s]+)$"
    )

    # Regex xóa chỉ số cước chú nhỏ gắn sau từ (superscript ¹²³⁴⁵⁶⁷⁸⁹⁰†‡* hoặc ngoặc vuông/tròn [1], (1) dính sát sau chữ)
    inline_citation_regex = re.compile(
        r"[¹²³⁴⁵⁶⁷⁸⁹⁰†‡]+|(?<=[a-zA-Z\u00c0-\u024f\u1e00-\u1eff\u4e00-\u9fff])[\(\[\{]\d+[\)\]\}]"
    )

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        # 1. Bỏ qua dòng chỉ chứa số hoặc ký tự đặc biệt/dấu câu (số trang mồ côi, gạch ngang trang...)
        if re.match(r"^[\d\s\W_]+$", line_str) or not any(c.isalpha() or '\u4e00' <= c <= '\u9fff' for c in line_str):
            continue

        # 2. Bỏ qua cước chú/citation hiện đại (khi xử lý tiếng Việt)
        if language == "vie" and footnote_regex.match(line_str):
            continue

        # 2b. Bỏ qua số trang Hán tự / số Hán mồ côi ở đầu/cuối trang (khi xử lý tiếng Hán)
        if language == "sino" and sino_page_number_regex.match(line_str):
            continue

        # 3. Xóa chỉ số cước chú gắn liền trong câu (như 'phần¹' -> 'phần', 'sách[2]' -> 'sách')
        line_str = inline_citation_regex.sub("", line_str)
        line_str = re.sub(r"\s+", " ", line_str).strip()

        if line_str:
            clean_lines.append(line_str)

    return "\n".join(clean_lines)


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
                content = re.sub(r"^```[a-zA-Z]*\r?\n?(.*?)\r?\n?```$", r"\1", content, flags=re.DOTALL).strip()
                return content
            print(f"  API {resp.status_code}: {resp.text[:150]} (retry {attempt + 1}/{LLM_MAX_RETRIES})")
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
        # Chunk tiếp theo: trim gối đầu an toàn (chỉ cắt tối đa đến dòng áp chót nếu số dòng ngắn hơn LLM_OVERLAP_LINES)
        if idx == 0:
            result_lines.extend(corrected_lines)
        else:
            trim_idx = min(LLM_OVERLAP_LINES, max(0, len(corrected_lines) - 1))
            result_lines.extend(corrected_lines[trim_idx:])

    merged_text = "\n".join(result_lines)
    # Lọc sạch lần cuối để loại bỏ số trang mồ côi, rác và cước chú phục vụ Sentence Alignment
    return filter_for_alignment(merged_text, language=language)