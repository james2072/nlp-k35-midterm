import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

LLM_API_URL     = os.getenv("LLM_API_URL",    "http://localhost:1234/v1/chat/completions")
LLM_API_KEY     = os.getenv("LLM_API_KEY",    "lm-studio")
LLM_MODEL_NAME  = os.getenv("LLM_MODEL_NAME", "qwen/qwen3-4b-2507")
LLM_TIMEOUT     = int(os.getenv("LLM_TIMEOUT",    300))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 3))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", 4096))
LLM_CHUNK_LINES = int(os.getenv("LLM_CHUNK_LINES", 100))
LLM_OVERLAP_LINES = int(os.getenv("LLM_OVERLAP_LINES", 20))

SYSTEM_PROMPT = (
    "Bạn là chuyên gia ngôn ngữ và biên tập viên văn bản tiếng Việt (Quốc ngữ). "
    "NHIỆM VỤ: Sửa lỗi chính tả, lỗi nhận diện OCR, khôi phục dấu thanh bị mờ để văn bản tiếng Việt chuẩn xác, tường minh "
    "phục vụ cho việc đối chiếu câu song ngữ (Sentence Alignment) với nguyên văn Hán ngữ sau này.\n"
    "QUY TẮC BẮT BUỘC (NGHIÊM NGẶT):\n"
    "1. TUYỆT ĐỐI KHÔNG TÓM TẮT (NO SUMMARIZATION), KHÔNG tự ý viết lại hay diễn giải/biến tấu lời văn. Phải giữ nguyên 100% ý nghĩa, từ ngữ và trật tự câu của tác giả.\n"
    "2. Chỉ sửa chữa những từ bị lỗi OCR rõ ràng (ví dụ: mất dấu, sai chính tả do quét mờ) và nối các câu bị ngắt xuống dòng sai ngữ pháp thành câu/đoạn hoàn chỉnh trôi chảy.\n"
    "3. Loại bỏ hoàn toàn các dòng rác không có nghĩa, số trang đơn lẻ (như '10', '0361903100135'), cước chú viền trang không thuộc phần bản văn chính, và xóa chỉ số cước chú gắn sát đuôi từ (ví dụ: 'phần¹' -> 'phần', 'sách[1]' -> 'sách').\n"
    "4. Nếu một đoạn văn bị nhận diện OCR quá nát/mất chữ đến mức không thể khôi phục chính xác từng từ, hãy sửa tối thiểu và giữ nguyên cấu trúc gốc, TUYỆT ĐỐI KHÔNG bịa đặt hay tóm lược lại cả đoạn thành 1 câu mới.\n"
    "5. Trả về văn bản tiếng Việt đã được biên tập sạch sẽ. KHÔNG kèm lời chào, giải thích hay bình luận."
)

STOP_TOKENS = ["User:", "Giải thích:", "Phân tích:", "Note:", "Chú thích:"]



def filter_for_alignment(text: str) -> str:
    """
    Lọc sạch văn bản tiếng Việt sau OCR/LLM để phục vụ tối ưu cho Sentence Alignment:
    - Xóa chỉ số cước chú gắn liền trong câu (ví dụ: 'phần¹' -> 'phần', 'sách[1]' -> 'sách').
    - Loại bỏ ký tự rác/nhiễu OCR (như | \ ~ ^ @ # $ % & * + = < > { } _ • ► ▪ § do quét giấy cũ/ố vàng/viền).
    - Chuẩn hóa khoảng trắng thừa trong mỗi dòng.
    - Loại bỏ các dòng trống hoặc dòng rác không có nghĩa tiếng Việt.
    """
    if not text:
        return ""

    lines = text.split("\n")
    inline_citation_regex = re.compile(
        r"[¹²³⁴⁵⁶⁷⁸⁹⁰†‡]+|(?<=[a-zA-Z\u00c0-\u024f\u1e00-\u1eff])[\(\[\{]\d+[\)\]\}]"
    )
    # Whitelist CHỈ GIỮ LẠI chữ cái Tiếng Việt Quốc Ngữ/Latin, số, dấu câu chuẩn ngữ pháp và khoảng trắng.
    # LOẠI BỎ HOÀN TOÀN Chữ Hán (CJK - \u3400-\u9FFF), chữ nước ngoài khác, và mọi symbol/rác OCR.
    strict_vietnamese_regex = re.compile(
        r"[^a-zA-Z0-9\u00c0-\u024f\u1e00-\u1eff\u0300-\u036f\s\.,:;?!\-–—\(\)\[\]\"'“”‘’/]",
        flags=re.UNICODE
    )

    clean_lines = []
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        line_str = inline_citation_regex.sub("", line_str)
        line_str = strict_vietnamese_regex.sub(" ", line_str)
        # Xóa các ngoặc rỗng sót lại sau khi loại bỏ Chữ Hán bên trong (ví dụ: "( )" hay "[ ]")
        line_str = re.sub(r"[\(\[\{]\s*[\)\]\}]", "", line_str)
        line_str = re.sub(r"\s+", " ", line_str).strip()
        if line_str and not _is_noise_line(line_str):
            clean_lines.append(line_str)

    return "\n".join(clean_lines)


def _is_noise_line(s: str) -> bool:
    """Kiểm tra nhanh xem dòng s có rõ ràng là rác số trang / ký tự lạc / nhiễu OCR không."""
    if not s or len(s) <= 1:
        return True
    if re.fullmatch(r'[\d\s/.,\-–\u2014:]+', s):
        return True
    # Kiểm tra số lượng chữ cái thực sự trong dòng (chữ tiếng Việt / Latin)
    letters = re.findall(r'[a-zA-Z\u00c0-\u024f\u1e00-\u1eff]', s)
    if len(letters) < 2:
        return True
    words = s.split()
    if len(words) >= 6:
        from collections import Counter
        most_common_count = Counter(words).most_common(1)[0][1]
        if most_common_count / len(words) >= 0.5:
            return True
    return False


def call_llm_api(system_prompt: str, user_prompt: str) -> str | None:
    """Gọi OpenAI-compatible API, retry tối đa LLM_MAX_RETRIES lần (cho lỗi mạng/timeout/API)."""
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
                content = re.sub(r"^```[a-zA-Z]*\r?\n?(.*?)\r?\n?```$", r"\1", content, flags=re.DOTALL).strip()
                return content
            print(f"  API {resp.status_code}: {resp.text[:150]} (retry {attempt + 1}/{LLM_MAX_RETRIES})")
        except requests.exceptions.Timeout:
            print(f"  Timeout {LLM_TIMEOUT}s (retry {attempt + 1}/{LLM_MAX_RETRIES})")
        except Exception as e:
            print(f"  Lỗi: {e} (retry {attempt + 1}/{LLM_MAX_RETRIES})")

        if attempt < LLM_MAX_RETRIES - 1:
            time.sleep(2 ** attempt)

    return None


def _build_user_prompt(work_title: str, chunk_lines: list[str], context: str) -> str:
    """Build prompt đưa văn bản OCR cần biên tập cho LLM."""
    ctx_text = context if context else "(Đây là phần đầu tác phẩm)"
    raw_text = "\n".join(chunk_lines)
    return (
        f'Tác phẩm: "{work_title}"\n'
        f"Bối cảnh đoạn trước (chỉ để tham khảo mạch văn, KHÔNG sửa lại đoạn này):\n{ctx_text}\n\n"
        f"VĂN BẢN OCR CẦN BIÊN TẬP (Chú ý: Sửa chính tả/lỗi dấu và nối câu mạch lạc, TUYỆT ĐỐI KHÔNG TÓM TẮT hay tự ý viết lại lời văn của tác giả):\n{raw_text}\n"
    )


def _correct_chunk(work_title: str, chunk_lines: list[str], context: str, chunk_no: int, total_chunks: int) -> list[str]:
    """
    Gọi LLM sửa 1 chunk văn bản. Không bắt buộc giữ nguyên số dòng,
    tập trung vào chất lượng, độ mạch lạc và ngữ pháp chuẩn xác.
    """
    print(f"  LLM chunk {chunk_no}/{total_chunks}...", end=" ", flush=True)

    for attempt in range(LLM_MAX_RETRIES):
        corrected = call_llm_api(SYSTEM_PROMPT, _build_user_prompt(work_title, chunk_lines, context))

        if corrected is not None:
            raw_lines = [l.strip() for l in corrected.strip().split("\n")]
            # Thử tự động gỡ bỏ số thứ tự '1. ', '2. ' nếu mô hình vô tình trả về theo thói quen cũ
            cleaned = [re.sub(r'^\s*[\(\[]?\d+[\)\]\.:]\s*', '', l).strip() for l in raw_lines]
            final_lines = [l for l in cleaned if l and not _is_noise_line(l)]
            print("OK")
            return final_lines

        print(f"[API failed, retry {attempt + 1}/{LLM_MAX_RETRIES}]", end=" ", flush=True)

    print("FAILED (giữ gốc)")
    return [l.strip() for l in chunk_lines if l.strip() and not _is_noise_line(l.strip())]


def correct_text_with_llm(full_text: str, work_title: str, language: str = "vie") -> str:
    """
    Chia text thành chunks (có context tham khảo từ đoạn trước), gửi cho LLM sửa lỗi OCR
    và khôi phục văn bản tiếng Việt tường minh, mạch lạc để phục vụ Sentence Alignment.
    """
    if not full_text.strip():
        return full_text

    lines = [l.strip() for l in full_text.split("\n") if l.strip()]
    stride = max(1, LLM_CHUNK_LINES)

    chunks = [
        (
            lines[i : i + LLM_CHUNK_LINES],
            "\n".join(lines[max(0, i - LLM_OVERLAP_LINES) : i]) if i > 0 else "",
        )
        for i in range(0, len(lines), stride)
    ]

    result_lines: list[str] = []
    for idx, (chunk_lines, context) in enumerate(chunks):
        if not chunk_lines:
            continue

        corrected_lines = _correct_chunk(
            work_title, chunk_lines, context, idx + 1, len(chunks)
        )
        result_lines.extend(corrected_lines)

    merged_text = "\n".join(result_lines)
    return filter_for_alignment(merged_text)
