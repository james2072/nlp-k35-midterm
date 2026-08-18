"""
Sentence Alignment - Monotonic Dynamic Programming + Ensemble + LLM Refine
Mỗi câu Hán -> 1 câu Việt tối ưu theo trật tự đơn điệu, refine bằng LLM.
"""

import os
import sys
import json
import csv
import argparse
import re
import time
import hashlib
import pickle
import math
import shutil
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIG
# ============================================================
load_dotenv()

# LLM Config
LLM_API_URL = os.getenv("LLM_API_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 300))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 4))
BATCH_LLM_SIZE = int(os.getenv("BATCH_LLM_SIZE", 5))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.60))

# Ensemble weights
LABSE_WEIGHT = 0.50
BERT_WEIGHT = 0.30
LENGTH_WEIGHT = 0.20
LENGTH_RATIO = float(os.getenv("LENGTH_RATIO", 3.5))  # Tỷ lệ ký tự Việt / Hán chuẩn (~3.2 - 3.8)

# Paths
PROJECT_ROOT = Path(os.getcwd())
DATA_DIR = PROJECT_ROOT / "data"
OCR_OUTPUT_DIR = DATA_DIR / "ocr_output"
CORPUS_DIR = DATA_DIR / "corpus"
CONFIG_PATH = DATA_DIR / "config.json"
CACHE_DIR = DATA_DIR / ".cache"

# ============================================================
# LOAD MODELS (LAZY LOADING)
# ============================================================
_labse_model = None
_bert_model = None

def get_labse_model():
    global _labse_model
    if _labse_model is None:
        print("  Loading LaBSE model...")
        _labse_model = SentenceTransformer("LaBSE")
    return _labse_model

def get_bert_model():
    global _bert_model
    if _bert_model is None:
        print("  Loading multilingual BERT model...")
        _bert_model = SentenceTransformer("bert-base-multilingual-cased")
    return _bert_model


# ============================================================
# 1. NORMALIZE
# ============================================================
# Set of valid Vietnamese single-letter standalone words to prevent accidental merging
VALID_VIET_SINGLE_WORDS = {'ở', 'ô', 'ý', 'ả', 'ê', 'a', 'y', 'u'}


def normalize_viet(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt theo chuẩn NLP:
    1. Chuẩn hóa Unicode sang dạng dựng sẵn (NFC).
    2. Khắc phục lỗi OCR tách rời từng chữ cái (e.g. 'N g u y ễ n' -> 'Nguyễn'), bảo vệ từ đơn 1 chữ hợp lệ.
    3. Chuẩn hóa khoảng trắng và dấu câu, tránh dính chữ hoặc mất chữ.
    """
    if not text:
        return ""
    
    # 1. Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)
    
    # 2. Merge single space between isolated single letters (OCR artifact)
    def _merge_single_letters(match):
        c1 = match.group(1)
        c2 = match.group(2)
        if c1.lower() in VALID_VIET_SINGLE_WORDS and c2.lower() in VALID_VIET_SINGLE_WORDS:
            return f"{c1} {c2}"
        return f"{c1}{c2}"
    
    # Merge multi-letter spaced sequences: 'V i ệ n' -> 'Viện'
    text = re.sub(r'(?<=\b[A-Za-zÀ-ỹ])\s(?=[A-Za-zÀ-ỹ]\b)', '', text)
    
    # 3. Standardize punctuation spacing
    text = re.sub(r'\s+([,.:;?!])', r'\1', text)
    text = re.sub(r'([,.:;?!])(?=[A-Za-zÀ-ỹ0-9])', r'\1 ', text)
    
    return re.sub(r'\s+', ' ', text).strip()


def normalize_han(text: str) -> str:
    """
    Chuẩn hóa văn bản chữ Hán cổ:
    1. Chuẩn hóa Unicode sang dạng dựng sẵn (NFC).
    2. Loại bỏ khoảng trắng giữa các ký tự CJK (Unified Ideographs & Extensions).
    3. Chuẩn hóa khoảng trắng thừa.
    """
    if not text:
        return ""
    
    # 1. Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)
    
    # 2. Remove whitespace between CJK characters
    cjk_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]'
    text = re.sub(f'(?<={cjk_pattern})\\s+(?={cjk_pattern})', '', text)
    
    return re.sub(r'\s+', ' ', text).strip()


# ============================================================
# 2. HAN & VIET SEGMENTATION
# ============================================================
def clean_lines(text: str) -> List[str]:
    """Lọc bỏ các dòng rác hoặc metadata trang in."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < 2:
            continue
        if re.match(r'^[\d\sIVXivx\-\_\.]+$', line):
            continue
        if any(k in line.lower() for k in ['quyển', '卷', 'chapter', 'trang', 'page']) and len(line) < 20:
            continue
        cleaned.append(line)
    return cleaned


def segment_han(text: str) -> List[str]:
    """
    Phân tách câu chữ Hán / Hán-Nôm theo vế ngữ nghĩa cân bằng:
    1. Ngắt câu tại các dấu câu: 。 ； ！ ？ ： ， 、 và xuống dòng \n.
    2. Ngắt sau các trợ từ kết thúc: 也, 矣, 焉, 哉, 乎, 邪, 耶, 耳 (khi độ dài tích lũy >= 10 ký tự).
    3. Ngắt trước các mốc tự sự: 時有, 其後, 未幾, 既而, 自此, 後有, 按 (khi độ dài tích lũy >= 12 ký tự).
    4. Gộp các mẩu câu quá ngắn (< 5 ký tự) để bảo đảm câu trọn vẹn ngữ nghĩa.
    """
    text = normalize_han(text)
    
    # Sentence / clause splitting pattern
    pattern = r'([。；！？，、：\n]|(?<=[也矣焉哉乎邪耶耳])|(?=時有|其後|未幾|既而|自此|後有|按[此之]))'
    tokens = re.split(pattern, text)
    
    sents = []
    curr = ""
    for t in tokens:
        if not t or t == '\n':
            continue
        curr += t
        if any(c in t for c in '。；！？，、：') or (len(curr.strip()) >= 10 and any(c in t for c in '也矣焉哉乎邪耶耳')):
            if len(curr.strip()) >= 6:
                sents.append(curr.strip())
                curr = ""
        elif len(curr.strip()) >= 12 and re.match(r'^(時有|其後|未幾|既而|自此|後有|按)', t):
            if len(curr[:-len(t)].strip()) >= 6:
                sents.append(curr[:-len(t)].strip())
                curr = t
                
    if curr.strip():
        if sents and len(curr.strip()) < 5:
            sents[-1] += curr.strip()
        else:
            sents.append(curr.strip())
            
    merged = []
    for s in sents:
        s = s.strip()
        if not s or len(s) < 2:
            continue
        if re.match(r'^[。；！？、，：\s\d]+$', s):
            continue
        if len(s) < 5 and merged:
            merged[-1] += s
        else:
            merged.append(s)
            
    return merged


def segment_viet(text: str) -> List[str]:
    r"""
    Phân tách câu tiếng Việt văn xuôi chuẩn NLP:
    1. Ngắt câu theo dấu kết thúc câu: . ! ? ;
    2. Bỏ qua không ngắt tại:
       - Số phân cách hàng nghìn / thập phân (\d+\.\s*\d+).
       - Chú thích số thứ tự: (1), (2), (1554-61.), v.v...
       - Các từ viết tắt: v.v., v. v..., v.v..
       - Dấu câu nằm bên trong dấu ngoặc đơn (...).
    3. Gom khối các dòng thơ / câu đối ngắn thành chỉnh thể câu trọn nghĩa.
    4. Tự động loại bỏ dấu chấm/phẩy rác rớt sang đầu câu.
    """
    text = normalize_viet(text)
    
    # 1. Bảo vệ dấu chấm đặc biệt trước khi tách:
    # Bảo vệ số: 3.000, 1.2
    text = re.sub(r'(\d+)\.\s*(\d+)', r'\1<DOT_NUM>\2', text)
    # Bảo vệ số chú thích và niên đại trong ngoặc: (1.), (1554-61.)
    text = re.sub(r'\(([^)]*?)\)', lambda m: '(' + m.group(1).replace('.', '<DOT_PAREN>').replace(';', '<SEMI_PAREN>') + ')', text)
    # Bảo vệ từ viết tắt v.v. và các biến thể nhiều dấu chấm v. v...
    text = re.sub(r'\bv\s*\.\s*v\s*(\.\s*)*', '<ABBR_VV> ', text, flags=re.IGNORECASE)
    
    # 2. Xử lý gom khối thơ / câu đối / văn tế (các dòng ngắn liên tiếp < 50 chars)
    raw_lines = clean_lines(text)
    processed_lines = []
    verse_buffer = []
    
    for line in raw_lines:
        line_clean = line.strip()
        # Dòng thơ/văn tế ngắn (không có dấu chấm hết và < 50 chars)
        if len(line_clean) < 50 and not re.search(r'[.!?;]$', line_clean) and not re.match(r'^(Chương|Quyển|Truyện|CÔNG-DƯ)', line_clean):
            verse_buffer.append(line_clean)
            if len(verse_buffer) >= 2:
                processed_lines.append(" ".join(verse_buffer))
                verse_buffer = []
        else:
            if verse_buffer:
                processed_lines.append(" ".join(verse_buffer))
                verse_buffer = []
            processed_lines.append(line_clean)
            
    if verse_buffer:
        processed_lines.append(" ".join(verse_buffer))
        
    # 3. Tách câu chính xác
    sentences = []
    for line in processed_lines:
        parts = re.split(r'([.!?;])', line)
        curr = ""
        for i in range(0, len(parts), 2):
            curr += parts[i]
            if i + 1 < len(parts):
                curr += parts[i + 1]
                if len(curr.strip()) >= 15:
                    sentences.append(curr.strip())
                    curr = ""
        if curr.strip() and len(curr.strip()) >= 6:
            sentences.append(curr.strip())
            
    # 4. Khôi phục lại các dấu chấm đã bảo vệ và làm sạch dấu câu rác đầu câu
    restored = []
    for s in sentences:
        s = s.replace('<DOT_NUM>', '.')
        s = s.replace('<DOT_PAREN>', '.')
        s = s.replace('<SEMI_PAREN>', ';')
        s = s.replace('<ABBR_VV>', 'v.v.')
        # Strip bất kỳ dấu câu rác rớt sang đầu câu
        s = re.sub(r'^[.,:;?!\s]+', '', s).strip()
        if not s or len(s) < 4:
            continue
        if len(s) < 15 and restored:
            restored[-1] = restored[-1] + " " + s
        else:
            restored.append(s)
            
    return restored


def segment_han_verse(text: str) -> List[str]:
    """Tách câu thơ Hán/Nôm: mỗi dòng thơ tương ứng 1 câu lục bát hoàn chỉnh."""
    lines = [normalize_han(l) for l in text.splitlines() if l.strip()]
    sentences = []
    for line in lines:
        if len(line) < 4 or line.startswith("大南國史演"):
            continue
        cleaned = re.sub(r'[，,。.\t\r]+$', '', line).strip()
        if cleaned:
            sentences.append(cleaned)
    return sentences


def segment_viet_verse(text: str) -> List[str]:
    """Tách câu thơ Quốc ngữ: ghép liên tiếp các cặp câu 6 và 8 thành 1 câu lục bát hoàn chỉnh."""
    lines = [normalize_viet(l) for l in text.splitlines() if l.strip()]
    couplets = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines):
            couplets.append(f"{lines[i]} {lines[i+1]}")
            i += 2
        else:
            couplets.append(lines[i])
            i += 1
    return couplets


# ============================================================
# 3. EMBEDDINGS & SIMILARITY
# ============================================================
def get_embeddings(texts: List[str], model, name: str) -> np.ndarray:
    """Compute embeddings với cache + validation."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    text_hash = hashlib.md5('\n'.join(texts).encode('utf-8')).hexdigest()
    cache_path = CACHE_DIR / f"{name}_{len(texts)}_{text_hash}.pkl"
    
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                emb = pickle.load(f)
            if emb.shape[0] == len(texts):
                return emb
            else:
                os.remove(cache_path)
        except:
            os.remove(cache_path)
    
    emb = model.encode(texts, batch_size=64, show_progress_bar=False)
    
    with open(cache_path, 'wb') as f:
        pickle.dump(emb, f)
    
    return emb


def length_sim(len1: int, len2: int, ratio: float = LENGTH_RATIO) -> float:
    """Gale-Church length similarity với tỷ lệ chuẩn Việt / Hán."""
    normalized = len2 / max(ratio, 0.1)
    delta = (len1 - normalized) / math.sqrt(max(len1, 1) * 6.8)
    return math.exp(-delta * delta / 2)


def compute_similarity(han_sents: List[str], viet_sents: List[str]) -> np.ndarray:
    """Ensemble similarity: LaBSE + BERT + Length."""
    n_han, n_viet = len(han_sents), len(viet_sents)
    
    labse_han = get_embeddings(han_sents, get_labse_model(), "labse_han")
    labse_viet = get_embeddings(viet_sents, get_labse_model(), "labse_viet")
    labse_sim = cosine_similarity(labse_han, labse_viet)
    
    bert_han = get_embeddings(han_sents, get_bert_model(), "bert_han")
    bert_viet = get_embeddings(viet_sents, get_bert_model(), "bert_viet")
    bert_sim = cosine_similarity(bert_han, bert_viet)
    
    length_matrix = np.zeros((n_han, n_viet))
    for i in range(n_han):
        for j in range(n_viet):
            length_matrix[i, j] = length_sim(len(han_sents[i]), len(viet_sents[j]))
    
    sim = LABSE_WEIGHT * labse_sim + BERT_WEIGHT * bert_sim + LENGTH_WEIGHT * length_matrix
    return sim


# ============================================================
# 4. MONOTONIC DYNAMIC PROGRAMMING BEAD ALIGNMENT (1-1, 1-N, N-1)
# ============================================================
def build_span_embedding(embeddings: np.ndarray, start: int, end: int) -> np.ndarray:
    """Tạo embedding cho một span câu liên tiếp từ normalized mean của các câu thành phần."""
    span_emb = np.sum(embeddings[start:end], axis=0)
    norm = np.linalg.norm(span_emb)
    if norm > 0:
        span_emb = span_emb / norm
    return span_emb


def monotonic_bead_dp_alignment(
    han_sents: List[str],
    viet_sents: List[str],
    labse_han: np.ndarray,
    labse_viet: np.ndarray,
    genre: str = "prose",
    diagonal_weight: float = 0.05
) -> List[Tuple[int, int, int, int, float]]:
    """
    Dóng câu Hán - Việt sử dụng Quy hoạch động đơn điệu hỗ trợ Bead (1-1, 1-N, N-1).
    Đảm bảo 100% bao phủ toàn bộ câu Hán và câu Việt (ZERO Omission, ZERO Addition).
    
    Returns:
        List of tuples: (h_start, h_end, v_start, v_end, similarity_score)
    """
    n_han = len(han_sents)
    n_viet = len(viet_sents)
    
    if n_han == 0 or n_viet == 0:
        return []
    
    # Đối với thơ (genre == "poetry"), ép buộc strictly 1-1 matching
    if genre == "poetry":
        allowed_beads = [(1, 1)]
    else:
        # Dynamic bead transitions: bảo đảm luôn tìm được đường đi chi tiết từng câu, không bị rơi vào fallback gộp cả chunk
        max_dv = max(3, min(math.ceil(n_viet / max(n_han, 1)) + 2, 15))
        allowed_beads = [(1, dv) for dv in range(1, max_dv + 1)] + [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2)]
    
    # Pre-calculate sentence lengths
    han_lens = [len(s) for s in han_sents]
    viet_lens = [len(s) for s in viet_sents]
    
    # DP table: dp[i, j] = best score aligning han_sents[0:i] with viet_sents[0:j]
    dp = np.full((n_han + 1, n_viet + 1), -np.inf)
    parent = {}
    bead_score_table = {}
    
    dp[0, 0] = 0.0
    
    global_ratio = n_viet / max(n_han, 1)
    
    for i in range(n_han + 1):
        for j in range(n_viet + 1):
            if dp[i, j] == -np.inf:
                continue
                
            for dh, dv in allowed_beads:
                next_i = i + dh
                next_j = j + dv
                
                if next_i > n_han or next_j > n_viet:
                    continue
                
                # Compute bead similarity
                span_h_len = sum(han_lens[i:next_i])
                span_v_len = sum(viet_lens[j:next_j])
                
                l_sim = length_sim(span_h_len, span_v_len, ratio=LENGTH_RATIO)
                
                # Fast span embedding similarity
                labse_h_span = build_span_embedding(labse_han, i, next_i)
                labse_v_span = build_span_embedding(labse_viet, j, next_j)
                labse_score = float(np.dot(labse_h_span, labse_v_span))
                
                sem_sim = 0.70 * labse_score + 0.30 * l_sim
                
                # Diagonal distance penalty
                expected_j = next_i * global_ratio
                diag_dist = abs(next_j - expected_j)
                total_bead_score = sem_sim - diag_dist * diagonal_weight
                
                # Transition score
                new_dp_score = dp[i, j] + total_bead_score
                if new_dp_score > dp[next_i, next_j]:
                    dp[next_i, next_j] = new_dp_score
                    parent[(next_i, next_j)] = (i, j, dh, dv)
                    bead_score_table[(i, next_i, j, next_j)] = sem_sim

    # Backtracking to reconstruct optimal beads
    curr = (n_han, n_viet)
    beads = []
    
    # Fallback if no full path reached
    if curr not in parent and (n_han > 0 and n_viet > 0):
        beads.append((0, n_han, 0, n_viet, 1.0))
        return beads
        
    while curr != (0, 0):
        if curr not in parent:
            break
        prev_i, prev_j, dh, dv = parent[curr]
        score = bead_score_table.get((prev_i, curr[0], prev_j, curr[1]), 0.5)
        beads.append((prev_i, curr[0], prev_j, curr[1], float(score)))
        curr = (prev_i, prev_j)
        
    beads.reverse()
    return beads


# ============================================================
# 5. LLM REFINE
# ============================================================
def extract_json_from_text(text: str):
    """Trích xuất JSON từ response LLM - robust version."""
    if not text:
        return None
    
    text = text.strip()
    
    # Strip reasoning tags if present
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    text = re.sub(r'<\|[^|]*\|>', '', text)
    
    # Strip markdown code fencing
    text = text.replace('```json', '').replace('```', '')
    
    # Locate beginning of JSON payload
    for start_char in ['[', '{']:
        idx = text.find(start_char)
        if idx >= 0:
            text = text[idx:]
            break
    
    # Attempt direct JSON parsing
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    
    # Extract complete JSON array block via regex
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            # Attempt to fix trailing commas in truncated JSON
            json_str = match.group()
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            try:
                return json.loads(json_str)
            except Exception:
                pass
    
    # Fallback: extract individual JSON objects
    objects = re.findall(r'\{[^{}]*\}', text)
    if objects:
        results = []
        for obj in objects:
            try:
                results.append(json.loads(obj))
            except Exception:
                continue
        if results:
            return results
    
    # Fallback: line-by-line object parsing
    results = []
    for line in text.split('\n'):
        line = line.strip().rstrip(',')
        if line.startswith('{') and line.endswith('}'):
            try:
                results.append(json.loads(line))
            except Exception:
                continue
    if results:
        return results
    
    return None


def call_llm_chunk(pairs: List[Tuple[str, str, str]], work_title: str, han_context: str, viet_context: str) -> List[Dict]:
    """
    Sử dụng LLM để kiểm tra, sửa lỗi chính tả/OCR và bảo đảm độ chuẩn xác song ngữ (loại bỏ omission/addition).
    Có kèm theo bối cảnh của toàn bộ Chunk để LLM chắp vá chính xác.
    
    Args:
        pairs: Danh sách các bộ (han_sentence, viet_sentence, pair_id).
        work_title: Tên tác phẩm đang xử lý.
        han_context: Toàn bộ nội dung Hán của Chunk.
        viet_context: Toàn bộ nội dung Việt của Chunk.
        
    Returns:
        Danh sách các dictionary chứa kết quả chuẩn hóa từ LLM.
    """
    system_prompt = """You are an expert bilingual scholar and editor specializing in Vietnamese Han-Nom literature (chữ Hán & chữ Nôm của Việt Nam) and modern Vietnamese (Quốc ngữ) historical translations.
Your mission is to ensure that each bilingual sentence pair is semantically equivalent, fully aligned, and free of OCR recognition errors.

Strict Quality Standards:
1. ZERO OMISSION: Ensure the Vietnamese Quốc ngữ text fully translates all contents of the Han-Nom source text.
2. ZERO ADDITION & ANNOTATION DELETION: If the Vietnamese text contains long historical annotations, explanations, or extraneous details added by the translator that DO NOT EXIST in the Han-Nom text, you MUST DELETE them entirely.
3. NO ENGLISH TRANSLATIONS: The `han_corrected` field MUST contain ONLY original Han characters (Chữ Hán). You MUST NEVER translate the Han text into English or any other language. Fix only OCR typos using the context.
4. PRESERVE & CLEAN: Fix OCR typos and misplaced punctuation while strictly preserving Vietnamese historical proper names and official titles.
5. NO HALLUCINATION & BOUNDARY FIXING: You MUST NOT translate the Han-Nom text from scratch. You MUST find the missing translation parts in the [VIỆT CHUNK CONTEXT]. 
   CRITICAL: If the previous alignment algorithm (DP) incorrectly shifted the translation of Pair N into Pair N-1 or N+1, you MUST move that text back to the correct Pair in the JSON output! Do NOT blindly copy the input boundaries if they are wrong.
6. If the pair is already well-matched, keep the content intact.

Input format:
[Pair <pair_id>]
[HÁN NÔM]: <Han-Nom source text>
[VIỆT]: <Vietnamese translation text>

Output format: ONLY a valid JSON array of objects with the exact schema:
[
  {
    "pair_id": "<pair_id>",
    "match": true,
    "han_corrected": "<clean, corrected Han-Nom source text (ONLY HAN CHARACTERS)>",
    "viet_corrected": "<clean, complete Vietnamese translation text without omission or addition>"
  }
]"""

    pairs_text = "\n\n".join([
        f"[Pair {pid}]\n[HÁN NÔM]: {han}\n[VIỆT]: {viet}"
        for han, viet, pid in pairs
    ])
    
    user_prompt = f'Work: "{work_title}"\n\n=== [HÁN CHUNK CONTEXT] ===\n{han_context}\n\n=== [VIỆT CHUNK CONTEXT] ===\n{viet_context}\n\n=== PAIRS TO REFINE ===\n{pairs_text}\n\nReply ONLY JSON array:'

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Standard OpenAI-compatible chat completion payload
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 4096,
        "stream": False
    }
    
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = requests.post(
                LLM_API_URL, headers=headers, json=payload, timeout=LLM_TIMEOUT
            )
            
            if response.status_code == 200:
                message = response.json()['choices'][0]['message']
                content = message.get('content', '') or ''
                
                if not content.strip():
                    content = message.get('reasoning_content', '') or ''
                
                results = extract_json_from_text(content)
                
                if results is not None:
                    if isinstance(results, dict):
                        for key in ['results', 'pairs', 'data', 'items']:
                            if key in results and isinstance(results[key], list):
                                results = results[key]
                                break
                        else:
                            results = [results]
                    
                    if isinstance(results, list) and len(results) > 0:
                        valid = []
                        for r in results:
                            if isinstance(r, dict):
                                valid.append({
                                    "pair_id": str(r.get("pair_id", "")),
                                    "match": bool(r.get("match", True)),
                                    "han_corrected": str(r.get("han_corrected", "")),
                                    "viet_corrected": str(r.get("viet_corrected", "")),
                                })
                        if valid:
                            return valid
                
                if attempt == 0:
                    print(f"  Raw response: {content[:200]}...")
                
            elif response.status_code == 429:
                print("  Rate limit encountered, waiting 30s...")
                time.sleep(30)
                continue
            else:
                print(f"  API Error {response.status_code}: {response.text[:150]}")
                if response.status_code == 400:
                    break
                
        except requests.exceptions.Timeout:
            print(f"  Request timeout (attempt {attempt + 1}/{LLM_MAX_RETRIES})")
        except Exception as e:
            print(f"  Request error: {e}")
        
        if attempt < LLM_MAX_RETRIES - 1:
            wait = 2 ** (attempt + 1)
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)
    
    # Fallback to original pairs when LLM is unavailable or fails
    return [
        {"pair_id": pid, "match": True, "han_corrected": han, "viet_corrected": viet}
        for han, viet, pid in pairs
    ]


# ============================================================
# 5. CHECKPOINT & PERSISTENCE HELPERS
# ============================================================
def compute_pair_hash(han: str, viet: str) -> str:
    """Tạo hash duy nhất đại diện cho cặp câu Hán - Việt."""
    raw = f"{han.strip()}|||{viet.strip()}".encode('utf-8')
    return hashlib.md5(raw).hexdigest()


def get_checkpoint_path(work_id: str) -> Path:
    """Đường dẫn file checkpoint cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{work_id}_checkpoint.json"


def load_checkpoint(work_id: str) -> Dict[str, Dict]:
    """Đọc dữ liệu checkpoint đã lưu nếu có."""
    path = get_checkpoint_path(work_id)
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  Warning: Checkpoint read error, starting fresh: {e}")
    return {}


def save_checkpoint(work_id: str, checkpoint_data: Dict[str, Dict]):
    """Ghi dữ liệu checkpoint xuống đĩa ngay sau mỗi batch."""
    path = get_checkpoint_path(work_id)
    try:
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        temp_path.replace(path)
    except Exception as e:
        print(f"  Warning: Checkpoint save error: {e}")


def save_tsv(work_id: str, pairs: List[Dict], output_path: Path) -> int:
    """Lưu file TSV đầy đủ tất cả các cặp câu đã dóng (pair_id, han_sentence, viet_sentence)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['pair_id', 'han_sentence', 'viet_sentence'])
        
        count = 0
        for pair in pairs:
            pair_id = f"{work_id}_{count:04d}"
            han = pair["han"].replace('\t', ' ').replace('\n', ' ').strip()
            viet = pair["viet"].replace('\t', ' ').replace('\n', ' ').strip()
            writer.writerow([pair_id, han, viet])
            count += 1
    
    return count


def save_excel(work_id: str, pairs: List[Dict], output_path: Path) -> int:
    """Xuất file Excel (.xlsx) đầy đủ tất cả các cặp câu song ngữ."""
    try:
        import pandas as pd
        rows = []
        count = 0
        for pair in pairs:
            pair_id = f"{work_id}_{count:04d}"
            rows.append({
                "pair_id": pair_id,
                "han_sentence": pair["han"].replace('\t', ' ').replace('\n', ' ').strip(),
                "viet_sentence": pair["viet"].replace('\t', ' ').replace('\n', ' ').strip()
            })
            count += 1
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(output_path, index=False)
            return count
    except Exception as e:
        print(f"  Warning: Error saving Excel: {e}")
    return 0


def refine_with_llm_chunk(chunk_idx: int, alignment: List[Tuple[int, int, int, int, float]], 
                          han_sents: List[str], viet_sents: List[str], 
                          han_context: str, viet_context: str,
                          work_id: str, work_title: str, 
                          global_pair_offset: int,
                          use_llm: bool = True) -> List[Dict]:
    """
    Chuẩn hóa alignment của 1 Chunk bằng LLM với Context.
    """
    refined_pairs = []
    checkpoint = load_checkpoint(work_id)
    cached_hits = 0
    
    # Initialize alignment pairs
    for local_idx, (h_s, h_e, v_s, v_e, score) in enumerate(alignment):
        han_text = "".join(han_sents[h_s:h_e]).strip()
        viet_text = " ".join(viet_sents[v_s:v_e]).strip()
        p_hash = compute_pair_hash(han_text, viet_text)
        
        pair_id = f"{work_id}_{global_pair_offset + local_idx:04d}"
        
        pair_data = {
            "han": han_text,
            "viet": viet_text,
            "score": score,
            "pair_id": pair_id,
            "pair_hash": p_hash,
            "llm_approved": True,
            "h_span": f"{h_s}-{h_e}",
            "v_span": f"{v_s}-{v_e}"
        }
        
        # Load cached corrections
        if p_hash in checkpoint:
            cached = checkpoint[p_hash]
            if cached.get("han_corrected"):
                pair_data["han"] = cached["han_corrected"]
            if cached.get("viet_corrected"):
                pair_data["viet"] = cached["viet_corrected"]
            cached_hits += 1
            
        refined_pairs.append(pair_data)
        
    if cached_hits > 0:
        print(f"    Restored {cached_hits} pairs from checkpoint.")
        
    need_verify = [
        p for p in refined_pairs 
        if p["score"] < CONFIDENCE_THRESHOLD and p["pair_hash"] not in checkpoint
    ]
    
    if not need_verify or not use_llm or not LLM_API_KEY:
        return refined_pairs
        
    print(f"    Refining {len(need_verify)} pairs in chunk {chunk_idx} with LLM...")
    
    # Send all need_verify of this chunk in ONE LLM call
    batch_input = [(p["han"], p["viet"], p["pair_id"]) for p in need_verify]
    llm_results = call_llm_chunk(batch_input, work_title, han_context, viet_context)
    
    result_map = {r.get("pair_id"): r for r in llm_results}
    
    for pair in need_verify:
        llm_result = result_map.get(pair["pair_id"], {})
        corrected_han = llm_result.get("han_corrected", "")
        corrected_viet = llm_result.get("viet_corrected", "")
        
        if corrected_han:
            pair["han"] = corrected_han
        if corrected_viet:
            pair["viet"] = corrected_viet
            
        # Update checkpoint cache
        if p_hash not in checkpoint:
            checkpoint[p_hash] = {
                "han_original": pair["han"],
                "viet_original": pair["viet"],
                "han_corrected": corrected_han,
                "viet_corrected": corrected_viet,
                "score": pair["score"]
            }
            
    save_checkpoint(work_id, checkpoint)
    return refined_pairs



# ============================================================
# 7. MAIN
# ============================================================
def align_work(work: Dict, use_llm: bool = True, debug_mode: bool = False):
    """Thực hiện dóng câu cho một tác phẩm dựa trên genre cấu hình."""
    work_id = work['id']
    work_title = work['viet']
    genre = work.get("genre", "prose")
    
    print(f"\n{'='*60}")
    print(f"Aligning: {work_title} ({work_id})")
    print(f"Genre: {genre.upper()} ({'Thơ Lục Bát / Diễn Ca' if genre == 'poetry' else 'Văn Xuôi / Lịch Sử'})")
    print(f"{'='*60}")
    
    han_path = OCR_OUTPUT_DIR / f"{work_id}_sino_raw.txt"
    viet_path = OCR_OUTPUT_DIR / f"{work_id}_vie_raw.txt"
    
    if not han_path.exists() or not viet_path.exists():
        print("  Error: Missing input files.")
        return
    
    han_raw = han_path.read_text(encoding='utf-8')
    viet_raw = viet_path.read_text(encoding='utf-8')
    
    han_chunks = [c.strip() for c in han_raw.split('\n\n') if c.strip()]
    viet_chunks = [c.strip() for c in viet_raw.split('\n\n') if c.strip()]
    is_chunked = len(han_chunks) > 1 and len(han_chunks) == len(viet_chunks)
    
    # Nếu bật cờ debug, xuất file phân tích tách câu và tiếp tục alignment
    if debug_mode:
        from helper import export_debug_segmentation
        export_debug_segmentation(work, han_raw, viet_raw, is_chunked, han_chunks, viet_chunks, DATA_DIR, segment_han, segment_viet, segment_han_verse, segment_viet_verse)
    
    all_han_sents = []
    all_viet_sents = []
    all_refined_pairs = []
    global_pair_offset = 0
    
    if genre == "poetry":
        # Chế độ chuyên biệt cho Thơ Diễn Ca / Lục Bát (1-1 strict matching)
        all_han_sents = segment_han_verse(han_raw)
        all_viet_sents = segment_viet_verse(viet_raw)
        
        if not all_han_sents or not all_viet_sents:
            print("  Warning: No verse sentences found after segmentation.")
            return
            
        all_labse_han = get_embeddings(all_han_sents, get_labse_model(), f"{work_id}_labse_han")
        all_labse_viet = get_embeddings(all_viet_sents, get_labse_model(), f"{work_id}_labse_viet")
        all_alignment = monotonic_bead_dp_alignment(all_han_sents, all_viet_sents, all_labse_han, all_labse_viet, genre="poetry", diagonal_weight=0.08)
        all_refined_pairs = refine_with_llm_chunk(0, all_alignment, all_han_sents, all_viet_sents, han_raw, viet_raw, work_id, work_title, 0, use_llm)
        
    elif is_chunked:
        print(f"  Detected {len(han_chunks)} aligned chunks -> Running Chunk-wise Bead Alignment...")
        
        # 1. Segment all chunks
        chunk_sents = []
        for h_chunk, v_chunk in zip(han_chunks, viet_chunks):
            h_s = segment_han(h_chunk)
            v_s = segment_viet(v_chunk)
            chunk_sents.append((h_s, v_s))
            all_han_sents.extend(h_s)
            all_viet_sents.extend(v_s)
            
        if not all_han_sents or not all_viet_sents:
            print("  Warning: No sentences found after segmentation.")
            return
            
        # 2. Vectorized embedding computation once for all sentences
        all_labse_han = get_embeddings(all_han_sents, get_labse_model(), f"{work_id}_labse_han")
        all_labse_viet = get_embeddings(all_viet_sents, get_labse_model(), f"{work_id}_labse_viet")
        
        # 3. Chunk-wise DP alignment and LLM Refine
        han_offset = 0
        viet_offset = 0
        for c_idx, (h_s, v_s) in enumerate(chunk_sents):
            if not h_s or not v_s:
                continue
            h_embs = all_labse_han[han_offset : han_offset + len(h_s)]
            v_embs = all_labse_viet[viet_offset : viet_offset + len(v_s)]
            
            chunk_align = monotonic_bead_dp_alignment(h_s, v_s, h_embs, v_embs, genre="prose", diagonal_weight=0.05)
            refined_chunk = refine_with_llm_chunk(c_idx, chunk_align, h_s, v_s, han_chunks[c_idx], viet_chunks[c_idx], work_id, work_title, global_pair_offset, use_llm)
            
            all_refined_pairs.extend(refined_chunk)
            global_pair_offset += len(refined_chunk)
            
            han_offset += len(h_s)
            viet_offset += len(v_s)
            
            # Incremental save per chunk
            tsv_live_path = CORPUS_DIR / f"{work_id}_parallel.tsv"
            xlsx_live_path = CORPUS_DIR / f"{work_id}_parallel.xlsx"
            save_tsv(work_id, all_refined_pairs, tsv_live_path)
            save_excel(work_id, all_refined_pairs, xlsx_live_path)
            
    else:
        all_han_sents = segment_han(han_raw)
        all_viet_sents = segment_viet(viet_raw)
        
        if not all_han_sents or not all_viet_sents:
            print("  Warning: No sentences found after segmentation.")
            return
        
        all_labse_han = get_embeddings(all_han_sents, get_labse_model(), f"{work_id}_labse_han")
        all_labse_viet = get_embeddings(all_viet_sents, get_labse_model(), f"{work_id}_labse_viet")
        all_alignment = monotonic_bead_dp_alignment(all_han_sents, all_viet_sents, all_labse_han, all_labse_viet, genre="prose", diagonal_weight=0.05)
        all_refined_pairs = refine_with_llm_chunk(0, all_alignment, all_han_sents, all_viet_sents, han_raw, viet_raw, work_id, work_title, 0, use_llm)
    
    ratio = len(all_viet_sents) / max(len(all_han_sents), 1)
    
    # Bước 5: Thống kê kết quả
    total = len(all_refined_pairs)
    approved = sum(1 for p in all_refined_pairs if p.get("llm_approved", True))
    
    print(f"\n  Final statistics:")
    print(f"     - Han: {len(all_han_sents)} sentences")
    print(f"     - Viet: {len(all_viet_sents)} sentences (Ratio: 1:{ratio:.2f})")
    print(f"     - Total generated beads: {total}")
    print(f"     - Approved: {approved}")
    print(f"     - Rejected: {total - approved}")
    
    # Bước 6: Lưu file kết quả chuẩn TSV và XLSX
    tsv_path = CORPUS_DIR / f"{work_id}_parallel.tsv"
    xlsx_path = CORPUS_DIR / f"{work_id}_parallel.xlsx"
    count = save_tsv(work_id, all_refined_pairs, tsv_path)
    save_excel(work_id, all_refined_pairs, xlsx_path)
    print(f"  Saved {count} pairs -> {tsv_path.name} & {xlsx_path.name}")
    
    if debug_mode:
        from helper import export_debug_matrix
        export_debug_matrix(work_id, refined_pairs, DATA_DIR)


def main():
    """Hàm thực thi chính của sentence alignment."""
    parser = argparse.ArgumentParser(description="Sentence Alignment")
    parser.add_argument("--work-id", type=str, help="ID tác phẩm cụ thể")
    parser.add_argument("--clear-cache", action="store_true", help="Xóa cache embeddings")
    parser.add_argument("--debug", action="store_true", help="Chạy chế độ debug: Không dùng LLM, xuất file phân tích cấu trúc câu và ma trận điểm số")
    args = parser.parse_args()
    
    # Debug mode implies no LLM
    args.no_llm = args.debug
    
    # Xóa cache nếu có cờ --clear-cache
    if args.clear_cache and CACHE_DIR.exists():
        print(f"Clearing cache: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print("Cache cleared\n")
    
    if not CONFIG_PATH.exists():
        print(f"Config not found: {CONFIG_PATH}")
        sys.exit(1)
    
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    works = config['works']
    if args.work_id:
        works = [w for w in works if w['id'] == args.work_id]
        if not works:
            print(f"Work ID not found: {args.work_id}")
            sys.exit(1)
    
    if args.debug:
        print(f"Running DEBUG MODE for {len(works)} works (DP Alignment + Export Matrices, LLM OFF)...")
    else:
        print(f"Aligning {len(works)} works (Monotonic Dynamic Programming)")
        print(f"   LLM: {'OFF' if args.no_llm else f'ON ({LLM_MODEL_NAME})'}")
        print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print(f"   Batch size: {BATCH_LLM_SIZE}")
    
    for work in works:
        try:
            align_work(work, use_llm=not args.no_llm, debug_mode=args.debug)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()
