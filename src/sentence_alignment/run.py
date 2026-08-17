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
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.40))

# Ensemble weights
LABSE_WEIGHT = 0.50
BERT_WEIGHT = 0.30
LENGTH_WEIGHT = 0.20
LENGTH_RATIO = float(os.getenv("LENGTH_RATIO", 2.8))  # Tỷ lệ ký tự Việt / Hán thực tế (~2.5 - 3.2)

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
    Phân tách câu chữ Hán cổ điển (Classical Chinese):
    1. Tách theo dấu câu chuẩn (。, ；, ！, ？, ，, 、).
    2. Với văn bản chưa chấm câu: ngắt theo hư từ kết thúc (矣, 也, 焉, 哉, 乎, 耳) hoặc liên từ mở đầu (乃, 遂, 因, 至, 曰...).
    3. Gộp các mẩu câu quá ngắn để bảo toàn ngữ nghĩa.
    """
    text = normalize_han(text)
    lines = clean_lines(text)
    
    raw_pieces = []
    for line in lines:
        # Segment by punctuation marks if available
        parts = re.split(r'([。；！？，、])', line)
        curr = ""
        for i in range(0, len(parts), 2):
            curr += parts[i]
            if i + 1 < len(parts):
                curr += parts[i + 1]
                if len(curr.strip()) >= 4:
                    raw_pieces.append(curr.strip())
                    curr = ""
        if curr.strip() and len(curr.strip()) >= 2:
            raw_pieces.append(curr.strip())
    
    # Process unpunctuated classical Chinese clauses
    final_sents = []
    for p in raw_pieces:
        if len(p) > 28:
            # Split after final particles or before clause-initial conjunctions
            sub = re.split(r'(?<=[矣也焉哉乎耳])|(?=[至乃遂因俄忽]|時有|其後|又|及|初|後|公曰|帝曰|曰)', p)
            curr_sub = ""
            for s in sub:
                if not s:
                    continue
                curr_sub += s
                if len(curr_sub) >= 12:
                    final_sents.append(curr_sub.strip())
                    curr_sub = ""
            if curr_sub.strip():
                if final_sents and len(curr_sub.strip()) < 5:
                    final_sents[-1] += curr_sub.strip()
                else:
                    final_sents.append(curr_sub.strip())
        else:
            final_sents.append(p)
    
    # Merge short fragments (< 3 characters)
    merged = []
    for seg in final_sents:
        if len(seg) < 3 and merged:
            merged[-1] = merged[-1] + seg
        else:
            merged.append(seg)
    
    # Filter out empty or punctuation-only segments
    result = []
    for seg in merged:
        seg = seg.strip()
        if not seg or len(seg) < 2:
            continue
        if re.match(r'^[。；！？、，\s\d]+$', seg):
            continue
        result.append(seg)
    
    return result


def segment_viet(text: str) -> List[str]:
    """Phân tách câu tiếng Việt theo dấu ngắt câu chuẩn."""
    text = normalize_viet(text)
    lines = clean_lines(text)
    
    sentences = []
    for line in lines:
        parts = re.split(r'([.!?;])', line)
        current = ""
        for i in range(0, len(parts), 2):
            current += parts[i]
            if i + 1 < len(parts):
                current += parts[i + 1]
                if len(current.strip()) >= 15:
                    sentences.append(current.strip())
                    current = ""
        if current.strip() and len(current.strip()) >= 10:
            sentences.append(current.strip())
    
    # Merge short fragments (< 15 characters)
    merged = []
    for s in sentences:
        if len(s) < 15 and merged:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    
    return merged


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
# 4. MONOTONIC DYNAMIC PROGRAMMING ALIGNMENT
# ============================================================
def monotonic_dp_alignment(sim_matrix: np.ndarray, 
                           diagonal_weight: float = 0.05) -> List[Tuple[int, int, float]]:
    """
    Dóng câu Hán - Việt sử dụng Quy hoạch động đơn điệu (Monotonic Dynamic Programming).
    
    Thuật toán đảm bảo 100% trật tự thời gian (Monotonicity):
    - Tìm đường đi đơn điệu từ (0, 0) đến (N-1, M-1) tối đa hóa độ tương đồng ngữ nghĩa.
    - Ép buộc thứ tự: nếu câu Hán i dóng với câu Việt j, thì câu Hán i+1 chỉ được dóng với câu Việt k (k >= j).
    - Phạt độ lệch khỏi đường chéo tỷ lệ để tránh trôi lệch quá xa khỏi vị trí tương đối.
    
    Args:
        sim_matrix: Ma trận tương đồng kích thước (N, M) giữa câu Hán và câu Việt.
        diagonal_weight: Hệ số phạt khoảng cách đường chéo (diagonal penalty).
        
    Returns:
        Danh sách các bộ (han_idx, viet_idx, similarity_score) đã sắp xếp theo thứ tự câu Hán.
    """
    n_han, n_viet = sim_matrix.shape
    if n_han == 0 or n_viet == 0:
        return []
    
    ratio = n_viet / max(n_han, 1)
    
    # Bảng quy hoạch động và bảng vết (backtracking)
    dp = np.full((n_han, n_viet), -np.inf)
    parent = np.full((n_han, n_viet), -1, dtype=int)
    
    # Bước khởi tạo (Hán câu đầu tiên i = 0)
    for j in range(n_viet):
        diag_dist = abs(j - 0.0)
        dp[0, j] = sim_matrix[0, j] - diag_dist * diagonal_weight
    
    # Lan truyền quy hoạch động thuận (Forward DP)
    for i in range(1, n_han):
        expected_j = i * ratio
        for j in range(n_viet):
            diag_dist = abs(j - expected_j)
            curr_score = sim_matrix[i, j] - diag_dist * diagonal_weight
            
            best_prev_score = -np.inf
            best_prev_j = -1
            
            # Scan valid previous states (prev_j <= j ensures strict monotonicity)
            search_start = max(0, int(j - ratio * 3 - 4))
            for prev_j in range(search_start, j + 1):
                prev_val = dp[i - 1, prev_j]
                # Apply step penalty if staying on the exact same Vietnamese sentence
                if prev_j == j:
                    prev_val -= 0.15
                if prev_val > best_prev_score:
                    best_prev_score = prev_val
                    best_prev_j = prev_j
            
            if best_prev_j != -1 and best_prev_score > -np.inf:
                dp[i, j] = best_prev_score + curr_score
                parent[i, j] = best_prev_j
            else:
                best_prev_j = int(np.argmax(dp[i - 1]))
                dp[i, j] = dp[i - 1, best_prev_j] + curr_score
                parent[i, j] = best_prev_j
    
    # Truy vết ngược tìm đường đi tối ưu (Backtracking)
    best_last_j = int(np.argmax(dp[n_han - 1]))
    path = []
    curr_j = best_last_j
    
    for i in range(n_han - 1, -1, -1):
        score = float(sim_matrix[i, curr_j])
        path.append((i, curr_j, score))
        curr_j = parent[i, curr_j]
        if curr_j == -1 and i > 0:
            curr_j = max(0, int((i - 1) * ratio))
    
    path.reverse()
    return path


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


def call_llm_batch(pairs: List[Tuple[str, str, str]], work_title: str) -> List[Dict]:
    """
    Sử dụng LLM để kiểm tra, sửa lỗi chính tả/OCR và chuẩn hóa các cặp câu song ngữ.
    
    Args:
        pairs: Danh sách các bộ (han_sentence, viet_sentence, pair_id).
        work_title: Tên tác phẩm đang xử lý.
        
    Returns:
        Danh sách các dictionary chứa kết quả chuẩn hóa từ LLM.
    """
    system_prompt = """You are a Vietnamese and Classical Chinese (Han-Nom) linguistic editor.
Your job is to polish, fix OCR recognition errors, and standardize sentence pairs.

Input format:
[Pair temp_0]
[HÁN]: <Han sentence>
[VIỆT]: <Vietnamese sentence>

Output format: ONLY a valid JSON array of objects with the exact schema:
[
  {
    "pair_id": "temp_0",
    "match": true,
    "han_corrected": "<corrected Han text with OCR noise removed>",
    "viet_corrected": "<corrected Vietnamese text with clean accents and punctuation>"
  }
]

Guidelines:
1. Preserve the meaning of both sentences.
2. Fix OCR artifacts (e.g. isolated spaces between letters, misread characters, missing punctuation).
3. If the text is already correct, keep it as is.
4. Output strictly the JSON array, no explanation, no markdown tags."""

    pairs_text = "\n\n".join([
        f"[Pair {pid}]\n[HÁN]: {han}\n[VIỆT]: {viet}"
        for han, viet, pid in pairs
    ])
    
    user_prompt = f'Work: "{work_title}"\n\n{pairs_text}\n\nReply ONLY JSON array:'

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


def refine_with_llm(alignment: List[Tuple[int, int, float]], 
                    han_sents: List[str], viet_sents: List[str], 
                    work_id: str, work_title: str, 
                    use_llm: bool = True) -> List[Dict]:
    """
    Chuẩn hóa alignment bằng LLM (hỗ trợ Checkpoint, Incremental Save và Resume).
    - Bảo toàn 100% tất cả các cặp câu (fallback về bản gốc nếu LLM không sửa).
    - Lưu tức thời kết quả xuống đĩa sau mỗi batch hoàn thành.
    """
    refined_pairs = []
    checkpoint = load_checkpoint(work_id)
    cached_hits = 0
    
    # Initialize alignment pairs and restore state from checkpoint
    for idx, (han_idx, viet_idx, score) in enumerate(alignment):
        if viet_idx < 0:
            continue
        
        han_text = han_sents[han_idx]
        viet_text = viet_sents[viet_idx]
        p_hash = compute_pair_hash(han_text, viet_text)
        
        pair_data = {
            "han": han_text,
            "viet": viet_text,
            "score": score,
            "pair_id": f"temp_{idx}",
            "pair_hash": p_hash,
            "llm_approved": True
        }
        
        # Load cached corrections from checkpoint if present
        if p_hash in checkpoint:
            cached = checkpoint[p_hash]
            if cached.get("han_corrected"):
                pair_data["han"] = cached["han_corrected"]
            if cached.get("viet_corrected"):
                pair_data["viet"] = cached["viet_corrected"]
            cached_hits += 1
            
        refined_pairs.append(pair_data)
    
    tsv_live_path = CORPUS_DIR / f"{work_id}_parallel.tsv"
    xlsx_live_path = CORPUS_DIR / f"{work_id}_parallel.xlsx"
    
    # Save initial monotonic alignment state to disk
    save_tsv(work_id, refined_pairs, tsv_live_path)
    save_excel(work_id, refined_pairs, xlsx_live_path)
    
    if cached_hits > 0:
        print(f"  Restored {cached_hits} pairs from checkpoint.")
    
    # Filter pairs requiring LLM verification
    need_verify = [
        p for p in refined_pairs 
        if p["score"] < CONFIDENCE_THRESHOLD and p["pair_hash"] not in checkpoint
    ]
    
    if not need_verify:
        return refined_pairs
    
    if not use_llm:
        return refined_pairs
    
    if not LLM_API_KEY:
        print("  Error: LLM_API_KEY not found.")
        return refined_pairs
    
    # Execute LLM batch processing and persist incremental progress
    print(f"  Refining {len(need_verify)} pairs with LLM (batch={BATCH_LLM_SIZE})...")
    
    for i in tqdm(range(0, len(need_verify), BATCH_LLM_SIZE), desc="    LLM"):
        batch = need_verify[i:i+BATCH_LLM_SIZE]
        batch_input = [(p["han"], p["viet"], p["pair_id"]) for p in batch]
        
        llm_results = call_llm_batch(batch_input, work_title)
        result_map = {r.get("pair_id"): r for r in llm_results}
        
        for pair in batch:
            llm_result = result_map.get(pair["pair_id"], {})
            corrected_han = llm_result.get("han_corrected", "")
            corrected_viet = llm_result.get("viet_corrected", "")
            
            # Update sentence with verified/corrected text
            if corrected_han:
                pair["han"] = corrected_han
            if corrected_viet:
                pair["viet"] = corrected_viet
            
            # Update checkpoint dictionary
            checkpoint[pair["pair_hash"]] = {
                "match": llm_result.get("match", True),
                "han_corrected": pair["han"],
                "viet_corrected": pair["viet"],
                "score": pair["score"]
            }
        
        # Persist checkpoint and update TSV/Excel exports incrementally
        save_checkpoint(work_id, checkpoint)
        save_tsv(work_id, refined_pairs, tsv_live_path)
        save_excel(work_id, refined_pairs, xlsx_live_path)
    
    print(f"  Completed refine for {len(refined_pairs)} pairs.")
    return refined_pairs


# ============================================================
def export_debug_segmentation(work_id: str, han_raw: str, viet_raw: str, is_chunked: bool, han_chunks: List[str], viet_chunks: List[str]):
    """Xuất file debug kết quả tách câu để kiểm tra ngữ đoạn mà không cần chạy mô hình."""
    debug_path = DATA_DIR / f"{work_id}_debug_sentences.txt"
    lines = []
    lines.append("=" * 80)
    lines.append(f"DEBUG SEGMENTATION REPORT: {work_id}")
    lines.append("=" * 80)
    
    if is_chunked:
        lines.append(f"\nMode: Chunk-wise ({len(han_chunks)} chunks)\n")
        total_h = 0
        total_v = 0
        for c_idx, (h_c, v_c) in enumerate(zip(han_chunks, viet_chunks)):
            h_s = segment_han(h_c)
            v_s = segment_viet(v_c)
            lines.append(f"\n{'#'*35} CHUNK {c_idx+1:02d} (Han: {len(h_s)}, Viet: {len(v_s)}) {'#'*35}")
            lines.append("--- [HAN SENTENCES] ---")
            for i, s in enumerate(h_s):
                lines.append(f"  [H_{total_h + i + 1:04d}] (chunk_pos={i+1:02d}, len={len(s):02d}) {s}")
            lines.append("--- [VIET SENTENCES] ---")
            for j, s in enumerate(v_s):
                lines.append(f"  [V_{total_v + j + 1:04d}] (chunk_pos={j+1:02d}, len={len(s):03d}) {s}")
            total_h += len(h_s)
            total_v += len(v_s)
        lines.append(f"\n{'='*80}")
        lines.append(f"SUMMARY: Total Han = {total_h} sentences | Total Viet = {total_v} sentences | Ratio = 1:{total_v/max(total_h, 1):.2f}")
        lines.append("=" * 80)
    else:
        h_s = segment_han(han_raw)
        v_s = segment_viet(viet_raw)
        lines.append(f"\nMode: Document-level\n")
        lines.append(f"--- [HAN SENTENCES] (Total: {len(h_s)}) ---")
        for i, s in enumerate(h_s):
            lines.append(f"  [H_{i+1:04d}] (len={len(s):02d}) {s}")
        lines.append(f"\n--- [VIET SENTENCES] (Total: {len(v_s)}) ---")
        for j, s in enumerate(v_s):
            lines.append(f"  [V_{j+1:04d}] (len={len(s):03d}) {s}")
        lines.append(f"\n{'='*80}")
        lines.append(f"SUMMARY: Total Han = {len(h_s)} sentences | Total Viet = {len(v_s)} sentences | Ratio = 1:{len(v_s)/max(len(h_s), 1):.2f}")
        lines.append("=" * 80)
        
    debug_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[DEBUG] - Debug file exported: {debug_path}")
    if is_chunked:
        print(f"        Mode: Chunk-wise ({len(han_chunks)} chunks) | Han: {total_h} câu | Viet: {total_v} câu")
    else:
        print(f"        Mode: Document-level | Han: {len(h_s)} câu | Viet: {len(v_s)} câu")


# ============================================================
# 7. MAIN
# ============================================================
def align_work(work: Dict, use_llm: bool = True, debug_mode: bool = False):
    """Thực hiện dóng câu cho một tác phẩm."""
    work_id = work['id']
    work_title = work['viet']
    
    print(f"\n{'='*60}")
    print(f"Aligning: {work_title} ({work_id})")
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
    
    # Kiểm tra xem có áp dụng Chunk-wise alignment không
    is_chunked = len(han_chunks) > 1 and len(han_chunks) == len(viet_chunks)
    
    # Nếu bật cờ debug, chỉ xuất file phân tích tách câu rồi thoát ngay
    if debug_mode:
        export_debug_segmentation(work_id, han_raw, viet_raw, is_chunked, han_chunks, viet_chunks)
        return
    
    all_han_sents = []
    all_viet_sents = []
    all_alignment = []
    
    if is_chunked:
        print(f"  Detected {len(han_chunks)} aligned chunks -> Running Chunk-wise Alignment...")
        han_offset = 0
        viet_offset = 0
        
        for chunk_idx, (h_chunk, v_chunk) in enumerate(zip(han_chunks, viet_chunks)):
            h_sents = segment_han(h_chunk)
            v_sents = segment_viet(v_chunk)
            
            if not h_sents or not v_sents:
                continue
            
            # Tính similarity và dóng câu đơn điệu trong phạm vi từng chunk
            chunk_sim = compute_similarity(h_sents, v_sents)
            chunk_align = monotonic_dp_alignment(chunk_sim, diagonal_weight=0.05)
            
            # Lưu lại câu với offset toàn cục
            for h_idx, v_idx, score in chunk_align:
                global_h = han_offset + h_idx
                global_v = (viet_offset + v_idx) if v_idx >= 0 else -1
                all_alignment.append((global_h, global_v, score))
            
            all_han_sents.extend(h_sents)
            all_viet_sents.extend(v_sents)
            han_offset += len(h_sents)
            viet_offset += len(v_sents)
    else:
        # Fallback: Document-level alignment khi không có cấu trúc chunk tương ứng
        all_han_sents = segment_han(han_raw)
        all_viet_sents = segment_viet(viet_raw)
        
        if not all_han_sents or not all_viet_sents:
            print("  Warning: No sentences found after segmentation.")
            return
        
        sim_matrix = compute_similarity(all_han_sents, all_viet_sents)
        all_alignment = monotonic_dp_alignment(sim_matrix, diagonal_weight=0.05)
    
    ratio = len(all_viet_sents) / max(len(all_han_sents), 1)
    matched = sum(1 for _, v, _ in all_alignment if v >= 0)
    scores = [s for _, v, s in all_alignment if v >= 0]
    
    print(f"\n  Preliminary statistics:")
    print(f"     - Han: {len(all_han_sents)} sentences")
    print(f"     - Viet: {len(all_viet_sents)} sentences (Ratio: 1:{ratio:.2f})")
    print(f"     - Matched: {matched}/{len(all_han_sents)} pairs")
    if scores:
        print(f"     - Score: min={min(scores):.3f}, max={max(scores):.3f}, mean={np.mean(scores):.3f}")
    
    # Bước 4: Chuẩn hóa bằng LLM
    refined_pairs = refine_with_llm(all_alignment, all_han_sents, all_viet_sents, work_id, work_title, use_llm=use_llm)
    
    # Bước 5: Thống kê kết quả
    total = len(refined_pairs)
    approved = sum(1 for p in refined_pairs if p.get("llm_approved", True))
    
    print(f"\n  Final statistics:")
    print(f"     - Total pairs: {total}")
    print(f"     - Approved: {approved}")
    print(f"     - Rejected: {total - approved}")
    
    # Bước 6: Lưu file kết quả chuẩn TSV và XLSX
    tsv_path = CORPUS_DIR / f"{work_id}_parallel.tsv"
    xlsx_path = CORPUS_DIR / f"{work_id}_parallel.xlsx"
    count = save_tsv(work_id, refined_pairs, tsv_path)
    save_excel(work_id, refined_pairs, xlsx_path)
    print(f"  Saved {count} pairs -> {tsv_path.name} & {xlsx_path.name}")


def main():
    """Hàm thực thi chính của sentence alignment."""
    parser = argparse.ArgumentParser(description="Sentence Alignment")
    parser.add_argument("--work-id", type=str, help="ID tác phẩm cụ thể")
    parser.add_argument("--no-llm", action="store_true", help="Không dùng LLM refine")
    parser.add_argument("--clear-cache", action="store_true", help="Xóa cache embeddings")
    parser.add_argument("--debug", action="store_true", help="Chỉ xuất file debug tách câu (không chạy embedding/alignment)")
    args = parser.parse_args()
    
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
    
    if not args.debug:
        print(f"Aligning {len(works)} works (Monotonic Dynamic Programming)")
        print(f"   LLM: {'OFF' if args.no_llm else f'ON ({LLM_MODEL_NAME})'}")
        print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print(f"   Batch size: {BATCH_LLM_SIZE}")
    else:
        print(f"Running DEBUG SEGMENTATION MODE for {len(works)} works...")
    
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
