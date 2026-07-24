"""
Sentence Alignment - Greedy Best-Match + Ensemble + LLM Refine (DeepSeek)
Mỗi câu Hán -> 1 câu Việt tốt nhất, refine bằng DeepSeek API
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
from pathlib import Path
from typing import List, Tuple, Dict

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

# LLM Config (DeepSeek API)
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.deepseek.com/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 300))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 3))
BATCH_LLM_SIZE = int(os.getenv("BATCH_LLM_SIZE", 3))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.40))

# Ensemble weights
LABSE_WEIGHT = 0.50
BERT_WEIGHT = 0.30
LENGTH_WEIGHT = 0.20

# Paths
PROJECT_ROOT = Path(os.getcwd())
DATA_DIR = PROJECT_ROOT / "data"
OCR_OUTPUT_DIR = DATA_DIR / "ocr_output"
CORPUS_DIR = DATA_DIR / "corpus"
CONFIG_PATH = DATA_DIR / "config.json"
CACHE_DIR = DATA_DIR / ".cache"

# ============================================================
# LOAD MODELS
# ============================================================
print("\n" + "="*60)
print("Loading models...")
print("="*60)

labse_model = SentenceTransformer("LaBSE")
bert_model = SentenceTransformer("bert-base-multilingual-cased")
print("\n All models ready!\n")


# ============================================================
# 1. NORMALIZE
# ============================================================
def normalize_viet(text: str) -> str:
    """Fix OCR: 'Vi ệ n Đạ i H ọ c' → 'Viện Đại Học'"""
    for _ in range(5):
        text = re.sub(r'(\S) (\S) (\S)', r'\1\2\3', text)
    for _ in range(3):
        text = re.sub(r'(\S) (\S)', r'\1\2', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_han(text: str) -> str:
    """Xóa khoảng trắng giữa chữ Hán."""
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
    return re.sub(r'\s+', ' ', text).strip()


# ============================================================
# 2. IMPROVED HAN SEGMENTATION
# ============================================================
def clean_lines(text: str) -> List[str]:
    """Lọc bỏ dòng rác."""
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
    Tách câu Hán CẢI TIẾN:
    Bước 1: Normalize + clean
    Bước 2: Tách theo xuống dòng
    Bước 3: Tách theo dấu câu mạnh: 。；！？
    Bước 4: Tách theo dấu câu yếu: 、， (nếu > 30 chars)
    Bước 5: Tách câu quá dài (> 50 chars)
    Bước 6: Gộp câu quá ngắn (< 3 chars)
    """
    text = normalize_han(text)
    lines = clean_lines(text)
    
    # Bước 1: Tách theo xuống dòng
    raw_segments = []
    for line in lines:
        line = line.strip()
        if line:
            raw_segments.append(line)
    
    # Bước 2: Tách theo dấu câu MẠNH
    strong_split = []
    for seg in raw_segments:
        parts = re.split(r'([。；！？])', seg)
        current = ""
        for i in range(0, len(parts), 2):
            current += parts[i]
            if i + 1 < len(parts):
                current += parts[i + 1]
                if len(current.strip()) >= 2:
                    strong_split.append(current.strip())
                    current = ""
        if current.strip() and len(current.strip()) >= 2:
            strong_split.append(current.strip())
    
    # Bước 3: Tách theo dấu câu YẾU nếu > 30 chars
    weak_split = []
    for seg in strong_split:
        if len(seg) > 30:
            parts = re.split(r'([、，])', seg)
            current = ""
            for i in range(0, len(parts), 2):
                current += parts[i]
                if i + 1 < len(parts):
                    current += parts[i + 1]
                    if len(current.strip()) >= 3:
                        weak_split.append(current.strip())
                        current = ""
            if current.strip() and len(current.strip()) >= 2:
                weak_split.append(current.strip())
        else:
            weak_split.append(seg)
    
    # Bước 4: Tách câu QUÁ DÀI (> 50 chars)
    final_split = []
    for seg in weak_split:
        if len(seg) > 50:
            chunk_size = 25
            for i in range(0, len(seg), chunk_size):
                chunk = seg[i:i+chunk_size]
                if chunk.strip():
                    final_split.append(chunk.strip())
        else:
            final_split.append(seg)
    
    # Bước 5: Gộp câu QUÁ NGẮN (< 3 chars)
    merged = []
    for seg in final_split:
        if len(seg) < 3 and merged:
            merged[-1] = merged[-1] + seg
        else:
            merged.append(seg)
    
    # Bước 6: Lọc câu rác
    result = []
    for seg in merged:
        seg = seg.strip()
        if not seg or len(seg) < 2:
            continue
        if re.match(r'^[。；！？、，\s]+$', seg):
            continue
        if re.match(r'^[\d\s]+$', seg):
            continue
        result.append(seg)
    
    return result


def segment_viet(text: str) -> List[str]:
    """Tách câu Việt, gộp câu ngắn."""
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
    
    # Gộp câu quá ngắn
    merged = []
    for s in sentences:
        if len(s) < 15 and merged:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    
    return merged


# ============================================================
# 3. EMBEDDINGS
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
                print(f"      {name}: cache ({len(texts)})")
                return emb
            else:
                print(f"      {name}: cache mismatch, recomputing...")
                os.remove(cache_path)
        except:
            os.remove(cache_path)
    
    print(f"     {name}: computing ({len(texts)})...")
    emb = model.encode(texts, batch_size=64, show_progress_bar=True)
    
    with open(cache_path, 'wb') as f:
        pickle.dump(emb, f)
    
    return emb


def length_sim(len1: int, len2: int, ratio: float = 1.5) -> float:
    """Gale-Church length similarity."""
    normalized = len2 / ratio
    delta = (len1 - normalized) / math.sqrt(max(len1, 1) * 6.8)
    return math.exp(-delta * delta / 2)


def compute_similarity(han_sents: List[str], viet_sents: List[str]) -> np.ndarray:
    """Ensemble similarity: LaBSE + BERT + Length."""
    n_han, n_viet = len(han_sents), len(viet_sents)
    print(f"\n     Similarity ({n_han} Han × {n_viet} Viet)...")
    
    labse_han = get_embeddings(han_sents, labse_model, "labse_han")
    labse_viet = get_embeddings(viet_sents, labse_model, "labse_viet")
    labse_sim = cosine_similarity(labse_han, labse_viet)
    
    bert_han = get_embeddings(han_sents, bert_model, "bert_han")
    bert_viet = get_embeddings(viet_sents, bert_model, "bert_viet")
    bert_sim = cosine_similarity(bert_han, bert_viet)
    
    if labse_sim.shape != bert_sim.shape:
        raise ValueError(f"Shape mismatch: {labse_sim.shape} vs {bert_sim.shape}")
    
    length_matrix = np.zeros((n_han, n_viet))
    for i in range(n_han):
        for j in range(n_viet):
            length_matrix[i, j] = length_sim(len(han_sents[i]), len(viet_sents[j]))
    
    sim = LABSE_WEIGHT * labse_sim + BERT_WEIGHT * bert_sim + LENGTH_WEIGHT * length_matrix
    
    print(f"     Ensemble: max={sim.max():.3f}, mean={sim.mean():.3f}")
    return sim


# ============================================================
# 4. GREEDY BEST-MATCH ALIGNMENT
# ============================================================
def greedy_best_match(sim_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Mỗi câu Hán → 1 câu Việt tốt nhất.
    Ưu tiên câu Hán có score cao hơn khi có xung đột.
    Returns: [(han_idx, viet_idx, score), ...]
    """
    n_han, n_viet = sim_matrix.shape
    
    han_best = []
    for i in range(n_han):
        best_j = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i, best_j])
        han_best.append((i, best_j, best_score))
    
    # Sắp xếp theo score giảm dần → ưu tiên câu Hán có score cao
    han_best.sort(key=lambda x: x[2], reverse=True)
    
    used_viet = set()
    alignment = []
    
    for han_idx, viet_idx, score in han_best:
        if viet_idx not in used_viet:
            alignment.append((han_idx, viet_idx, score))
            used_viet.add(viet_idx)
        else:
            # Tìm câu Việt tốt tiếp theo chưa được dùng
            row = sim_matrix[han_idx].copy()
            for used_j in used_viet:
                row[used_j] = -1
            
            next_j = int(np.argmax(row))
            next_score = float(row[next_j])
            
            if next_score > 0:
                alignment.append((han_idx, next_j, next_score))
                used_viet.add(next_j)
            else:
                alignment.append((han_idx, -1, 0.0))
    
    # Sắp xếp theo thứ tự câu Hán
    alignment.sort(key=lambda x: x[0])
    
    matched = sum(1 for _, v, _ in alignment if v >= 0)
    scores = [s for _, v, s in alignment if v >= 0]
    
    print(f"\n  🔗 Greedy Best-Match:")
    print(f"     - Matched: {matched}/{n_han}")
    if scores:
        print(f"     - Score: min={min(scores):.3f}, max={max(scores):.3f}, mean={np.mean(scores):.3f}")
    
    return alignment


# ============================================================
# 5. LLM REFINE (DEEPSEEK API)
# ============================================================
def extract_json_from_text(text: str):
    """Trích xuất JSON từ response LLM - robust version."""
    if not text:
        return None
    
    text = text.strip()
    
    # 1. Xóa thinking tags (nếu có)
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    text = re.sub(r'<\|[^|]*\|>', '', text)
    
    # 2. Xóa markdown code blocks
    text = text.replace('```json', '').replace('```', '')
    
    # 3. Tìm JSON array/object đầu tiên
    for start_char in ['[', '{']:
        idx = text.find(start_char)
        if idx >= 0:
            text = text[idx:]
            break
    
    # 4. Thử parse trực tiếp
    try:
        return json.loads(text.strip())
    except:
        pass
    
    # 5. Tìm JSON array hoàn chỉnh
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            # Thử sửa truncated JSON
            json_str = match.group()
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            try:
                return json.loads(json_str)
            except:
                pass
    
    # 6. Tìm tất cả JSON objects riêng lẻ
    objects = re.findall(r'\{[^{}]*\}', text)
    if objects:
        results = []
        for obj in objects:
            try:
                results.append(json.loads(obj))
            except:
                continue
        if results:
            return results
    
    # 7. Parse từng dòng
    results = []
    for line in text.split('\n'):
        line = line.strip().rstrip(',')
        if line.startswith('{') and line.endswith('}'):
            try:
                results.append(json.loads(line))
            except:
                continue
    if results:
        return results
    
    return None


def call_llm_batch(pairs: List[Tuple[str, str, str]], work_title: str) -> List[Dict]:
    """
    Gọi DeepSeek API để verify các cặp câu Hán-Việt.
    Không dùng extra_body, không dùng response_format.
    """
    system_prompt = """You are a Han-Nom expert. Check if Han-Vietnamese sentence pairs are equivalent translations.

Reply with ONLY a JSON array. No explanation. No markdown. No thinking. Just the JSON array.

Example input:
[Pair temp_0]
[HÁN]: 安南志略
[VIỆT]: An Nam chí lược

[Pair temp_1]
[HÁN]: 黎崱撰
[VIỆT]: Lê Tắc soạn

Example output:
[{"pair_id":"temp_0","match":true,"han_corrected":"安南志略","viet_corrected":"An Nam chí lược"},{"pair_id":"temp_1","match":true,"han_corrected":"黎崱撰","viet_corrected":"Lê Tắc soạn"}]

Rules:
- match=true: equivalent meaning
- match=false: not equivalent
- han_corrected/viet_corrected: keep original if OK, fix if needed
- Reply ONLY the JSON array, nothing else"""

    pairs_text = "\n\n".join([
        f"[Pair {pid}]\n[HÁN]: {han}\n[VIỆT]: {viet}"
        for han, viet, pid in pairs
    ])
    
    user_prompt = f'Work: "{work_title}"\n\n{pairs_text}\n\nReply ONLY JSON array:'

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Payload đơn giản - KHÔNG có extra_body, KHÔNG có response_format
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
                
                # Nếu content rỗng, thử reasoning_content
                if not content.strip():
                    content = message.get('reasoning_content', '') or ''
                
                # Extract JSON
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
                
                # Log raw response để debug
                if attempt == 0:
                    print(f" Raw: {content[:200]}...")
                
            elif response.status_code == 429:
                print(f"    Rate limit, waiting 30s...")
                time.sleep(30)
                continue
            else:
                print(f"    API {response.status_code}: {response.text[:150]}")
                # Nếu lỗi 400 → dừng retry, fallback ngay
                if response.status_code == 400:
                    break
                
        except requests.exceptions.Timeout:
            print(f"    Timeout (attempt {attempt+1})")
        except Exception as e:
            print(f"    Error: {e}")
        
        if attempt < LLM_MAX_RETRIES - 1:
            wait = 2 ** (attempt + 1)
            print(f"    Retrying in {wait}s...")
            time.sleep(wait)
    
    # Fallback: giữ nguyên
    return [
        {"pair_id": pid, "match": True, "han_corrected": han, "viet_corrected": viet}
        for han, viet, pid in pairs
    ]


def refine_with_llm(alignment: List[Tuple[int, int, float]], 
                    han_sents: List[str], viet_sents: List[str], 
                    work_title: str, use_llm: bool = True) -> List[Dict]:
    """
    Refine alignment bằng LLM (DeepSeek API).
    Nhận tuple 3 phần tử (han_idx, viet_idx, score) từ greedy_best_match.
    Chỉ verify các cặp có score < CONFIDENCE_THRESHOLD.
    """
    refined_pairs = []
    
    # Bước 1: Xây dựng danh sách cặp
    for idx, (han_idx, viet_idx, score) in enumerate(alignment):
        # Bỏ qua cặp không match
        if viet_idx < 0:
            continue
        
        han_text = han_sents[han_idx]
        viet_text = viet_sents[viet_idx]
        
        pair_data = {
            "han": han_text,
            "viet": viet_text,
            "score": score,
            "pair_id": f"temp_{idx}",
            "llm_approved": score >= CONFIDENCE_THRESHOLD
        }
        refined_pairs.append(pair_data)
    
    # Bước 2: Xác định cặp cần verify (score thấp)
    need_verify = [p for p in refined_pairs if p["score"] < CONFIDENCE_THRESHOLD]
    
    if not need_verify:
        print(f"\nAll pairs have score >= {CONFIDENCE_THRESHOLD}, no LLM needed.")
        return refined_pairs
    
    if not use_llm:
        print(f"\nSkipping LLM refine (--no-llm). {len(need_verify)} pairs unverified.")
        return refined_pairs
    
    if not LLM_API_KEY:
        print(f"\nLLM_API_KEY not set. Skipping LLM refine.")
        return refined_pairs
    
    # Bước 3: Batch verify bằng LLM
    print(f"\n  Refining {len(need_verify)} pairs with LLM (batch={BATCH_LLM_SIZE})...")
    
    for i in tqdm(range(0, len(need_verify), BATCH_LLM_SIZE), desc="    LLM"):
        batch = need_verify[i:i+BATCH_LLM_SIZE]
        batch_input = [(p["han"], p["viet"], p["pair_id"]) for p in batch]
        
        llm_results = call_llm_batch(batch_input, work_title)
        result_map = {r.get("pair_id"): r for r in llm_results}
        
        for pair in batch:
            llm_result = result_map.get(pair["pair_id"], {})
            pair["llm_approved"] = llm_result.get("match", True)
            
            if llm_result.get("match", False):
                # LLM xác nhận match → cập nhật text nếu có sửa
                corrected_han = llm_result.get("han_corrected", "")
                corrected_viet = llm_result.get("viet_corrected", "")
                if corrected_han:
                    pair["han"] = corrected_han
                if corrected_viet:
                    pair["viet"] = corrected_viet
    
    # Thống kê
    approved = sum(1 for p in refined_pairs if p.get("llm_approved", True))
    rejected = sum(1 for p in refined_pairs if not p.get("llm_approved", True))
    print(f"  LLM Results: {approved} approved, {rejected} rejected")
    
    return refined_pairs


# ============================================================
# 6. SAVE
# ============================================================
def save_tsv(work_id: str, pairs: List[Dict], output_path: Path, 
             include_rejected: bool = False) -> int:
    """Save TSV: pair_id, han_sentence, viet_sentence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['pair_id', 'han_sentence', 'viet_sentence'])
        
        count = 0
        for pair in pairs:
            # Filter: bỏ cặp bị LLM reject (trừ khi include_rejected=True)
            if not include_rejected and not pair.get("llm_approved", True):
                continue
            
            pair_id = f"{work_id}_{count:04d}"
            han = pair["han"].replace('\t', ' ').replace('\n', ' ')
            viet = pair["viet"].replace('\t', ' ').replace('\n', ' ')
            writer.writerow([pair_id, han, viet])
            count += 1
    
    return count


# ============================================================
# 7. MAIN
# ============================================================
def align_work(work: Dict, use_llm: bool = True):
    """Align one work."""
    work_id = work['id']
    work_title = work['viet']
    
    print(f"\n{'='*60}")
    print(f"Aligning: {work_title} ({work_id})")
    print(f"{'='*60}")
    
    han_path = OCR_OUTPUT_DIR / f"{work_id}_sino_raw.txt"
    viet_path = OCR_OUTPUT_DIR / f"{work_id}_vie_raw.txt"
    
    if not han_path.exists() or not viet_path.exists():
        print(f"  Missing files")
        return
    
    han_text = han_path.read_text(encoding='utf-8')
    viet_text = viet_path.read_text(encoding='utf-8')
    
    # 1. Segment
    han_sents = segment_han(han_text)
    viet_sents = segment_viet(viet_text)
    
    ratio = len(viet_sents) / max(len(han_sents), 1)
    
    print(f"\n  Segmentation:")
    print(f"     - Han: {len(han_sents)} sentences")
    print(f"     - Viet: {len(viet_sents)} sentences")
    print(f"     - Ratio: 1:{ratio:.2f}")
    
    # Debug samples
    print(f"\n  Sample Han (first 10):")
    for i, s in enumerate(han_sents[:10]):
        print(f"     [{i}] ({len(s)}c) {s[:60]}")
    
    print(f"\n  Sample Viet (first 5):")
    for i, s in enumerate(viet_sents[:5]):
        print(f"     [{i}] ({len(s)}c) {s[:60]}...")
    
    if not han_sents or not viet_sents:
        print("  No sentences")
        return
    
    # 2. Similarity
    sim_matrix = compute_similarity(han_sents, viet_sents)
    
    # 3. Alignment
    alignment = greedy_best_match(sim_matrix)
    
    # 4. LLM Refine
    refined_pairs = refine_with_llm(alignment, han_sents, viet_sents, work_title, use_llm=use_llm)
    
    # 5. Stats
    total = len(refined_pairs)
    approved = sum(1 for p in refined_pairs if p.get("llm_approved", True))
    
    print(f"\n     Final:")
    print(f"     - Total pairs: {total}")
    print(f"     - Approved: {approved}")
    print(f"     - Rejected: {total - approved}")
    
    # 6. Save (chỉ cặp approved)
    tsv_path = CORPUS_DIR / f"{work_id}_parallel.tsv"
    count = save_tsv(work_id, refined_pairs, tsv_path, include_rejected=False)
    print(f"Saved {count} approved pairs → {tsv_path}")
    
    # 7. Save all (bao gồm rejected, để debug)
    tsv_all_path = CORPUS_DIR / f"{work_id}_parallel_all.tsv"
    count_all = save_tsv(work_id, refined_pairs, tsv_all_path, include_rejected=True)
    print(f"  Saved {count_all} all pairs → {tsv_all_path}")


def main():
    parser = argparse.ArgumentParser(description="Sentence Alignment (Greedy Best-Match + LLM Refine)")
    parser.add_argument("--work-id", type=str, help="ID tác phẩm cụ thể")
    parser.add_argument("--no-llm", action="store_true", help="Không dùng LLM refine")
    parser.add_argument("--clear-cache", action="store_true", help="Xóa cache embeddings")
    args = parser.parse_args()
    
    # Clear cache
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
    
    print(f"Aligning {len(works)} works (Greedy Best-Match)")
    print(f"   LLM: {'OFF' if args.no_llm else 'ON (DeepSeek)'}")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"   Batch size: {BATCH_LLM_SIZE}")
    
    for work in works:
        try:
            align_work(work, use_llm=not args.no_llm)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()
