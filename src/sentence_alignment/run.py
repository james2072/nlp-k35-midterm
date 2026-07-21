"""
Sentence Alignment - Greedy Best-Match + Improved Han Segmentation
Mỗi câu Hán → 1 câu Việt tốt nhất
"""

import os
import sys
import json
import csv
import argparse
import re
import hashlib
import pickle
import math
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIG
# ============================================================
load_dotenv()

LABSE_WEIGHT = 0.50
BERT_WEIGHT = 0.30
LENGTH_WEIGHT = 0.20

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
print("  LaBSE ready")

bert_model = SentenceTransformer("bert-base-multilingual-cased")
print("  BERT ready")



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
# 2. IMPROVED HAN SEGMENTATION (CẢI TIẾN MẠNH)
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
    Bước 2: Tách theo xuống dòng (\n)
    Bước 3: Tách theo dấu câu mạnh: 。；！？
    Bước 4: Tách theo dấu câu yếu: 、， (nếu câu > 30 chars)
    Bước 5: Tách câu quá dài (>50 chars) thành các đoạn nhỏ
    Bước 6: Gộp câu quá ngắn (<3 chars) vào câu trước
    
    Kết quả: Mỗi câu 3-50 chars, tách đúng theo ngữ nghĩa
    """
    text = normalize_han(text)
    lines = clean_lines(text)
    
    # 1: Tách theo xuống dòng 
    raw_segments = []
    for line in lines:
        line = line.strip()
        if line:
            raw_segments.append(line)
    
    # 2: Tách theo dấu câu MẠNH (。；！？) 
    strong_split = []
    for seg in raw_segments:
        # Tách theo dấu câu mạnh, GIỮ LẠI dấu câu
        parts = re.split(r'([。；！？])', seg)
        current = ""
        for i in range(0, len(parts), 2):
            current += parts[i]
            if i + 1 < len(parts):
                current += parts[i + 1]  # Thêm dấu câu
                if len(current.strip()) >= 2:
                    strong_split.append(current.strip())
                    current = ""
        if current.strip() and len(current.strip()) >= 2:
            strong_split.append(current.strip())
    
    # 3: Tách theo dấu câu YẾU (、，) nếu câu > 30 chars
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
    
    # 4: Tách câu QUÁ DÀI (>50 chars) theo vị trí cố định
    final_split = []
    for seg in weak_split:
        if len(seg) > 50:
            # Tách mỗi 25-30 chars tại vị trí không cắt giữa chữ
            chunk_size = 25
            for i in range(0, len(seg), chunk_size):
                chunk = seg[i:i+chunk_size]
                if chunk.strip():
                    final_split.append(chunk.strip())
        else:
            final_split.append(seg)
    
    # 5: Gộp câu QUÁ NGẮN (<3 chars) vào câu trước
    merged = []
    for seg in final_split:
        if len(seg) < 3 and merged:
            merged[-1] = merged[-1] + seg
        else:
            merged.append(seg)
    
    # 6: Lọc bỏ câu rác
    result = []
    for seg in merged:
        seg = seg.strip()
        if not seg or len(seg) < 2:
            continue
        # Bỏ câu chỉ chứa dấu câu
        if re.match(r'^[。；！？、，\s]+$', seg):
            continue
        # Bỏ câu chỉ chứa số
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
    """Compute embeddings, tự động xóa cache nếu shape sai."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    text_hash = hashlib.md5('\n'.join(texts).encode('utf-8')).hexdigest()
    cache_path = CACHE_DIR / f"{name}_{len(texts)}_{text_hash}.pkl"
    
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                emb = pickle.load(f)
            if emb.shape[0] == len(texts):
                print(f"     {name}: cache ({len(texts)})")
                return emb
            else:
                print(f"     {name}: cache mismatch, recomputing...")
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
    """Ensemble similarity."""
    n_han, n_viet = len(han_sents), len(viet_sents)
    print(f"\n  Similarity ({n_han} Han × {n_viet} Viet)...")
    
    labse_han = get_embeddings(han_sents, labse_model, "labse_han")
    labse_viet = get_embeddings(viet_sents, labse_model, "labse_viet")
    labse_sim = cosine_similarity(labse_han, labse_viet)
    
    bert_han = get_embeddings(han_sents, bert_model, "bert_han")
    bert_viet = get_embeddings(viet_sents, bert_model, "bert_viet")
    bert_sim = cosine_similarity(bert_han, bert_viet)
    
    # Validation
    if labse_sim.shape != bert_sim.shape:
        raise ValueError(f"Shape mismatch: {labse_sim.shape} vs {bert_sim.shape}. Xóa cache: Remove-Item -Recurse -Force data\\.cache")
    
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
    """
    n_han, n_viet = sim_matrix.shape
    
    # Với mỗi câu Hán, tìm câu Việt tốt nhất
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
    
    print(f"\n  Greedy Best-Match:")
    print(f"     - Matched: {matched}/{n_han}")
    if scores:
        print(f"     - Score: min={min(scores):.3f}, max={max(scores):.3f}, mean={np.mean(scores):.3f}")
    
    return alignment


# ============================================================
# 5. SAVE
# ============================================================
def save_tsv(work_id: str, pairs: List[Dict], output_path: Path) -> int:
    """Save TSV: pair_id, han_sentence, viet_sentence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['pair_id', 'han_sentence', 'viet_sentence'])
        
        count = 0
        for pair in pairs:
            pair_id = f"{work_id}_{count:04d}"
            han = pair["han"].replace('\t', ' ').replace('\n', ' ')
            viet = pair["viet"].replace('\t', ' ').replace('\n', ' ')
            writer.writerow([pair_id, han, viet])
            count += 1
    
    return count


# ============================================================
# 6. MAIN
# ============================================================
def align_work(work: Dict):
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
    
    # 4. Build pairs
    pairs = []
    for han_idx, viet_idx, score in alignment:
        if viet_idx >= 0:
            pairs.append({
                "han": han_sents[han_idx],
                "viet": viet_sents[viet_idx],
                "score": score
            })
    
    print(f"\n  Final: {len(pairs)} pairs")
    
    # 5. Save
    tsv_path = CORPUS_DIR / f"{work_id}_parallel.tsv"
    count = save_tsv(work_id, pairs, tsv_path)
    print(f"  Saved {count} pairs → {tsv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-id", type=str)
    parser.add_argument("--no-llm", action="store_true")
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
    
    for work in works:
        try:
            align_work(work)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()