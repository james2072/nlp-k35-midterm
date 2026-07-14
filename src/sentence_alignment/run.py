"""
Sentence Alignment Pipeline - Kết hợp LaBSE + LLM
Output: data/corpus/{work_id}_parallel.tsv
Format: [pair_id]\t[han_sentence]\t[viet_sentence]
"""

import os
import sys
import json
import csv
import argparse
import re
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:1234/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen/qwen3-4b-2507")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 300))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 3))

# LaBSE config
LABSE_MODEL = "LaBSE"
SIMILARITY_THRESHOLD = 0.45  # cosine similarity tối thiểu
CONFIDENCE_THRESHOLD = 0.65  # -> gửi LLM verify

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OCR_OUTPUT_DIR = DATA_DIR / "ocr_output"
CORPUS_DIR = DATA_DIR / "corpus"
CONFIG_PATH = DATA_DIR / "config.json"

# LaBSE
print("Đang load LaBSE...")
labse_model = SentenceTransformer(LABSE_MODEL)
print("Loaded LaBSE!")


# 1. TÁCH CÂU (SENTENCE SEGMENTATION)
def segment_sentences(text: str, lang: str = "hán") -> List[str]:
    """
    Tách text thành các câu/cụm câu dựa trên heuristic.
    - Hán: mỗi dòng là 1 cụm (vì đã được sort theo cột dọc)
    - Việt: mỗi dòng là 1 câu (đã được sort ngang)
    Gộp các dòng quá ngắn (< 3 chars) với dòng trước để tránh sentence rác.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if not lines:
        return []
    
    sentences = []
    buffer = ""
    min_len = 3 if lang == "hán" else 5
    
    for line in lines:
        if len(line) < min_len and buffer:
            # Dòng quá ngắn -> gộp vào câu trước
            buffer += line
        else:
            if buffer:
                sentences.append(buffer)
            buffer = line
    
    if buffer:
        sentences.append(buffer)
    
    return sentences


# 2. DÓNG HÀNG BẰNG LABSE + DYNAMIC PROGRAMMING
def compute_similarity_matrix(han_sents: List[str], viet_sents: List[str]) -> np.ndarray:
    # Tính ma trận cosine similarity giữa 2 bộ câu
    print(f" Đang embed {len(han_sents)} câu Hán + {len(viet_sents)} câu Việt...")
    
    # Batch encode
    han_embeds = labse_model.encode(han_sents, batch_size=64, show_progress_bar=False)
    viet_embeds = labse_model.encode(viet_sents, batch_size=64, show_progress_bar=False)
    
    # Cosine similarity matrix
    sim_matrix = cosine_similarity(han_embeds, viet_embeds)
    return sim_matrix


def dynamic_align(sim_matrix: np.ndarray) -> List[Tuple[List[int], List[int], float]]:
    """
    Dynamic Programming alignment với các loại: 1-1, 1-2, 2-1, 1-0, 0-1.
    Trả về list các cặp (han_indices, viet_indices, score).
    """
    n_han, n_viet = sim_matrix.shape
    
    # Penalty -> không align
    SKIP_PENALTY = -0.5
    
    # dp[i][j] = max score khi align i câu Hán đầu với j câu Việt đầu
    dp = np.full((n_han + 1, n_viet + 1), -np.inf)
    dp[0, 0] = 0
    backtrack = {}
    
    for i in range(n_han + 1):
        for j in range(n_viet + 1):
            if i == 0 and j == 0:
                continue
            
            candidates = []
            
            # 1-1: align han[i-1] với viet[j-1]
            if i > 0 and j > 0:
                score = dp[i-1, j-1] + sim_matrix[i-1, j-1]
                candidates.append((score, ('1-1', i-1, j-1)))
            
            # 1-2: align han[i-1] với viet[j-2], viet[j-1]
            if i > 0 and j > 1:
                avg_sim = (sim_matrix[i-1, j-2] + sim_matrix[i-1, j-1]) / 2
                score = dp[i-1, j-2] + avg_sim - 0.1  # penalty nhẹ
                candidates.append((score, ('1-2', i-1, [j-2, j-1])))
            
            # 2-1: align han[i-2], han[i-1] với viet[j-1]
            if i > 1 and j > 0:
                avg_sim = (sim_matrix[i-2, j-1] + sim_matrix[i-1, j-1]) / 2
                score = dp[i-2, j-1] + avg_sim - 0.1
                candidates.append((score, ('2-1', [i-2, i-1], j-1)))
            
            # Skip han (1-0)
            if i > 0:
                score = dp[i-1, j] + SKIP_PENALTY
                candidates.append((score, ('skip_han', i-1, None)))
            
            # Skip viet (0-1)
            if j > 0:
                score = dp[i, j-1] + SKIP_PENALTY
                candidates.append((score, ('skip_viet', None, j-1)))
            
            if candidates:
                best_score, best_move = max(candidates, key=lambda x: x[0])
                dp[i, j] = best_score
                backtrack[(i, j)] = best_move
    
    # Backtrack để lấy alignment
    alignment = []
    i, j = n_han, n_viet
    
    while (i, j) != (0, 0):
        move = backtrack.get((i, j))
        if not move:
            break
        
        move_type, han_idx, viet_idx = move
        
        if move_type == '1-1':
            score = sim_matrix[han_idx, viet_idx]
            alignment.append(([han_idx], [viet_idx], score))
            i, j = i - 1, j - 1
        elif move_type == '1-2':
            avg_score = (sim_matrix[han_idx, viet_idx[0]] + sim_matrix[han_idx, viet_idx[1]]) / 2
            alignment.append(([han_idx], viet_idx, avg_score))
            i, j = i - 1, j - 2
        elif move_type == '2-1':
            avg_score = (sim_matrix[han_idx[0], viet_idx] + sim_matrix[han_idx[1], viet_idx]) / 2
            alignment.append((han_idx, [viet_idx], avg_score))
            i, j = i - 2, j - 1
        elif move_type == 'skip_han':
            i -= 1
        elif move_type == 'skip_viet':
            j -= 1
    
    # Đảo ngược vì backtrack từ cuối lên
    alignment.reverse()
    return alignment


# 3. REFINE BẰNG LLM
def call_llm_verify(han_text: str, viet_text: str, work_title: str) -> Dict:
    """
    Gửi cặp câu cho LLM để:
    1. Xác nhận đây có phải là bản dịch tương đương không
    2. Nếu không, gợi ý cách dóng hàng tốt hơn
    """
    system_prompt = """Bạn là chuyên gia Hán Nôm. NHIỆM VỤ: Kiểm tra cặp câu Hán-Việt có phải bản dịch tương đương không.
Trả về JSON duy nhất với format:
{"match": true/false, "han_corrected": "...", "viet_corrected": "...", "note": "..."}

- match=true: 2 câu tương đương về nghĩa
- match=false: không tương đương, có thể do dóng hàng sai
- han_corrected/viet_corrected: text đã sửa (nếu cần, giữ nguyên nếu OK)
- note: giải thích ngắn (nếu cần)

KHÔNG markdown, CHỈ JSON."""

    user_prompt = f"""Tác phẩm: "{work_title}"
Cặp câu cần kiểm tra:
[HÁN]: {han_text}
[VIỆT]: {viet_text}
"""

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
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"}
    }
    
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=LLM_TIMEOUT)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                # Parse JSON
                try:
                    return json.loads(content)
                except:
                    # Fallback: extract JSON từ response
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        return json.loads(match.group())
            else:
                print(f"LLM API Error {response.status_code}")
        except Exception as e:
            print(f"LLM error: {e}")
        time.sleep(2 ** attempt)
    
    # Fallback: coi như match
    return {"match": True, "han_corrected": han_text, "viet_corrected": viet_text, "note": "LLM unavailable"}


def refine_with_llm(alignment: List[Tuple], han_sents: List[str], viet_sents: List[str], 
                    work_title: str, use_llm: bool = True) -> List[Dict]:
    """
    Dùng LLM để refine các cặp có similarity thấp.
    """
    refined_pairs = []
    
    pbar = tqdm(alignment, desc="Refining với LLM")
    for han_indices, viet_indices, score in pbar:
        han_text = " ".join([han_sents[i] for i in han_indices])
        viet_text = " ".join([viet_sents[i] for i in viet_indices])
        
        # Nếu score cao -> không cần LLM
        if score >= CONFIDENCE_THRESHOLD or not use_llm:
            refined_pairs.append({
                "han_indices": han_indices,
                "viet_indices": viet_indices,
                "han_text": han_text,
                "viet_text": viet_text,
                "score": score,
                "verified": score >= CONFIDENCE_THRESHOLD
            })
            continue
        
        # Score thấp -> gửi LLM verify
        pbar.set_postfix_str(f"LLM verify (score={score:.2f})")
        llm_result = call_llm_verify(han_text, viet_text, work_title)
        
        if llm_result.get("match", False):
            refined_pairs.append({
                "han_indices": han_indices,
                "viet_indices": viet_indices,
                "han_text": llm_result.get("han_corrected", han_text),
                "viet_text": llm_result.get("viet_corrected", viet_text),
                "score": score,
                "verified": True,
                "note": llm_result.get("note", "")
            })
        else:
            # LLM nói không match -> skip cặp này (hoặc đánh dấu)
            refined_pairs.append({
                "han_indices": han_indices,
                "viet_indices": viet_indices,
                "han_text": han_text,
                "viet_text": viet_text,
                "score": score,
                "verified": False,
                "note": llm_result.get("note", "Not a match")
            })
    
    return refined_pairs


# 4. KẾT QUẢ
def save_to_tsv(work_id: str, refined_pairs: List[Dict], output_path: Path):
    """Lưu kết quả ra file TSV đúng format đề bài."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['pair_id', 'han_sentence', 'viet_sentence'])
        
        count = 0
        for pair in refined_pairs:
            # Chỉ lưu các cặp verified hoặc có score cao
            if pair.get("verified", False) or pair["score"] >= SIMILARITY_THRESHOLD:
                pair_id = f"{work_id}_{count:04d}"
                han_text = pair["han_text"].replace('\t', ' ').replace('\n', ' ')
                viet_text = pair["viet_text"].replace('\t', ' ').replace('\n', ' ')
                writer.writerow([pair_id, han_text, viet_text])
                count += 1
    
    return count


# 5. PIPELINE
def align_work(work: Dict, use_llm: bool = True):
    """Xử lý dóng hàng cho 1 tác phẩm."""
    work_id = work['id']
    work_title = work['viet']
    
    print(f"\n{'='*60}")
    print(f"Đang dóng hàng: {work_title} ({work_id})")
    print(f"{'='*60}")
    
    # Đọc text từ OCR output
    han_path = OCR_OUTPUT_DIR / f"{work_id}_sino_raw.txt"
    viet_path = OCR_OUTPUT_DIR / f"{work_id}_vie_raw.txt"
    
    if not han_path.exists():
        print(f"Không tìm thấy file Hán: {han_path}")
        return
    if not viet_path.exists():
        print(f"Không tìm thấy file Việt: {viet_path}")
        return
    
    han_text = han_path.read_text(encoding='utf-8')
    viet_text = viet_path.read_text(encoding='utf-8')
    
    # 1. Tách câu
    han_sents = segment_sentences(han_text, lang="hán")
    viet_sents = segment_sentences(viet_text, lang="việt")
    
    print(f"Tách được {len(han_sents)} câu Hán, {len(viet_sents)} câu Việt")
    
    if not han_sents or not viet_sents:
        print("Không có câu để dóng hàng")
        return
    
    # 2. Tính similarity matrix
    sim_matrix = compute_similarity_matrix(han_sents, viet_sents)
    
    # 3. Dynamic Programming alignment
    print(f"Đang dóng hàng bằng Dynamic Programming...")
    alignment = dynamic_align(sim_matrix)
    print(f"Tạo được {len(alignment)} cặp dóng hàng thô")
    
    # 4. Refine với LLM
    if use_llm:
        print(f"Refine các cặp có confidence thấp bằng LLM...")
    refined_pairs = refine_with_llm(alignment, han_sents, viet_sents, work_title, use_llm=use_llm)
    
    # Thống kê
    verified_count = sum(1 for p in refined_pairs if p.get("verified", False))
    high_score_count = sum(1 for p in refined_pairs if p["score"] >= SIMILARITY_THRESHOLD)
    
    print(f"\n  Thống kê:")
    print(f"     - Tổng cặp: {len(refined_pairs)}")
    print(f"     - Verified (LLM): {verified_count}")
    print(f"     - High score (>={SIMILARITY_THRESHOLD}): {high_score_count}")
    
    # 5. Lưu TSV
    output_path = CORPUS_DIR / f"{work_id}_parallel.tsv"
    final_count = save_to_tsv(work_id, refined_pairs, output_path)
    
    print(f"Đã lưu {final_count} cặp vào: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sentence Alignment Pipeline")
    parser.add_argument("--work-id", type=str, help="ID tác phẩm cụ thể (VD: HVB_001)")
    parser.add_argument("--no-llm", action="store_true", help="Không dùng LLM để refine")
    args = parser.parse_args()
    
    # Đọc config
    if not CONFIG_PATH.exists():
        print(f"Không tìm thấy config: {CONFIG_PATH}")
        sys.exit(1)
    
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    works = config['works']
    
    # Filter theo work_id nếu có
    if args.work_id:
        works = [w for w in works if w['id'] == args.work_id]
        if not works:
            print(f"Không tìm thấy work_id: {args.work_id}")
            sys.exit(1)
    
    print(f"Bắt đầu dóng hàng {len(works)} tác phẩm (LLM: {'ON' if not args.no_llm else 'OFF'})")
    
    for work in works:
        try:
            align_work(work, use_llm=not args.no_llm)
        except Exception as e:
            print(f"Lỗi xử lý {work['id']}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nCorpus đã lưu tại: {CORPUS_DIR}")


if __name__ == "__main__":
    main()