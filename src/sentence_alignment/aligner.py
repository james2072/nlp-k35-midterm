"""
Sentence alignment: maps Han sentences to Viet sentences.

Three methods:
  1. labse          — LaBSE cosine similarity (recommended)
  2. auto_translate — Gemini translate Han→Viet then TF-IDF compare
  3. greedy         — greedy length-ratio fallback (no external deps)
"""


import os
import math
from collections import Counter

_labse_model = None

def get_labse_model():
    global _labse_model
    if _labse_model is None:
        from sentence_transformers import SentenceTransformer
        _labse_model = SentenceTransformer('LaBSE')
    return _labse_model


class SimpleTFIDF:
    def __init__(self, docs: list[str]):
        self.num_docs = len(docs)
        self.df = Counter()
        for doc in docs:
            words = set(doc.lower().split())
            for w in words:
                self.df[w] += 1
    
    def get_tfidf_vec(self, text: str) -> dict:
        words = text.lower().split()
        tf = Counter(words)
        vec = {}
        for w, count in tf.items():
            df_w = self.df.get(w, 0)
            idf = math.log(1.0 + self.num_docs / (df_w + 1.0))
            vec[w] = count * idf
        return vec
    
    def cosine_similarity(self, vec1: dict, vec2: dict) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[w] * vec2[w] for w in intersection)
        
        sum1 = sum(v**2 for v in vec1.values())
        sum2 = sum(v**2 for v in vec2.values())
        
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        return numerator / denominator


def align(
    han_sentences: list[str],
    viet_sentences: list[str],
    method: str = "labse",
) -> list[dict]:
    """
    Align Han sentences to Viet sentences.

    Returns:
        List of dicts: {"han": str, "viet": str, "score": float}
    """
    if not han_sentences:
        return []
    if not viet_sentences:
        return [{"han": h, "viet": "", "score": 0.0} for h in han_sentences]

    if method == "labse":
        try:
            return _align_labse(han_sentences, viet_sentences)
        except Exception as e:
            print(f"LaBSE alignment failed, falling back to greedy: {e}")
            return _align_greedy(han_sentences, viet_sentences)
    elif method == "auto_translate":
        return _align_auto_translate(han_sentences, viet_sentences)
    elif method == "greedy":
        return _align_greedy(han_sentences, viet_sentences)
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def _align_labse(han: list[str], viet: list[str]) -> list[dict]:
    model = get_labse_model()
    H = model.encode(han, convert_to_tensor=True)
    V = model.encode(viet, convert_to_tensor=True)
    
    from sentence_transformers import util
    cosine_scores = util.cos_sim(H, V)
    
    pairs = []
    for i in range(len(han)):
        best_j = cosine_scores[i].argmax().item()
        score = cosine_scores[i][best_j].item()
        pairs.append({"han": han[i], "viet": viet[best_j], "score": round(score, 4)})
    return pairs


def _align_auto_translate(han: list[str], viet: list[str]) -> list[dict]:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        print("Warning: GEMINI_API_KEY not found in .env, translation may fail.")
        
    model = genai.GenerativeModel("gemini-2.5-flash")
    translated = []
    
    print("    Translating Hán sentences using Gemini...")
    for idx, s in enumerate(han):
        prompt = f"Dịch câu chữ Hán sau đây sang tiếng Việt. Chỉ trả về kết quả dịch trực tiếp, không giải thích gì thêm:\n{s}"
        try:
            response = model.generate_content(prompt)
            translated.append(response.text.strip())
        except Exception as e:
            print(f"      Failed to translate line {idx}: {e}")
            translated.append("")

    tfidf = SimpleTFIDF(viet)
    viet_vecs = [tfidf.get_tfidf_vec(v) for v in viet]
    
    pairs = []
    for i, trans_h in enumerate(translated):
        if not trans_h:
            # Fallback to simple relative index if translation failed
            pos_j = min(len(viet) - 1, int(i * len(viet) / len(han)))
            pairs.append({"han": han[i], "viet": viet[pos_j], "score": 0.0})
            continue
            
        h_vec = tfidf.get_tfidf_vec(trans_h)
        best_j = 0
        best_score = -1.0
        
        for j, v_vec in enumerate(viet_vecs):
            # Calculate similarity combined with relative position distance
            sim = tfidf.cosine_similarity(h_vec, v_vec)
            pos_penalty = 1.0 - abs(i / len(han) - j / len(viet))
            score = sim * pos_penalty
            if score > best_score:
                best_score = score
                best_j = j
                
        pairs.append({"han": han[i], "viet": viet[best_j], "score": round(best_score, 4)})
    return pairs


def _align_greedy(han: list[str], viet: list[str]) -> list[dict]:
    total_han_len = sum(len(h) for h in han)
    total_viet_len = sum(len(v) for v in viet)
    target_ratio = total_han_len / total_viet_len if total_viet_len > 0 else 1.0

    pairs = []
    for i, h in enumerate(han):
        best_j = 0
        best_score = -1.0
        
        for j, v in enumerate(viet):
            ratio = len(h) / len(v) if len(v) > 0 else 0.0
            # Length ratio score: closer to target_ratio is better
            ratio_diff = abs(ratio - target_ratio)
            len_score = 1.0 / (1.0 + ratio_diff)
            
            # Positional score: closer relative positions is better
            pos_diff = abs(i / len(han) - j / len(viet))
            pos_score = 1.0 - pos_diff
            
            score = len_score * pos_score
            if score > best_score:
                best_score = score
                best_j = j
                
        pairs.append({"han": h, "viet": viet[best_j], "score": round(best_score, 4)})
    return pairs

