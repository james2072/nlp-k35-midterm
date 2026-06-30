"""
Character-level alignment verification using:
  - MED (Minimum Edit Distance / Levenshtein)
  - SinoNom_Similar.dic  (visually similar Han characters)
  - QuocNgu_SinoNom.dic  (Quoc ngu → SinoNom mapping)
"""


import os
import re

# Global cache for dictionaries
_similar_dic = None
_quocngu_dic = None

def load_dic(path: str) -> dict:
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                parts = line.split(":", 1)
                key = parts[0].strip()
                val_str = parts[1].strip()
                vals = [v.strip() for v in val_str.replace(",", " ").split() if v.strip()]
                mapping[key] = vals
    return mapping


def get_dictionaries():
    global _similar_dic, _quocngu_dic
    if _similar_dic is None:
        _similar_dic = load_dic("dictionaries/SinoNom_Similar.dic")
    if _quocngu_dic is None:
        _quocngu_dic = load_dic("dictionaries/QuocNgu_SinoNom.dic")
    return _similar_dic, _quocngu_dic


def is_han_char(c: str) -> bool:
    o = ord(c)
    return (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF) or (0x20000 <= o <= 0x2A6DF)


def verify(pairs: list[dict]) -> list[dict]:
    """
    Annotate each pair with per-character OCR confidence.

    Args:
        pairs: List of {"han": str, "viet": str, "score": float}

    Returns:
        Same list with added "char_status" field per character:
        "ok" | "corrected" | "error"
    """
    similar_dic, quocngu_dic = get_dictionaries()
    
    annotated_pairs = []
    for pair in pairs:
        han = pair["han"]
        viet = pair["viet"]
        
        char_annotations = _verify_pair(han, viet, similar_dic, quocngu_dic)
        
        new_pair = pair.copy()
        new_pair["char_status"] = char_annotations
        annotated_pairs.append(new_pair)
        
    return annotated_pairs


def _verify_pair(han: str, viet: str, similar_dic: dict, quocngu_dic: dict) -> list[dict]:
    """
    Returns per-character annotation for a Han/Viet sentence pair.
    """
    # Filter Han characters (only CJK ideographs)
    H_indices = []
    H_chars = []
    for idx, c in enumerate(han):
        if is_han_char(c):
            H_indices.append(idx)
            H_chars.append(c)
            
    # Extract Viet syllables (lowercased)
    V_syllables = [w.lower() for w in re.findall(r'\b\w+\b', viet) if w.strip()]
    
    M = len(H_chars)
    N = len(V_syllables)
    
    # Initialize DP table for Levenshtein Distance
    # dp[i][j] stores the min cost to align H_chars[:i] with V_syllables[:j]
    dp = [[0.0] * (N + 1) for _ in range(M + 1)]
    parent = [[None] * (N + 1) for _ in range(M + 1)]
    
    for i in range(1, M + 1):
        dp[i][0] = i * 1.0
        parent[i][0] = (i - 1, 0, "delete")
    for j in range(1, N + 1):
        dp[0][j] = j * 1.0
        parent[0][j] = (0, j - 1, "insert")
        
    for i in range(1, M + 1):
        h = H_chars[i - 1]
        s1_h_list = similar_dic.get(h, [])
        s1_h = set(s1_h_list)
        s1_h.add(h)  # Include the character itself
        
        for j in range(1, N + 1):
            v = V_syllables[j - 1]
            s2_v = set(quocngu_dic.get(v, []))
            
            # Compute match / correction cost
            if h in s2_v:
                sub_cost = 0.0
            elif s1_h & s2_v:
                sub_cost = 0.5
            else:
                sub_cost = 2.0
                
            opt_del = dp[i - 1][j] + 1.0
            opt_ins = dp[i][j - 1] + 1.0
            opt_sub = dp[i - 1][j - 1] + sub_cost
            
            min_val = min(opt_del, opt_ins, opt_sub)
            dp[i][j] = min_val
            
            if min_val == opt_sub:
                op_name = "match" if sub_cost == 0.0 else "corrected" if sub_cost == 0.5 else "mismatch"
                parent[i][j] = (i - 1, j - 1, op_name)
            elif min_val == opt_del:
                parent[i][j] = (i - 1, j, "delete")
            else:
                parent[i][j] = (i, j - 1, "insert")
                
    # Backtrack to reconstruct alignment path
    curr_i, curr_j = M, N
    alignment_path = []
    while curr_i > 0 or curr_j > 0:
        p = parent[curr_i][curr_j]
        if p is None:
            break
        prev_i, prev_j, op = p
        if op != "insert":
            # For "match", "corrected", "mismatch", "delete", there is a Han character involved (curr_i - 1)
            alignment_path.append((curr_i - 1, curr_j - 1, op))
        curr_i, curr_j = prev_i, prev_j
        
    alignment_path.reverse()
    
    # Map alignment results to H_chars indices
    han_char_status = {}
    for idx_h, idx_v, op in alignment_path:
        h = H_chars[idx_h]
        if op == "match":
            han_char_status[idx_h] = {"status": "ok", "char": h}
        elif op == "corrected":
            v = V_syllables[idx_v]
            s1_h_list = similar_dic.get(h, [])
            s2_v = set(quocngu_dic.get(v, []))
            intersection = set(s1_h_list) & s2_v
            
            if len(intersection) >= 1:
                # Find the one with highest visual similarity (leftmost in s1_h_list)
                corrected_char = min(intersection, key=lambda c: s1_h_list.index(c) if c in s1_h_list else 999)
                han_char_status[idx_h] = {"status": "corrected", "char": corrected_char}
            else:
                han_char_status[idx_h] = {"status": "error", "char": h}
        else: # mismatch or delete
            han_char_status[idx_h] = {"status": "error", "char": h}
            
    # Prepare the final output mapping all characters in the original han string
    annotations = []
    for idx, c in enumerate(han):
        if not is_han_char(c):
            annotations.append({"char": c, "status": "ok"})
        else:
            # Look up filtered index
            if idx in H_indices:
                filtered_idx = H_indices.index(idx)
                status_info = han_char_status.get(filtered_idx, {"status": "error", "char": c})
                annotations.append({"char": status_info["char"], "status": status_info["status"]})
            else:
                annotations.append({"char": c, "status": "error"})
                
    return annotations

