import json
from pathlib import Path
from typing import Dict, List

def export_debug_matrix(work_id: str, pairs: List[Dict], data_dir: Path):
    """Xuất file debug matrix để theo dõi kết quả của thuật toán DP Bead."""
    debug_dir = data_dir.parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"{work_id}_bead_trace.txt"
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"DEBUG ALIGNMENT MATRIX REPORT: {work_id}")
    lines.append("=" * 80)
    lines.append("Each bead represents a chunk of Han aligned to Viet.")
    lines.append("Score shows the semantic + structural similarity confidence.\n")
    
    for i, pair in enumerate(pairs):
        lines.append(f"[{pair.get('pair_id', f'{work_id}_{i:04d}')}] - Score: {pair.get('score', 0):.3f}")
        lines.append(f"Han : {pair.get('han', '').strip()}")
        lines.append(f"Viet: {pair.get('viet', '').strip()}")
        lines.append("-" * 60)
        
    debug_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[DEBUG] - Matrix trace exported: {debug_path.name}")

def export_debug_segmentation(work: Dict, han_raw: str, viet_raw: str, is_chunked: bool, han_chunks: List[str], viet_chunks: List[str], 
                              data_dir: Path, segment_han_fn, segment_viet_fn, segment_han_verse_fn, segment_viet_verse_fn):
    """Xuất file debug kết quả tách câu để kiểm tra ngữ đoạn."""
    work_id = work['id']
    genre = work.get("genre", "prose")
    debug_dir = data_dir.parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"{work_id}_debug_sentences.txt"
    lines = []
    lines.append("=" * 80)
    lines.append(f"DEBUG SEGMENTATION REPORT: {work_id} ({work.get('viet', '')})")
    lines.append(f"Genre: {genre.upper()}")
    lines.append("=" * 80)
    
    if genre == "poetry":
        h_s = segment_han_verse_fn(han_raw)
        v_s = segment_viet_verse_fn(viet_raw)
        lines.append(f"\nMode: Verse / Poetry (Line-by-Line Couplets)\n")
        lines.append(f"--- [HAN / NOM VERSE LINES] (Total: {len(h_s)}) ---")
        for i, s in enumerate(h_s):
            lines.append(f"  [H_{i+1:04d}] (len={len(s):02d}c) {s}")
        lines.append(f"\n--- [VIET LUC BAT COUPLETS] (Total: {len(v_s)}) ---")
        for j, s in enumerate(v_s):
            lines.append(f"  [V_{j+1:04d}] (len={len(s):03d}c) {s}")
        lines.append(f"\n{'='*80}")
        lines.append(f"SUMMARY: Total Han = {len(h_s)} verse lines | Total Viet = {len(v_s)} couplets | Ratio = 1:{len(v_s)/max(len(h_s), 1):.2f}")
        lines.append("=" * 80)
    elif is_chunked:
        lines.append(f"\nMode: Chunk-wise ({len(han_chunks)} chunks)\n")
        total_h = 0
        total_v = 0
        for c_idx, (h_c, v_c) in enumerate(zip(han_chunks, viet_chunks)):
            h_s = segment_han_fn(h_c)
            v_s = segment_viet_fn(v_c)
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
        h_s = segment_han_fn(han_raw)
        v_s = segment_viet_fn(viet_raw)
        lines.append(f"\nMode: Document-level Prose\n")
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
    print(f"\n[DEBUG] - Debug file exported: {debug_path.name}")
    if genre == "poetry":
        print(f"        Genre: Poetry / Verse | Han: {len(h_s)} dòng thơ | Viet: {len(v_s)} cặp lục bát")
    elif is_chunked:
        print(f"        Mode: Chunk-wise ({len(han_chunks)} chunks) | Han: {total_h} câu | Viet: {total_v} câu")
    else:
        print(f"        Mode: Document-level | Han: {len(h_s)} câu | Viet: {len(v_s)} câu")
