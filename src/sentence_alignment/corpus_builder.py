"""
Builds the final HVB corpus TSV from aligned sentence pairs.

ID format: HVB_fff.ccc.ppp.ss
  fff = file/work index (3 digits)
  ccc = chapter index (3 digits)
  ppp = page index (3 digits)
  ss  = sentence index within page (2 digits)
"""


import os
import sys

# Ensure utils can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

def build(pairs: list[dict], work_id: str, work_index: int) -> list[dict]:
    """
    Assign HVB IDs and format pairs as TSV rows.

    Args:
        pairs:      List of {"han": str, "viet": str, "score": float}
        work_id:    Work slug (e.g. "an-nam-chi-luoc")
        work_index: 1-based index of this work in config.

    Returns:
        List of dicts ready to write as TSV rows.
    """
    rows = []
    for idx, pair in enumerate(pairs):
        # chapter/page defaults to 1 unless metadata is available
        chapter = 1
        page = 1
        sent_idx = idx + 1
        
        id_str = utils.generate_id(work_index, chapter, page, sent_idx)
        
        rows.append({
            "id": id_str,
            "han_text": pair["han"],
            "viet_text": pair["viet"],
            "align_score": pair["score"]
        })
    return rows


def write_tsv(rows: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # Write header
        f.write("id\than_text\tviet_text\talign_score\n")
        # Write rows
        for row in rows:
            # Clean tabs and newlines from fields to preserve TSV format
            han = str(row["han_text"]).replace("\t", " ").replace("\n", " ").replace("\r", "")
            viet = str(row["viet_text"]).replace("\t", " ").replace("\n", " ").replace("\r", "")
            f.write(f"{row['id']}\t{han}\t{viet}\t{row['align_score']}\n")


def merge_tsv(input_paths: list[str], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("id\than_text\tviet_text\talign_score\n")
        for path in input_paths:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as in_f:
                lines = in_f.readlines()
                if len(lines) <= 1:
                    continue
                # Skip header line (index 0)
                for line in lines[1:]:
                    out_f.write(line)

