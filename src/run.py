import os
import sys
import json
import csv
import argparse
import time
from pathlib import Path

try:
    import pandas as pd
    from google import genai
    from google.genai import types
    import typing_extensions as typing
except ImportError:
    print("Please install requirements: pip install google-genai pydantic typing_extensions pandas")
    sys.exit(1)

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# API CONFIGURATION
# ============================================================
API_KEY = os.getenv("LLM_API_KEY", "")
if not API_KEY:
    print("Error: Missing LLM_API_KEY in .env")
    sys.exit(1)
    
client = genai.Client(api_key=API_KEY)
# Read the model name from .env, default to gemini-1.5-flash
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")
if not MODEL_NAME:
    print("Error: Missing LLM_MODEL_NAME in .env")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OCR_OUTPUT_DIR = DATA_DIR / "ocr_output"
CORPUS_DIR = DATA_DIR / "corpus"

CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# REQUIRED STRUCTURED OUTPUT SCHEMA
# ============================================================
class SentencePair(typing.TypedDict):
    han_sentence: str
    viet_sentence: str

class AlignmentResult(typing.TypedDict):
    pairs: list[SentencePair]

# ============================================================
# LLM API LOGIC
# ============================================================
def call_gemini_structured(system_instruction: str, user_prompt: str) -> list:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=AlignmentResult,
                    temperature=0.0
                )
            )
            data = json.loads(response.text)
            return data.get("pairs", [])
        except Exception as e:
            print(f"    [Error] Gemini API failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return []

# ============================================================
# MAIN PIPELINE
# ============================================================
def process_alignment(work_id: str, id_start: int, n: int, k: int):
    h_path = OCR_OUTPUT_DIR / f"{work_id}_sino_raw.txt"
    v_path = OCR_OUTPUT_DIR / f"{work_id}_vie_raw.txt"
    
    if not h_path.exists() or not v_path.exists():
        print(f"Error: Missing raw files for {work_id} at {h_path} or {v_path}")
        return

    # Split by \n\n (assuming chunks are already present in the files)
    h_raw = h_path.read_text(encoding='utf-8')
    v_raw = v_path.read_text(encoding='utf-8')
    
    h_chunks = [c.strip() for c in h_raw.split('\n\n') if c.strip()]
    v_chunks = [c.strip() for c in v_raw.split('\n\n') if c.strip()]
    
    total_chunks = max(len(h_chunks), len(v_chunks))
    print(f"Found {len(h_chunks)} Han chunks and {len(v_chunks)} Viet chunks.")
    
    # Updated System Prompt as requested
    system_instruction = (
        "You are an expert in Sino-Nom linguistics and Vietnamese history.\n"
        "Your task is to align sentence pairs (sentence alignment) between Chinese (Han) and Vietnamese texts from the provided chunks.\n"
        "STRICT RULES:\n"
        "1. GROUND TRUTH IS HAN: The Han text is the ground truth. You must find the exact corresponding Vietnamese sentence to align with the Han sentence.\n"
        "2. DETECT GENRE: The text may contain a mix of prose and poetry. You must automatically detect which part is poetry and which is prose to segment the sentences appropriately.\n"
        "3. FINE-GRAINED SEGMENTATION: DO NOT output giant, long blocks of text. If a Han paragraph or sentence is long, you MUST break it down into shorter, bite-sized sub-clauses (e.g., splitting at commas, semicolons, or logical pauses) and align each small segment precisely with its corresponding Vietnamese translation.\n"
        "4. FIX OCR & PUNCTUATION: Automatically correct spelling mistakes and OCR noise in BOTH Han and Viet texts, and standardize the punctuation grammatically. YOU MUST NOT alter the original meaning.\n"
        "5. FILTER NOISE: IF a Han sentence exists but no corresponding Vietnamese sentence can be found (due to missing sections or OCR errors), DROP that sentence.\n"
        "6. NO DUPLICATES: Never output the same sentence twice. If the source text contains duplicated paragraphs/sentences due to OCR repetition, process only the first occurrence and DROP the duplicates.\n"
        "7. MAINTAIN ORDER: You MUST preserve the exact top-to-bottom order of the sentences based on the original Han text."
    )
    
    all_pairs = []
    seen_han = set() # To strictly prevent duplicates across the entire run
    current_id_num = id_start
    
    # Process in batches of N chunks
    for i in range(0, total_chunks, n):
        h_batch = h_chunks[i : i + n]
        v_batch = v_chunks[i : i + n]
        
        # Get K chunks overlap for context (if available)
        prev_h = h_chunks[max(0, i - k) : i] if k > 0 and i > 0 else []
        prev_v = v_chunks[max(0, i - k) : i] if k > 0 and i > 0 else []
        
        user_prompt = f"I need to align {len(h_batch)} chunk(s).\n\n"
        if prev_h or prev_v:
            user_prompt += "=== CONTEXT (PREVIOUS K CHUNKS - FOR REFERENCE ONLY, DO NOT ALIGN OR INCLUDE IN JSON) ===\n"
            user_prompt += "[HAN CONTEXT]:\n" + "\n\n".join(prev_h) + "\n\n"
            user_prompt += "[VIET CONTEXT]:\n" + "\n\n".join(prev_v) + "\n\n"
            
        user_prompt += "=== CHUNKS TO ALIGN (RETURN JSON FOR THESE SENTENCES) ===\n"
        user_prompt += "[HAN TEXT]:\n" + "\n\n".join(h_batch) + "\n\n"
        user_prompt += "[VIET TEXT]:\n" + "\n\n".join(v_batch)
        
        print(f"Processing batch chunk {i+1} to {min(total_chunks, i+n)} of {total_chunks}...")
        
        aligned_pairs = call_gemini_structured(system_instruction, user_prompt)
        
        if not aligned_pairs:
            print("  -> Warning: No pairs returned or JSON parse failed.")
            continue
            
        for pair in aligned_pairs:
            h_sent = pair.get("han_sentence", "").strip().replace('\n', ' ')
            v_sent = pair.get("viet_sentence", "").strip().replace('\n', ' ')
            
            # Check: if Han or Viet is empty, skip (adhering to noise filtering rule)
            # Also strictly prevent duplicate Han sentences
            if h_sent and v_sent and h_sent not in seen_han:
                seen_han.add(h_sent)
                pair_id = f"{work_id}_{str(current_id_num).zfill(4)}"
                all_pairs.append({
                    "pair_id": pair_id,
                    "han_sentence": h_sent,
                    "viet_sentence": v_sent
                })
                current_id_num += 1
                
        print(f"  -> Extracted {len(aligned_pairs)} pairs. (Current ID: {work_id}_{str(current_id_num-1).zfill(4)})")
        
        # Save TSV incrementally
        tsv_path = CORPUS_DIR / f"{work_id}_parallel.tsv"
        with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["pair_id", "han_sentence", "viet_sentence"], delimiter='\t')
            writer.writeheader()
            writer.writerows(all_pairs)
            
        # Save XLSX incrementally
        xlsx_path = CORPUS_DIR / f"{work_id}_parallel.xlsx"
        pd.DataFrame(all_pairs).to_excel(xlsx_path, index=False)
        print(f"  -> Saved intermediate progress to TSV and XLSX.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Google AI Studio Alignment with Structured Output")
    parser.add_argument("--work-id", type=str, required=True, help="Work ID (e.g. HVB_005)")
    parser.add_argument("--id-start", type=int, default=0, help="Starting ID (e.g. 0 -> HVB_005_0000)")
    parser.add_argument("--n", type=int, default=5, help="Number of chunks to align per batch (default: 5)")
    parser.add_argument("--k", type=int, default=1, help="Number of chunks to use as overlap context (default: 1)")
    
    args = parser.parse_args()
    process_alignment(args.work_id, args.id_start, args.n, args.k)
