"""
Entrypoint for Sentence Alignment pipeline.

Usage:
    python src/sentence_alignment/run.py
    python src/sentence_alignment/run.py --work-id an-nam-chi-luoc
    python src/sentence_alignment/run.py --method labse
    python src/sentence_alignment/run.py --method auto_translate
    python src/sentence_alignment/run.py --method greedy
"""

import argparse
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from ocr_segmentation import config_loader
from sentence_alignment import aligner, char_aligner, corpus_builder

def main():
    parser = argparse.ArgumentParser(description="Sentence Alignment Pipeline for HVB Corpus")
    parser.add_argument("--work-id", type=str, help="Specific work ID to process")
    parser.add_argument("--method", type=str, choices=["labse", "auto_translate", "greedy"], help="Alignment method")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.getcwd(), "data", "config.json")
    try:
        config = config_loader.load_config(config_path)
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)

    # Determine alignment method
    method = args.method
    if not method:
        pipeline_cfg = aligner.get_pipeline_config()
        method = pipeline_cfg.get("alignment", {}).get("default_method", "labse")
    print(f"Using alignment method: {method}")

    works = config.get("works", [])
    if args.work_id:
        works = [w for w in works if w.get("id") == args.work_id]
        if not works:
            print(f"Work ID '{args.work_id}' not found.")
            sys.exit(1)

    tsv_paths = []
    
    # Process each work
    # We find the 1-based index of the work in the original config list
    original_works = config_loader.load_config(config_path).get("works", [])
    work_id_to_index = {w["id"]: idx + 1 for idx, w in enumerate(original_works)}

    for work in works:
        work_id = work["id"]
        work_index = work_id_to_index.get(work_id, 1)
        print(f"\nAligning work: {work['name']} ({work_id})")
        
        han_json_path = os.path.join("data", "processed", work_id, "han.json")
        viet_json_path = os.path.join("data", "processed", work_id, "viet.json")
        
        if not os.path.exists(han_json_path) or not os.path.exists(viet_json_path):
            print(f"    Missing processed files: {han_json_path} or {viet_json_path}. Run Part 1 first.")
            continue
            
        try:
            han_data = utils.read_json(han_json_path)
            viet_data = utils.read_json(viet_json_path)
            
            han_sentences = [s["text"] for s in han_data.get("sentences", [])]
            viet_sentences = [s["text"] for s in viet_data.get("sentences", [])]
            
            print(f"    Loaded {len(han_sentences)} Hán sentences, {len(viet_sentences)} Việt sentences.")
            
            # Run alignment
            pairs = aligner.align(han_sentences, viet_sentences, method)
            
            # Character verification if enabled
            pipeline_cfg = aligner.get_pipeline_config()
            char_align_enabled = pipeline_cfg.get("char_aligner", {}).get("enabled", True)
            if char_align_enabled:
                print("    Running character-level verification...")
                pairs = char_aligner.verify(pairs)
                # Print sample character verification results
                if pairs:
                    sample = pairs[0]
                    print(f"      Sample char alignment verification: {sample['char_status'][:10]}...")
            
            # Build TSV rows
            rows = corpus_builder.build(pairs, work_id, work_index)
            
            # Write individual TSV
            output_tsv = os.path.join("data", "corpus", work_id, "HVB_corpus.tsv")
            corpus_builder.write_tsv(rows, output_tsv)
            print(f"    Saved aligned corpus to: {output_tsv}")
            tsv_paths.append(output_tsv)
            
        except Exception as e:
            print(f"    Failed to align work {work_id}: {e}")

    # Merge all individual TSVs
    if tsv_paths:
        merged_tsv = os.path.join("data", "corpus", "HVB_corpus_all.tsv")
        print(f"\nMerging all individual TSVs into: {merged_tsv}")
        corpus_builder.merge_tsv(tsv_paths, merged_tsv)
        print("    Merging complete.")

if __name__ == "__main__":
    main()

