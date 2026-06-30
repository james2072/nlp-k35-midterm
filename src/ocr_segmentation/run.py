"""
Entrypoint for OCR + Segmentation pipeline.

Usage:
    python src/ocr_segmentation/run.py
    python src/ocr_segmentation/run.py --work-id an-nam-chi-luoc
    python src/ocr_segmentation/run.py --work-id an-nam-chi-luoc --lang han
"""

import argparse
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from ocr_segmentation import config_loader, pdf_to_images, ocr_engine, bbox_sort, segmenter
import fitz  # PyMuPDF to get page count

def main():
    parser = argparse.ArgumentParser(description="OCR + Segmentation Pipeline for HVB Corpus")
    parser.add_argument("--work-id", type=str, help="Specific work ID to process")
    parser.add_argument("--lang", type=str, choices=["han", "viet"], help="Specific language to process")
    args = parser.parse_args()

    # Load and validate config
    config_path = os.path.join(os.getcwd(), "data", "config.json")
    try:
        config = config_loader.load_config(config_path)
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)

    works = config.get("works", [])
    if args.work_id:
        works = [w for w in works if w.get("id") == args.work_id]
        if not works:
            print(f"Work ID '{args.work_id}' not found.")
            sys.exit(1)

    for work in works:
        work_id = work["id"]
        print(f"\nProcessing work: {work['name']} ({work_id})")
        
        resources = work.get("resources", [])
        if args.lang:
            resources = [r for r in resources if r.get("lang") == args.lang]

        for res in resources:
            lang = res["lang"]
            fmt = res["format"]
            path = res["path"]
            pages = res.get("pages")
            
            print(f"  Language: {lang} | Format: {fmt} | Path: {path}")
            
            lines = []
            bboxes = None
            
            if fmt == "text":
                # Read direct text, skip OCR
                lines = utils.read_lines(path)
                print(f"    Skipping OCR, read {len(lines)} lines from text file.")
            elif fmt in ["pdf", "mixed_page"]:
                # Convert PDF pages to temporary images
                print(f"    Converting PDF to images...")
                try:
                    images = pdf_to_images.convert(path, pages=pages, dpi=200)
                    print(f"    Rendered {len(images)} page images.")
                    
                    # Parse page indices for text layer extraction
                    doc = fitz.open(path)
                    total_pages = len(doc)
                    page_indices = pdf_to_images.parse_page_range(pages, total_pages)
                    doc.close()
                    
                    # Run OCR / text layer extraction
                    lines, bboxes = ocr_engine.run(
                        images, lang, fmt, pdf_path=path, page_indices=page_indices
                    )
                except Exception as e:
                    print(f"    Error processing PDF {path}: {e}")
                    continue
            elif fmt == "images":
                # Read images from directory
                if not os.path.exists(path) or not os.path.isdir(path):
                    print(f"    Image directory not found: {path}")
                    continue
                # Collect sorted list of image file paths
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
                images = []
                for f in sorted(os.listdir(path)):
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        images.append(os.path.abspath(os.path.join(path, f)))
                print(f"    Found {len(images)} images in directory.")
                lines, bboxes = ocr_engine.run(images, lang, fmt)
            else:
                print(f"    Unsupported format: {fmt}")
                continue

            # Sort Hán bounding boxes if applicable
            if lang == "han" and bboxes:
                print(f"    Sorting Hán bounding boxes...")
                lines = bbox_sort.sort(lines, bboxes)

            # Write raw OCR / read lines
            txt_output_path = os.path.join("data", "ocr_output", work_id, f"{lang}.txt")
            utils.write_lines(lines, txt_output_path)
            print(f"    Saved raw text to: {txt_output_path}")

            # Segment into sentences
            print(f"    Segmenting text into sentences...")
            sentences = segmenter.segment(lines, lang)
            
            # Format and save JSON output
            json_output_path = os.path.join("data", "processed", work_id, f"{lang}.json")
            json_data = {
                "work_id": work_id,
                "lang": lang,
                "total_sentences": len(sentences),
                "sentences": [{"idx": i, "text": s} for i, s in enumerate(sentences)]
            }
            utils.write_json(json_data, json_output_path)
            print(f"    Saved {len(sentences)} sentences to: {json_output_path}")

if __name__ == "__main__":
    main()

