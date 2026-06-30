"""
Converts PDF pages to JPG images using PyMuPDF (fitz).
"""


import fitz  # PyMuPDF
import os
import tempfile

def convert(pdf_path: str, pages: str | None = None, dpi: int = 200) -> list[str]:
    """
    Convert PDF pages to temporary JPG images.

    Args:
        pdf_path: Path to the PDF file.
        pages:    Page range string ("1-80", "1,3,5") or None for all pages.
        dpi:      Resolution for rendering.

    Returns:
        List of paths to temporary image files.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    page_indices = parse_page_range(pages, total_pages)
    
    # Use a temp directory inside the project root for safety and easy cleanup
    temp_dir = os.path.join(os.getcwd(), "data", "temp_images")
    os.makedirs(temp_dir, exist_ok=True)
    
    image_paths = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for idx in page_indices:
        if idx < 0 or idx >= total_pages:
            continue
        page = doc[idx]
        pix = page.get_pixmap(matrix=matrix)
        
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        img_name = f"{pdf_name}_page_{idx + 1:04d}.jpg"
        img_path = os.path.join(temp_dir, img_name)
        
        pix.save(img_path)
        image_paths.append(img_path)
        
    doc.close()
    return image_paths


def parse_page_range(pages: str | None, total: int) -> list[int]:
    """
    Parse a page range string into a list of 0-indexed page numbers.

    Args:
        pages: "1-80", "1,3,5", or None (all pages).
        total: Total number of pages in the PDF.

    Returns:
        List of 0-indexed page numbers.
    """
    if not pages:
        return list(range(total))
        
    indices = []
    tokens = pages.split(",")
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-")
            if len(parts) == 2:
                try:
                    start = int(parts[0].strip()) - 1
                    end = int(parts[1].strip()) - 1
                    indices.extend(range(max(0, start), min(total, end + 1)))
                except ValueError:
                    pass
        else:
            try:
                val = int(token) - 1
                if 0 <= val < total:
                    indices.append(val)
            except ValueError:
                pass
    return sorted(list(set(indices)))

