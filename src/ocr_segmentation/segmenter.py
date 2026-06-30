"""
Sentence segmentation:
  - Han text  → HanLP sentence tokenizer
  - Viet text → Underthesea sent_tokenize
"""


import re

def segment(lines: list[str], lang: str) -> list[str]:
    """
    Tokenize a list of raw text lines into sentences.

    Args:
        lines: Raw OCR lines.
        lang:  "han" or "viet".

    Returns:
        List of sentence strings.
    """
    if not lines:
        return []
        
    text = " ".join(lines)
    
    if lang == "han":
        sentences = _segment_han(text)
    elif lang == "viet":
        sentences = _segment_viet(text)
    else:
        raise ValueError(f"Unsupported language: {lang}")
        
    return [s.strip() for s in sentences if s.strip()]


def _segment_han(text: str) -> list[str]:
    # Rule-based Chinese sentence splitter
    # Chinese sentence terminators: 。 ！ ？ ； \n
    # We use a regex that splits and keeps the punctuation.
    try:
        # Try importing hanlp and using its split_sentence if available
        import hanlp
        # Some versions have hanlp.utils.rules.split_sentence
        # If not, or if model load fails, we fall back to rule-based.
        try:
            return hanlp.utils.rules.split_sentence(text)
        except AttributeError:
            pass
    except ImportError:
        pass
        
    # Regex fallback
    pattern = re.compile(r'([^。！？；\n]+[。！？；\n]*)')
    sentences = pattern.findall(text)
    return sentences


def _segment_viet(text: str) -> list[str]:
    try:
        from underthesea import sent_tokenize
        return sent_tokenize(text)
    except ImportError:
        # Regex fallback: split by . ! ? \n
        pattern = re.compile(r'([^.!?\n]+[.!?\n]*)')
        sentences = pattern.findall(text)
        return sentences

