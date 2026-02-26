import os
import logging
from io import StringIO
from typing import List, Dict
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextContainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _extract_text_by_page(pdf_path: str) -> List[str]:
    pages_text: List[str] = []
    logger.info(f"[pdf_processor] Extracting pages: {pdf_path}")
    try:
        for page_layout in extract_pages(pdf_path, laparams=LAParams()):
            buf = StringIO()
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    buf.write(element.get_text())
            pages_text.append(buf.getvalue())
        logger.info(f"[pdf_processor] {os.path.basename(pdf_path)}: pages={len(pages_text)}")
    except Exception as e:
        logger.error(f"[pdf_processor] Error extracting pages from {pdf_path}: {e}")
    return pages_text


def chunk_text(text: str, chunk_size=3000, overlap=1000) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def process_single_pdf(file_path: str) -> List[Dict]:
    filename = os.path.basename(file_path)
    pages = _extract_text_by_page(file_path)
    out: List[Dict] = []
    total = 0
    for page_idx, page_text in enumerate(pages, start=1):
        page_chunks = chunk_text(page_text, chunk_size=900, overlap=120)
        total += len(page_chunks)
        for i, chunk in enumerate(page_chunks):
            out.append({
                "text": chunk,
                "source": filename,
                "page": page_idx,
                "chunk_id": i,
            })
    logger.info(f"[pdf_processor] {filename}: final chunk count={total}")
    return out
