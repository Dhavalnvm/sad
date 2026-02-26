import os


def chunk_text(text: str, chunk_size=900, overlap=120):
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


def process_text_file(file_path: str):
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    page_chunks = chunk_text(text, chunk_size=900, overlap=120)
    out = []
    for i, chunk in enumerate(page_chunks):
        out.append({
            "text": chunk,
            "source": filename,
            "page": None,
            "chunk_id": i,
        })
    return out
