import os, logging
from tqdm import tqdm
from vector_store import VectorStore
from embedding_generator import EmbeddingGenerator
from tabular_processor import process_csv_file, process_xlsx_file
from pdf_processor import process_single_pdf
from text_processor import process_text_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CHROMA_STORE = "chroma_store"


def _preflight():
    try:
        eg = EmbeddingGenerator()
        eg.generate_embeddings_batch(["ping1", "ping2"])
        logger.info("Embedding preflight OK.")
    except Exception as e:
        raise RuntimeError(f"Embedding preflight failed: {e}")


def index_single_file(path: str, db: str = CHROMA_STORE, batch: int = 17):
    _preflight()

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    filename = os.path.basename(path)

    if ext == ".pdf":
        chunks = process_single_pdf(path)
    elif ext == ".csv":
        chunks = process_csv_file(path)
    elif ext == ".xlsx":
        chunks = process_xlsx_file(path)
    elif ext == ".txt":
        chunks = process_text_file(path)
    else:
        raise ValueError("Unsupported format.")

    logger.info(f"{filename}: {len(chunks)} chunks to embed.")
    if not chunks:
        logger.warning(f"{filename}: no chunks found.")
        return

    eg = EmbeddingGenerator()
    store = VectorStore(1536)

    try:
        store.load(db)
        logger.info("Loaded existing Chroma store.")
    except Exception:
        logger.info("No existing Chroma store; creating new.")

    total = len(chunks)
    added = 0

    for start in tqdm(range(0, total, batch), desc="Embedding"):
        end = min(start + batch, total)
        batch_chunks = chunks[start:end]
        texts = [c["text"] for c in batch_chunks]
        vecs = eg.generate_embeddings_batch(texts)
        store.add_vectors(vecs, batch_chunks)
        added += len(vecs)

    # Chroma auto-persists, but calling save() keeps the interface consistent
    store.save(db)
    logger.info(f"Indexed {added} vectors from {filename}.")
