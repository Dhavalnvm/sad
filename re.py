import logging
from typing import List, Tuple

from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SUMMARY_KEYWORDS = {"summarize", "summary", "overview", "summarise", "what is in", "what's in"}


class Retriever:
    def __init__(self, db_path: str = "chroma_store", dimension: int = 1536, max_distance: float = 1.15):
        self.db_path = db_path
        self.dimension = dimension
        self.max_distance = max_distance

        self.store = VectorStore(dimension)
        self.eg = EmbeddingGenerator()

        try:
            self.store.load(self.db_path)
            logger.info("[Retriever] Loaded Chroma store with %d vectors.", len(self.store.texts))
        except Exception:
            logger.warning("[Retriever] No Chroma store found, starting empty.")

    def _embed(self, query: str) -> List[float]:
        return self.eg.generate_embedding(query)

    def _format_source(self, meta: dict, show_page: bool = True) -> str:
        filename = meta.get("source") or "unknown"
        page = meta.get("page")

        if page is None or not show_page:
            return f"[SOURCE: {filename}]"
        return f"[SOURCE: {filename} p.{page}]"

    def retrieve(self, query: str, k: int = 5, show_page: bool = True) -> Tuple[str, List[str]]:
        if not self.store.texts:
            logger.warning("[Retriever] Empty vector store.")
            return "NO_RELEVANT", []

        q_emb = self._embed(query)

        results = self.store.search(q_emb, k=k)
        if not results:
            return "NO_RELEVANT", []

        best_distance = results[0]["score"]

        if best_distance > self.max_distance:
            return "NO_RELEVANT", []

        context_blocks = []
        source_list = []

        for item in results:
            text = item.get("text", "")
            meta = item.get("meta", {})

            src_tag = self._format_source(meta, show_page=show_page)
            block = f"{src_tag}\n{text}"
            context_blocks.append(block)
            source_list.append(src_tag)

        context = "\n\n".join(context_blocks)
        return context, source_list
