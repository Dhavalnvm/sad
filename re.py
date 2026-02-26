import logging
from typing import List, Tuple

from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

        if page is None or page == "" or not show_page:
            return f"[SOURCE: {filename}]"
        return f"[SOURCE: {filename} p.{page}]"

    def _get_all_chunks(self) -> Tuple[str, List[str]]:
        """Fetch every chunk from the store, ordered by source and page."""
        all_meta = self.store.meta
        all_texts = self.store.texts

        if not all_texts:
            return "NO_RELEVANT", []

        # Sort by source filename then page number
        combined = list(zip(all_texts, all_meta))
        combined.sort(key=lambda x: (
            x[1].get("source") or "",
            int(x[1].get("page") or 0),
            int(x[1].get("chunk_id") or 0),
        ))

        context_blocks = []
        source_set = []
        seen_sources = {}

        for text, meta in combined:
            src_tag = self._format_source(meta, show_page=False)
            context_blocks.append(f"{src_tag}\n{text}")

            source = meta.get("source") or "unknown"
            if source not in seen_sources:
                seen_sources[source] = set()
            page = meta.get("page")
            if page and page != "":
                seen_sources[source].add(str(page))

        # Build source list
        for source, pages in seen_sources.items():
            if pages:
                sorted_pages = sorted(pages, key=lambda x: int(x) if x.isdigit() else 0)
                source_set.append(f"[SOURCE: {source} p.{', p.'.join(sorted_pages)}]")
            else:
                source_set.append(f"[SOURCE: {source}]")

        # Cap context to avoid exceeding LLM token limits (~12000 chars)
        MAX_CHARS = 12000
        context = "\n\n".join(context_blocks)
        if len(context) > MAX_CHARS:
            context = context[:MAX_CHARS] + "\n\n[... remaining content truncated for length ...]"

        logger.info("[Retriever] Summary mode: fetched %d chunks, %d chars.", len(combined), len(context))
        return context, source_set

    def retrieve(self, query: str, k: int = 5, show_page: bool = True) -> Tuple[str, List[str]]:
        if not self.store.texts:
            logger.warning("[Retriever] Empty vector store.")
            return "NO_RELEVANT", []

        # For summary queries, bypass similarity search and return all chunks
        is_summary = not show_page
        if is_summary:
            return self._get_all_chunks()

        q_emb = self._embed(query)
        results = self.store.search(q_emb, k=k)

        if not results:
            return "NO_RELEVANT", []

        best_distance = results[0]["score"]

        # Relax threshold for short queries
        if len(query.split()) <= 3:
            effective_threshold = self.max_distance * 1.5
        else:
            effective_threshold = self.max_distance

        if best_distance > effective_threshold:
            logger.warning(
                "[Retriever] Best distance %.3f exceeds threshold %.3f",
                best_distance,
                effective_threshold,
            )
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
