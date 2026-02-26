import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, dimension: int, collection_name: str = "vector_store"):
        self.dimension = dimension
        self.collection_name = collection_name
        self.texts = []
        self.meta = []
        self._client = None
        self._collection = None

    def _init_collection(self, folder: str):
        self._client = chromadb.PersistentClient(path=folder)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "l2"},
        )

    def add_vectors(self, vectors, metadata_list):
        if not vectors or self._collection is None:
            return

        ids = [str(len(self.texts) + i) for i in range(len(vectors))]
        embeddings = [v if isinstance(v, list) else v.tolist() for v in vectors]
        documents = [m["text"] for m in metadata_list]
        metadatas = []
        for m in metadata_list:
            clean = {}
            for k, v in m.items():
                if v is None:
                    clean[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            metadatas.append(clean)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        for m in metadata_list:
            self.texts.append(m["text"])
            self.meta.append(m)

    def save(self, folder: str = "chroma_store"):
        # ChromaDB PersistentClient auto-saves — no manual step needed.
        # Kept for API compatibility.
        if self._client is None:
            self._init_collection(folder)

    def load(self, folder: str = "chroma_store"):
        self._init_collection(folder)
        results = self._collection.get(include=["documents", "metadatas"])
        self.texts = results.get("documents") or []
        self.meta = results.get("metadatas") or []

    def search(self, query_emb, k: int = 5):
        if self._collection is None:
            return []

        count = self._collection.count()
        if count == 0:
            return []

        embedding = query_emb if isinstance(query_emb, list) else query_emb.tolist()

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )

        out = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for text, meta, score in zip(documents, metadatas, distances):
            out.append({
                "text": text,
                "meta": meta,
                "score": score,
            })

        return out
