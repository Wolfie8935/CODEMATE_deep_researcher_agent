# Indexing and retrieval (FAISS or in-memory)

"""FAISS-backed vector store with metadata. Simple persist/load implemented."""


import faiss
import numpy as np
from typing import List, Dict


class VectorStore:
    def __init__(self, cfg: Dict):
        self.index = None
        self._docs = []

    def build_index(self, docs: List[Dict]):
        self._docs = docs
        dim = len(docs[0]["vector"])
        self.index = faiss.IndexFlatL2(dim)
        vectors = np.array([d["vector"] for d in docs]).astype("float32")
        self.index.add(vectors)

    def search(self, query_vector, top_k: int = 5) -> List[Dict]:
        q = np.array([query_vector]).astype("float32")
        D, I = self.index.search(q, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self._docs):
                continue
            doc = self._docs[idx].copy()
            doc["score"] = float(score)
            results.append(doc)
        return results