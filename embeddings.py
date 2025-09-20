from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EmbeddingModel:
    """
    Wraps a local SentenceTransformer model to embed text.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model {model_name} on CPU...")
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string.
        """
        vec = self.model.encode(text, convert_to_numpy=True)
        return vec

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string (alias of embed_query)
        """
        return self.embed_query(text)

    def embed(self, docs: List[Dict]) -> List[Dict]:
        """
        Embed a list of documents (dicts with 'text' key) in-place.
        Returns the updated list of dicts with a 'vector' key added.
        """
        texts = [d["text"] for d in docs]
        logger.info("Embedding documents...")
        vectors = self.model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
        for doc, vec in zip(docs, vectors):
            doc["vector"] = vec
        return docs
