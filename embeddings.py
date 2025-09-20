from typing import List, Dict, Any, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
import os
import pickle
import json
from pathlib import Path
from utils import ensure_directory, create_cache_key, get_file_hash

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EmbeddingModel:
    """
    Advanced embedding model with caching, multiple model support, and persistence.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize embedding model with configuration.
        
        :param config: Configuration dictionary containing embedding settings
        """
        self.config = config
        self.model_name = config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.batch_size = config.get('batch_size', 32)
        self.max_seq_length = config.get('max_seq_length', 512)
        self.normalize_embeddings = config.get('normalize_embeddings', True)
        self.cache_embeddings = config.get('cache_embeddings', True)
        self.cache_dir = config.get('embedding_cache_dir', 'cache/embeddings')
        
        # Ensure cache directory exists
        if self.cache_embeddings:
            ensure_directory(self.cache_dir)
        
        # Load model
        self._load_model()
        
        # Initialize cache
        self.embedding_cache = {}
        if self.cache_embeddings:
            self._load_cache()
        
        logger.info(f"EmbeddingModel initialized with model: {self.model_name}")
        logger.info(f"Cache enabled: {self.cache_embeddings}, Cache dir: {self.cache_dir}")

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            
            # Set max sequence length if specified
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.max_seq_length
            
            logger.info(f"Model loaded successfully. Max sequence length: {self.max_seq_length}")
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise

    def _load_cache(self):
        """Load embedding cache from disk."""
        cache_file = os.path.join(self.cache_dir, 'embedding_cache.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded embedding cache with {len(self.embedding_cache)} entries")
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {e}")
                self.embedding_cache = {}

    def _save_cache(self):
        """Save embedding cache to disk."""
        if not self.cache_embeddings:
            return
        
        cache_file = os.path.join(self.cache_dir, 'embedding_cache.pkl')
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved embedding cache with {len(self.embedding_cache)} entries")
        except Exception as e:
            logger.warning(f"Could not save embedding cache: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return create_cache_key(self.model_name, text)

    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding if available."""
        if not self.cache_embeddings:
            return None
        
        cache_key = self._get_cache_key(text)
        return self.embedding_cache.get(cache_key)

    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding."""
        if not self.cache_embeddings:
            return
        
        cache_key = self._get_cache_key(text)
        self.embedding_cache[cache_key] = embedding.copy()

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string with caching support.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(384)  # Default dimension for all-MiniLM-L6-v2
        
        # Check cache first
        cached_embedding = self._get_cached_embedding(text)
        if cached_embedding is not None:
            logger.debug("Using cached embedding for query")
            return cached_embedding
        
        try:
            # Generate embedding
            vec = self.model.encode(
                text, 
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False
            )
            
            # Cache the embedding
            self._cache_embedding(text, vec)
            
            return vec
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string (alias of embed_query)
        """
        return self.embed_query(text)

    def embed_documents(self, docs: List[Dict]) -> List[Dict]:
        """
        Embed a list of documents with batch processing and caching.
        
        Args:
            docs: List of document dictionaries with 'text' key
            
        Returns:
            Updated list of documents with 'vector' key added
        """
        if not docs:
            logger.warning("No documents provided for embedding")
            return docs
        
        logger.info(f"Embedding {len(docs)} documents...")
        
        # Separate cached and uncached texts
        texts = [d["text"] for d in docs]
        cached_vectors = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            cached_vec = self._get_cached_embedding(text)
            if cached_vec is not None:
                cached_vectors.append((i, cached_vec))
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        logger.info(f"Found {len(cached_vectors)} cached embeddings, generating {len(uncached_texts)} new ones")
        
        # Generate embeddings for uncached texts
        new_vectors = []
        if uncached_texts:
            try:
                new_vectors = self.model.encode(
                    uncached_texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=True
                )
                
                # Cache new embeddings
                for text, vec in zip(uncached_texts, new_vectors):
                    self._cache_embedding(text, vec)
                    
            except Exception as e:
                logger.error(f"Error embedding documents: {e}")
                raise
        
        # Combine cached and new vectors
        all_vectors = [None] * len(docs)
        
        # Add cached vectors
        for i, vec in cached_vectors:
            all_vectors[i] = vec
        
        # Add new vectors
        for i, vec in zip(uncached_indices, new_vectors):
            all_vectors[i] = vec
        
        # Add vectors to documents
        for doc, vec in zip(docs, all_vectors):
            if vec is not None:
                doc["vector"] = vec
            else:
                logger.warning(f"No vector generated for document: {doc.get('id', 'unknown')}")
                doc["vector"] = np.zeros(384)  # Default dimension
        
        # Save cache periodically
        if self.cache_embeddings and len(uncached_texts) > 0:
            self._save_cache()
        
        logger.info("Document embedding complete")
        return docs

    def embed(self, docs: List[Dict]) -> List[Dict]:
        """
        Alias for embed_documents for backward compatibility.
        """
        return self.embed_documents(docs)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        try:
            # Create a dummy embedding to get dimension
            dummy_embedding = self.embed_query("test")
            return len(dummy_embedding)
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            return 384  # Default for all-MiniLM-L6-v2

    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        if self.cache_embeddings:
            cache_file = os.path.join(self.cache_dir, 'embedding_cache.pkl')
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    logger.info("Embedding cache cleared")
                except Exception as e:
                    logger.warning(f"Could not remove cache file: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache."""
        return {
            'cache_enabled': self.cache_embeddings,
            'cache_size': len(self.embedding_cache),
            'cache_dir': self.cache_dir,
            'model_name': self.model_name
        }
