"""
Advanced vector store with FAISS backend, persistence, metadata filtering, and hybrid search.
"""

import faiss
import numpy as np
import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from utils import ensure_directory, create_cache_key

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Advanced vector store with FAISS backend, persistence, and metadata filtering.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vector store with configuration.
        
        :param config: Configuration dictionary containing vector store settings
        """
        self.config = config
        self.index_type = config.get('index_type', 'faiss')
        self.faiss_index_type = config.get('faiss_index_type', 'IndexFlatL2')
        self.persist_index = config.get('persist_index', True)
        self.index_path = config.get('index_path', 'cache/vector_index')
        self.search_top_k = config.get('search_top_k', 10)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.enable_metadata_filtering = config.get('enable_metadata_filtering', True)
        
        # Ensure index directory exists
        if self.persist_index:
            ensure_directory(os.path.dirname(self.index_path))
        
        # Initialize storage
        self.index = None
        self._docs = []
        self._metadata_index = {}
        self._dimension = None
        
        logger.info(f"VectorStore initialized with index_type: {self.index_type}")
        logger.info(f"Persistence enabled: {self.persist_index}, Index path: {self.index_path}")

    def build_index(self, docs: List[Dict]):
        """
        Build FAISS index from documents.
        
        :param docs: List of document dictionaries with 'vector' key
        """
        if not docs:
            logger.warning("No documents provided for indexing")
            return
        
        logger.info(f"Building vector index for {len(docs)} documents...")
        
        self._docs = docs
        self._dimension = len(docs[0]["vector"])
        
        # Create FAISS index based on configuration
        if self.faiss_index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self._dimension)
        elif self.faiss_index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self._dimension)
        elif self.faiss_index_type == "IndexIVFFlat":
            # Use IVF index for larger datasets
            quantizer = faiss.IndexFlatL2(self._dimension)
            nlist = min(100, len(docs) // 10)  # Number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, self._dimension, nlist)
        elif self.faiss_index_type == "IndexHNSWFlat":
            # Use HNSW index for approximate search
            self.index = faiss.IndexHNSWFlat(self._dimension, 32)
        else:
            logger.warning(f"Unknown index type {self.faiss_index_type}, using IndexFlatL2")
            self.index = faiss.IndexFlatL2(self._dimension)
        
        # Prepare vectors
        vectors = np.array([d["vector"] for d in docs]).astype("float32")
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(vectors)
        
        # Add vectors to index
        self.index.add(vectors)
        
        # Build metadata index for filtering
        if self.enable_metadata_filtering:
            self._build_metadata_index(docs)
        
        # Persist index if enabled
        if self.persist_index:
            self._save_index()
        
        logger.info(f"Index built successfully. Dimension: {self._dimension}, Index type: {self.faiss_index_type}")

    def _build_metadata_index(self, docs: List[Dict]):
        """Build metadata index for filtering."""
        self._metadata_index = {
            'sources': set(),
            'file_types': set(),
            'keywords': set(),
            'chunk_lengths': []
        }
        
        for doc in docs:
            meta = doc.get('meta', {})
            
            # Index sources
            source = meta.get('source', 'unknown')
            self._metadata_index['sources'].add(source)
            
            # Index file types
            file_metadata = meta.get('file_metadata', {})
            file_type = file_metadata.get('file_extension', 'unknown')
            self._metadata_index['file_types'].add(file_type)
            
            # Index keywords
            keywords = meta.get('keywords', [])
            for keyword in keywords:
                self._metadata_index['keywords'].add(keyword.lower())
            
            # Index chunk lengths
            chunk_length = meta.get('chunk_length', 0)
            self._metadata_index['chunk_lengths'].append(chunk_length)
        
        # Convert sets to lists for serialization
        self._metadata_index['sources'] = list(self._metadata_index['sources'])
        self._metadata_index['file_types'] = list(self._metadata_index['file_types'])
        self._metadata_index['keywords'] = list(self._metadata_index['keywords'])

    def search(self, query_vector: np.ndarray, top_k: int = None, 
               filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Search for similar vectors with optional metadata filtering.
        
        :param query_vector: Query vector
        :param top_k: Number of results to return
        :param filters: Optional metadata filters
        :return: List of similar documents with scores
        """
        if self.index is None:
            logger.error("Index not built. Call build_index() first.")
            return []
        
        if top_k is None:
            top_k = self.search_top_k
        
        # Ensure query vector is the right shape
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Search in FAISS index
        try:
            D, I = self.index.search(query_vector.astype("float32"), min(top_k * 2, len(self._docs)))
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            return []
        
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self._docs):
                continue
            
            doc = self._docs[idx].copy()
            
            # Convert distance to similarity score (for L2 distance)
            if self.faiss_index_type in ["IndexFlatL2", "IndexIVFFlat"]:
                # Convert L2 distance to similarity (lower distance = higher similarity)
                similarity_score = 1.0 / (1.0 + float(score))
            else:
                # For IP and HNSW, score is already similarity
                similarity_score = float(score)
            
            # Apply similarity threshold
            if similarity_score < self.similarity_threshold:
                continue
            
            doc["score"] = similarity_score
            
            # Apply metadata filters if provided
            if filters and not self._apply_filters(doc, filters):
                continue
            
            results.append(doc)
            
            # Stop if we have enough results
            if len(results) >= top_k:
                break
        
        # Sort by score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        logger.debug(f"Search returned {len(results)} results")
        return results

    def _apply_filters(self, doc: Dict, filters: Dict[str, Any]) -> bool:
        """Apply metadata filters to a document."""
        meta = doc.get('meta', {})
        
        # Filter by source
        if 'sources' in filters:
            source = meta.get('source', '')
            if source not in filters['sources']:
                return False
        
        # Filter by file type
        if 'file_types' in filters:
            file_metadata = meta.get('file_metadata', {})
            file_type = file_metadata.get('file_extension', '')
            if file_type not in filters['file_types']:
                return False
        
        # Filter by keywords
        if 'keywords' in filters:
            doc_keywords = [k.lower() for k in meta.get('keywords', [])]
            filter_keywords = [k.lower() for k in filters['keywords']]
            if not any(k in doc_keywords for k in filter_keywords):
                return False
        
        # Filter by chunk length range
        if 'chunk_length_range' in filters:
            chunk_length = meta.get('chunk_length', 0)
            min_length, max_length = filters['chunk_length_range']
            if not (min_length <= chunk_length <= max_length):
                return False
        
        return True

    def hybrid_search(self, query_vector: np.ndarray, query_text: str, 
                     top_k: int = None, alpha: float = 0.7) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        :param query_vector: Semantic query vector
        :param query_text: Text query for keyword matching
        :param top_k: Number of results to return
        :param alpha: Weight for semantic search (1-alpha for keyword search)
        :return: List of documents with combined scores
        """
        if top_k is None:
            top_k = self.search_top_k
        
        # Semantic search
        semantic_results = self.search(query_vector, top_k * 2)
        
        # Keyword search
        keyword_results = self._keyword_search(query_text, top_k * 2)
        
        # Combine results
        combined_scores = {}
        
        # Add semantic scores
        for doc in semantic_results:
            doc_id = doc['id']
            combined_scores[doc_id] = {
                'doc': doc,
                'semantic_score': doc['score'],
                'keyword_score': 0.0
            }
        
        # Add keyword scores
        for doc in keyword_results:
            doc_id = doc['id']
            if doc_id in combined_scores:
                combined_scores[doc_id]['keyword_score'] = doc['score']
            else:
                combined_scores[doc_id] = {
                    'doc': doc,
                    'semantic_score': 0.0,
                    'keyword_score': doc['score']
                }
        
        # Calculate combined scores
        final_results = []
        for doc_id, scores in combined_scores.items():
            combined_score = (alpha * scores['semantic_score'] + 
                            (1 - alpha) * scores['keyword_score'])
            doc = scores['doc'].copy()
            doc['score'] = combined_score
            doc['semantic_score'] = scores['semantic_score']
            doc['keyword_score'] = scores['keyword_score']
            final_results.append(doc)
        
        # Sort by combined score
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        return final_results[:top_k]

    def _keyword_search(self, query_text: str, top_k: int) -> List[Dict]:
        """Perform keyword-based search."""
        query_words = set(query_text.lower().split())
        results = []
        
        for doc in self._docs:
            doc_keywords = set([k.lower() for k in doc['meta'].get('keywords', [])])
            doc_text_words = set(doc['text'].lower().split())
            
            # Calculate keyword overlap
            keyword_overlap = len(query_words.intersection(doc_keywords))
            text_overlap = len(query_words.intersection(doc_text_words))
            
            # Combined keyword score
            score = (keyword_overlap * 2 + text_overlap) / (len(query_words) + 1)
            
            if score > 0:
                result_doc = doc.copy()
                result_doc['score'] = score
                results.append(result_doc)
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        if not self.persist_index:
            return
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{self.index_path}.faiss")
            
            # Save documents and metadata
            with open(f"{self.index_path}.pkl", 'wb') as f:
                pickle.dump({
                    'docs': self._docs,
                    'metadata_index': self._metadata_index,
                    'dimension': self._dimension,
                    'config': self.config
                }, f)
            
            logger.info(f"Index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")

    def _load_index(self) -> bool:
        """Load FAISS index and metadata from disk."""
        if not self.persist_index:
            return False
        
        try:
            # Check if index files exist
            if not (os.path.exists(f"{self.index_path}.faiss") and 
                    os.path.exists(f"{self.index_path}.pkl")):
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(f"{self.index_path}.faiss")
            
            # Load documents and metadata
            with open(f"{self.index_path}.pkl", 'rb') as f:
                data = pickle.load(f)
                self._docs = data['docs']
                self._metadata_index = data['metadata_index']
                self._dimension = data['dimension']
            
            logger.info(f"Index loaded from {self.index_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_documents': len(self._docs),
            'dimension': self._dimension,
            'index_type': self.faiss_index_type,
            'persist_enabled': self.persist_index,
            'metadata_filtering': self.enable_metadata_filtering,
            'available_sources': len(self._metadata_index.get('sources', [])),
            'available_file_types': len(self._metadata_index.get('file_types', [])),
            'total_keywords': len(self._metadata_index.get('keywords', []))
        }