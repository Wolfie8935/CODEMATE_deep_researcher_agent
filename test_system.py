#!/usr/bin/env python3
"""
Simple test script to validate the Deep Researcher Agent system.

This script performs basic functionality tests to ensure all components
are working correctly.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, validate_query, create_cache_key, calculate_similarity
from ingestion import DocumentLoader
from embeddings import EmbeddingModel
from vector_store import VectorStore
from reasoning import Reasoner
from synthesizer import Synthesizer
from exporter import Exporter


def test_utils():
    """Test utility functions."""
    print("🧪 Testing utility functions...")
    
    # Test query validation
    valid, msg = validate_query("What is machine learning?")
    assert valid, f"Query validation failed: {msg}"
    
    invalid, msg = validate_query("")
    assert not invalid, "Empty query should be invalid"
    
    # Test cache key generation
    key1 = create_cache_key("test", "query")
    key2 = create_cache_key("test", "query")
    assert key1 == key2, "Cache keys should be consistent"
    
    # Test similarity calculation
    sim = calculate_similarity("machine learning", "artificial intelligence")
    assert 0 <= sim <= 1, "Similarity should be between 0 and 1"
    
    print("✅ Utility functions working correctly")


def test_document_loader():
    """Test document loading functionality."""
    print("🧪 Testing document loader...")
    
    # Create temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test text file
        test_file = Path(temp_dir) / "test.txt"
        test_content = "This is a test document about machine learning and artificial intelligence. It contains information about neural networks, deep learning, and natural language processing."
        test_file.write_text(test_content)
        
        # Test document loader
        config = {
            'data_dir': temp_dir,
            'chunk_size': 100,
            'chunk_overlap': 20,
            'supported_formats': ['.txt'],
            'max_file_size_mb': 10
        }
        
        loader = DocumentLoader(config)
        docs = loader.load()
        
        assert len(docs) > 0, "Should load at least one document"
        assert 'text' in docs[0], "Document should have text field"
        assert 'meta' in docs[0], "Document should have meta field"
        assert 'source' in docs[0]['meta'], "Document should have source in meta"
        
        # Test statistics
        stats = loader.get_document_stats(docs)
        assert stats['total_documents'] > 0, "Should have at least one document"
        assert stats['total_chunks'] > 0, "Should have at least one chunk"
    
    print("✅ Document loader working correctly")


def test_embeddings():
    """Test embedding generation."""
    print("🧪 Testing embedding model...")
    
    config = {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'batch_size': 2,
        'cache_embeddings': False,  # Disable cache for testing
        'normalize_embeddings': True
    }
    
    embedder = EmbeddingModel(config)
    
    # Test single query embedding
    query = "What is machine learning?"
    embedding = embedder.embed_query(query)
    assert len(embedding) > 0, "Should generate embedding vector"
    assert len(embedding) == 384, "Should have correct dimension for all-MiniLM-L6-v2"
    
    # Test document embedding
    docs = [
        {'text': 'Machine learning is a subset of artificial intelligence.'},
        {'text': 'Deep learning uses neural networks with multiple layers.'}
    ]
    
    embedded_docs = embedder.embed_documents(docs)
    assert len(embedded_docs) == 2, "Should embed all documents"
    assert 'vector' in embedded_docs[0], "Should add vector field to documents"
    
    # Test cache stats
    stats = embedder.get_cache_stats()
    assert 'model_name' in stats, "Should include model name in stats"
    
    print("✅ Embedding model working correctly")


def test_vector_store():
    """Test vector store functionality."""
    print("🧪 Testing vector store...")
    
    config = {
        'index_type': 'faiss',
        'faiss_index_type': 'IndexFlatL2',
        'persist_index': False,  # Disable persistence for testing
        'search_top_k': 3,
        'similarity_threshold': 0.5
    }
    
    store = VectorStore(config)
    
    # Create test documents with embeddings
    docs = [
        {
            'id': 'doc1',
            'text': 'Machine learning is a subset of artificial intelligence.',
            'vector': [0.1] * 384,  # Dummy embedding
            'meta': {'source': 'test1.txt'}
        },
        {
            'id': 'doc2', 
            'text': 'Deep learning uses neural networks with multiple layers.',
            'vector': [0.2] * 384,  # Dummy embedding
            'meta': {'source': 'test2.txt'}
        },
        {
            'id': 'doc3',
            'text': 'Natural language processing deals with text and speech.',
            'vector': [0.3] * 384,  # Dummy embedding
            'meta': {'source': 'test3.txt'}
        }
    ]
    
    # Build index
    store.build_index(docs)
    
    # Test search
    query_vector = [0.15] * 384  # Dummy query vector
    results = store.search(query_vector, top_k=2)
    
    assert len(results) <= 2, "Should return at most top_k results"
    assert all('score' in r for r in results), "Results should have scores"
    
    # Test hybrid search
    hybrid_results = store.hybrid_search(query_vector, "machine learning", top_k=2)
    assert len(hybrid_results) <= 2, "Hybrid search should return at most top_k results"
    
    # Test stats
    stats = store.get_stats()
    assert stats['total_documents'] == 3, "Should have correct document count"
    
    print("✅ Vector store working correctly")


def test_reasoning():
    """Test reasoning engine."""
    print("🧪 Testing reasoning engine...")
    
    # Create mock components
    class MockEmbedder:
        def embed_query(self, text):
            return [0.1] * 384
    
    class MockStore:
        def search(self, vector, top_k=5):
            return [
                {
                    'text': 'Machine learning is a subset of AI.',
                    'score': 0.8,
                    'meta': {'source': 'test.txt'}
                }
            ]
    
    config = {
        'decomposition_strategy': 'simple',
        'max_subtasks': 3,
        'top_k': 2,
        'enable_abstractive_summarization': False  # Disable for testing
    }
    
    mock_embedder = MockEmbedder()
    mock_store = MockStore()
    
    reasoner = Reasoner(mock_store, mock_embedder, config)
    
    # Test query decomposition
    query = "What is machine learning and how does it work?"
    subtasks = reasoner.decompose(query)
    assert len(subtasks) > 0, "Should decompose query into subtasks"
    assert all(isinstance(s, str) for s in subtasks), "Subtasks should be strings"
    
    # Test solving
    results = reasoner.solve(subtasks)
    assert len(results) == len(subtasks), "Should solve all subtasks"
    assert all('answer' in r for r in results), "Results should have answers"
    assert all('confidence' in r for r in results), "Results should have confidence scores"
    
    print("✅ Reasoning engine working correctly")


def test_synthesizer():
    """Test synthesis functionality."""
    print("🧪 Testing synthesizer...")
    
    config = {
        'max_chars': 1000,
        'max_evidence': 3,
        'enable_citation_tracking': True,
        'confidence_scoring': True,
        'enable_contradiction_detection': True
    }
    
    synthesizer = Synthesizer(config)
    
    # Test data
    results = [
        {
            'subtask': 'What is machine learning?',
            'answer': 'Machine learning is a subset of artificial intelligence.',
            'confidence': 0.8,
            'evidence': [
                {
                    'text': 'ML is a subset of AI.',
                    'score': 0.9,
                    'meta': {'source': 'test1.txt'}
                }
            ]
        },
        {
            'subtask': 'How does it work?',
            'answer': 'It works by learning patterns from data.',
            'confidence': 0.7,
            'evidence': [
                {
                    'text': 'ML learns from data patterns.',
                    'score': 0.8,
                    'meta': {'source': 'test2.txt'}
                }
            ]
        }
    ]
    
    query = "What is machine learning and how does it work?"
    final_answer = synthesizer.combine(results, query)
    
    assert isinstance(final_answer, str), "Should return string"
    assert len(final_answer) > 0, "Should generate non-empty answer"
    assert query in final_answer, "Should include original query"
    
    print("✅ Synthesizer working correctly")


def test_exporter():
    """Test export functionality."""
    print("🧪 Testing exporter...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'output_dir': temp_dir,
            'formats': ['markdown', 'json'],
            'include_metadata': True,
            'include_citations': True
        }
        
        exporter = Exporter(config)
        
        # Test data
        query = "What is machine learning?"
        final_answer = "Machine learning is a subset of artificial intelligence."
        results = [
            {
                'subtask': 'What is machine learning?',
                'answer': 'Machine learning is a subset of AI.',
                'confidence': 0.8,
                'evidence': []
            }
        ]
        
        # Test export
        export_path = exporter.export(query, final_answer, results)
        
        assert export_path is not None, "Should return export path"
        assert os.path.exists(export_path), "Export file should exist"
        
        # Check if markdown file was created
        markdown_files = list(Path(temp_dir).glob("*.md"))
        assert len(markdown_files) > 0, "Should create markdown file"
        
        # Check if JSON file was created
        json_files = list(Path(temp_dir).glob("*.json"))
        assert len(json_files) > 0, "Should create JSON file"
        
        # Test export stats
        stats = exporter.get_export_stats()
        assert 'output_dir' in stats, "Should include output directory in stats"
    
    print("✅ Exporter working correctly")


def test_integration():
    """Test full system integration."""
    print("🧪 Testing system integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test document
        test_file = Path(temp_dir) / "test.txt"
        test_content = """
        Machine learning is a subset of artificial intelligence that focuses on algorithms 
        that can learn from data. Deep learning is a subset of machine learning that uses 
        neural networks with multiple layers. Natural language processing is another area 
        of AI that deals with understanding and generating human language.
        """
        test_file.write_text(test_content)
        
        # Load configuration
        config = load_config("config.yml")
        
        # Override data directory for testing
        config['ingestion']['data_dir'] = temp_dir
        config['export']['output_dir'] = temp_dir
        config['embeddings']['cache_embeddings'] = False
        config['vector_store']['persist_index'] = False
        
        # Initialize components
        loader = DocumentLoader(config['ingestion'])
        embedder = EmbeddingModel(config['embeddings'])
        store = VectorStore(config['vector_store'])
        reasoner = Reasoner(store, embedder, config['reasoning'])
        synthesizer = Synthesizer(config['synthesizer'])
        exporter = Exporter(config['export'])
        
        # Process documents
        docs = loader.load()
        assert len(docs) > 0, "Should load test document"
        
        docs = embedder.embed_documents(docs)
        store.build_index(docs)
        
        # Research query
        query = "What is machine learning?"
        subtasks = reasoner.decompose(query)
        results = reasoner.solve(subtasks)
        final_answer = synthesizer.combine(results, query)
        
        # Export
        export_path = exporter.export(query, final_answer, results)
        
        assert export_path is not None, "Should complete full pipeline"
        assert os.path.exists(export_path), "Should create export file"
    
    print("✅ System integration working correctly")


def main():
    """Run all tests."""
    print("🧪 Deep Researcher Agent - System Tests")
    print("=" * 50)
    
    try:
        test_utils()
        test_document_loader()
        test_embeddings()
        test_vector_store()
        test_reasoning()
        test_synthesizer()
        test_exporter()
        test_integration()
        
        print("\n🎉 All tests passed successfully!")
        print("✅ The Deep Researcher Agent system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
