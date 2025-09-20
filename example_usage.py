#!/usr/bin/env python3
"""
Example usage script for the Deep Researcher Agent.

This script demonstrates various ways to use the system programmatically
and showcases different configuration options and features.
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, validate_query
from ingestion import DocumentLoader
from embeddings import EmbeddingModel
from vector_store import VectorStore
from reasoning import Reasoner
from synthesizer import Synthesizer
from exporter import Exporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Example of basic usage with default configuration."""
    print("🔍 Example 1: Basic Usage")
    print("=" * 50)
    
    # Load configuration
    config = load_config("config.yml")
    
    # Initialize components
    loader = DocumentLoader(config['ingestion'])
    embedder = EmbeddingModel(config['embeddings'])
    store = VectorStore(config['vector_store'])
    reasoner = Reasoner(store, embedder, config['reasoning'])
    synthesizer = Synthesizer(config['synthesizer'])
    exporter = Exporter(config['export'])
    
    # Process documents
    print("📄 Loading documents...")
    docs = loader.load()
    
    if not docs:
        print("❌ No documents found. Please add documents to the data/ directory.")
        return
    
    print(f"✅ Loaded {len(docs)} document chunks")
    
    # Generate embeddings
    print("🧠 Generating embeddings...")
    docs = embedder.embed_documents(docs)
    
    # Build vector index
    print("🔍 Building vector index...")
    store.build_index(docs)
    
    # Research query
    query = "What are the main findings and conclusions?"
    print(f"🤔 Research query: {query}")
    
    # Decompose and solve
    subtasks = reasoner.decompose(query)
    print(f"📋 Decomposed into {len(subtasks)} subtasks")
    
    results = reasoner.solve(subtasks)
    
    # Synthesize final answer
    final_answer = synthesizer.combine(results, query)
    
    # Export results
    export_path = exporter.export(query, final_answer, results)
    
    print(f"✅ Research complete! Results exported to: {export_path}")
    print("\n" + "=" * 50)


def example_advanced_configuration():
    """Example with advanced configuration options."""
    print("🔍 Example 2: Advanced Configuration")
    print("=" * 50)
    
    # Custom configuration for advanced features
    custom_config = {
        'ingestion': {
            'data_dir': 'data',
            'chunk_size': 800,  # Smaller chunks for more precise retrieval
            'chunk_overlap': 150,
            'supported_formats': ['.pdf', '.docx', '.txt', '.md'],
            'enable_metadata_extraction': True
        },
        'embeddings': {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'batch_size': 16,  # Smaller batch size for memory efficiency
            'cache_embeddings': True,
            'normalize_embeddings': True
        },
        'vector_store': {
            'index_type': 'faiss',
            'faiss_index_type': 'IndexFlatL2',
            'persist_index': True,
            'search_top_k': 8,
            'similarity_threshold': 0.6,
            'enable_metadata_filtering': True
        },
        'reasoning': {
            'decomposition_strategy': 'intelligent',
            'max_subtasks': 4,
            'top_k': 5,
            'rerank_results': True,
            'evidence_diversity_threshold': 0.7,
            'enable_abstractive_summarization': True
        },
        'synthesizer': {
            'max_chars': 1500,
            'max_evidence': 4,
            'enable_citation_tracking': True,
            'citation_format': 'apa',
            'confidence_scoring': True,
            'source_verification': True,
            'enable_contradiction_detection': True
        },
        'export': {
            'output_dir': 'exports',
            'formats': ['markdown', 'json', 'html'],
            'include_metadata': True,
            'include_citations': True
        }
    }
    
    # Initialize with custom configuration
    loader = DocumentLoader(custom_config['ingestion'])
    embedder = EmbeddingModel(custom_config['embeddings'])
    store = VectorStore(custom_config['vector_store'])
    reasoner = Reasoner(store, embedder, custom_config['reasoning'])
    synthesizer = Synthesizer(custom_config['synthesizer'])
    exporter = Exporter(custom_config['export'])
    
    # Process documents
    docs = loader.load()
    if not docs:
        print("❌ No documents found.")
        return
    
    docs = embedder.embed_documents(docs)
    store.build_index(docs)
    
    # Complex research query
    query = "Compare and contrast different approaches, identify key trends, and analyze potential implications"
    print(f"🤔 Complex query: {query}")
    
    # Process with advanced features
    subtasks = reasoner.decompose(query)
    results = reasoner.solve(subtasks)
    final_answer = synthesizer.combine(results, query)
    
    # Export in multiple formats
    export_path = exporter.export(query, final_answer, results)
    
    print(f"✅ Advanced research complete! Results exported to: {export_path}")
    print("\n" + "=" * 50)


def example_hybrid_search():
    """Example demonstrating hybrid search capabilities."""
    print("🔍 Example 3: Hybrid Search")
    print("=" * 50)
    
    config = load_config("config.yml")
    
    # Initialize components
    loader = DocumentLoader(config['ingestion'])
    embedder = EmbeddingModel(config['embeddings'])
    store = VectorStore(config['vector_store'])
    
    # Process documents
    docs = loader.load()
    if not docs:
        print("❌ No documents found.")
        return
    
    docs = embedder.embed_documents(docs)
    store.build_index(docs)
    
    # Demonstrate different search types
    query_text = "machine learning algorithms"
    query_vector = embedder.embed_query(query_text)
    
    print(f"🔍 Query: {query_text}")
    
    # Semantic search
    print("\n📊 Semantic Search Results:")
    semantic_results = store.search(query_vector, top_k=3)
    for i, result in enumerate(semantic_results, 1):
        source = result['meta']['source']
        score = result['score']
        text_preview = result['text'][:100] + "..."
        print(f"  {i}. {source} (score: {score:.3f})")
        print(f"     {text_preview}")
    
    # Hybrid search
    print("\n🔀 Hybrid Search Results:")
    hybrid_results = store.hybrid_search(query_vector, query_text, top_k=3, alpha=0.7)
    for i, result in enumerate(hybrid_results, 1):
        source = result['meta']['source']
        combined_score = result['score']
        semantic_score = result.get('semantic_score', 0)
        keyword_score = result.get('keyword_score', 0)
        text_preview = result['text'][:100] + "..."
        print(f"  {i}. {source} (combined: {combined_score:.3f}, semantic: {semantic_score:.3f}, keyword: {keyword_score:.3f})")
        print(f"     {text_preview}")
    
    print("\n" + "=" * 50)


def example_confidence_analysis():
    """Example demonstrating confidence scoring and analysis."""
    print("🔍 Example 4: Confidence Analysis")
    print("=" * 50)
    
    config = load_config("config.yml")
    
    # Initialize components
    loader = DocumentLoader(config['ingestion'])
    embedder = EmbeddingModel(config['embeddings'])
    store = VectorStore(config['vector_store'])
    reasoner = Reasoner(store, embedder, config['reasoning'])
    synthesizer = Synthesizer(config['synthesizer'])
    
    # Process documents
    docs = loader.load()
    if not docs:
        print("❌ No documents found.")
        return
    
    docs = embedder.embed_documents(docs)
    store.build_index(docs)
    
    # Research query
    query = "What are the key findings and their reliability?"
    print(f"🤔 Query: {query}")
    
    # Process with confidence analysis
    subtasks = reasoner.decompose(query)
    results = reasoner.solve(subtasks)
    
    # Analyze confidence scores
    print("\n📊 Confidence Analysis:")
    total_confidence = 0
    for i, result in enumerate(results, 1):
        confidence = result.get('confidence', 0)
        total_confidence += confidence
        evidence_count = len(result.get('evidence', []))
        subtask = result.get('subtask', 'Unknown')
        
        confidence_level = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        print(f"  {i}. {subtask[:50]}...")
        print(f"     Confidence: {confidence:.2f} ({confidence_level})")
        print(f"     Evidence pieces: {evidence_count}")
        print()
    
    avg_confidence = total_confidence / len(results) if results else 0
    print(f"📈 Overall Confidence: {avg_confidence:.2f}")
    
    # Generate final answer with confidence assessment
    final_answer = synthesizer.combine(results, query)
    
    print(f"\n📝 Final Answer Preview:")
    print(final_answer[:300] + "..." if len(final_answer) > 300 else final_answer)
    
    print("\n" + "=" * 50)


def example_export_formats():
    """Example demonstrating different export formats."""
    print("🔍 Example 5: Export Formats")
    print("=" * 50)
    
    config = load_config("config.yml")
    
    # Initialize components
    loader = DocumentLoader(config['ingestion'])
    embedder = EmbeddingModel(config['embeddings'])
    store = VectorStore(config['vector_store'])
    reasoner = Reasoner(store, embedder, config['reasoning'])
    
    # Process documents
    docs = loader.load()
    if not docs:
        print("❌ No documents found.")
        return
    
    docs = embedder.embed_documents(docs)
    store.build_index(docs)
    
    # Research query
    query = "Summarize the main points and provide detailed analysis"
    print(f"🤔 Query: {query}")
    
    # Process
    subtasks = reasoner.decompose(query)
    results = reasoner.solve(subtasks)
    
    # Test different export configurations
    export_configs = [
        {
            'name': 'Markdown Only',
            'config': {'formats': ['markdown'], 'include_citations': True}
        },
        {
            'name': 'Multiple Formats',
            'config': {'formats': ['markdown', 'json', 'html'], 'include_citations': True}
        },
        {
            'name': 'PDF Export',
            'config': {'formats': ['markdown', 'pdf'], 'include_citations': True}
        }
    ]
    
    for export_config in export_configs:
        print(f"\n📄 Testing {export_config['name']}:")
        
        # Create exporter with specific configuration
        exporter_config = {**config.get('export', {}), **export_config['config']}
        exporter = Exporter(exporter_config)
        
        try:
            export_path = exporter.export(query, "Sample research results", results)
            print(f"  ✅ Exported to: {export_path}")
        except Exception as e:
            print(f"  ❌ Export failed: {e}")
    
    print("\n" + "=" * 50)


def main():
    """Run all examples."""
    print("🚀 Deep Researcher Agent - Usage Examples")
    print("=" * 60)
    
    # Check if data directory exists and has files
    data_dir = Path("data")
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("⚠️  Warning: No documents found in 'data/' directory.")
        print("   Please add some documents (PDF, DOCX, TXT, MD, HTML) to the data/ directory.")
        print("   You can use the sample PDF that's already there.")
        print()
    
    try:
        # Run examples
        example_basic_usage()
        example_advanced_configuration()
        example_hybrid_search()
        example_confidence_analysis()
        example_export_formats()
        
        print("🎉 All examples completed successfully!")
        print("\n💡 Tips:")
        print("   - Check the 'exports/' directory for generated reports")
        print("   - Modify 'config.yml' to customize behavior")
        print("   - Use 'python main.py --help' for CLI options")
        print("   - Run 'streamlit run app.py' for the web interface")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   - Check that documents are in the 'data/' directory")
        print("   - Verify the configuration file 'config.yml' exists")
        print("   - Check the logs for detailed error information")


if __name__ == "__main__":
    main()
