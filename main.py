# Entry point (CLI or web UI to run queries)

import argparse
import logging
from utils import load_config, validate_query
from ingestion import DocumentLoader
from embeddings import EmbeddingModel
from vector_store import VectorStore
from reasoning import Reasoner
from synthesizer import Synthesizer
from exporter import Exporter

logger = logging.getLogger(__name__)

def run(query: str, config_path: str = "config.yml"):
    """
    Run the deep researcher agent with the given query.
    
    :param query: Research query string
    :param config_path: Path to configuration file
    """
    # Load configuration
    cfg = load_config(config_path)
    
    # Validate query
    is_valid, error_msg = validate_query(query)
    if not is_valid:
        print(f"Error: {error_msg}")
        return
    
    print(f"🔍 Deep Researcher Agent")
    print(f"Query: {query}")
    print("=" * 50)
    
    try:
        # Step 1: Document Ingestion
        print("📄 Ingesting documents...")
        loader = DocumentLoader(cfg.get('ingestion', {}))
        docs = loader.load()
        
        if not docs:
            print("❌ No documents found. Please add documents to the data directory.")
            return
        
        stats = loader.get_document_stats(docs)
        print(f"✅ Loaded {stats['total_documents']} documents, {stats['total_chunks']} chunks")
        print(f"   File types: {list(stats['file_types'].keys())}")
        print(f"   Average chunk length: {stats['average_chunk_length']:.0f} characters")

        # Step 2: Embedding Generation
        print("\n🧠 Generating embeddings...")
        embedder = EmbeddingModel(cfg.get('embeddings', {}))
        docs = embedder.embed_documents(docs)
        
        cache_stats = embedder.get_cache_stats()
        print(f"✅ Embeddings generated. Cache size: {cache_stats['cache_size']}")

        # Step 3: Vector Index Building
        print("\n🔍 Building vector index...")
        store = VectorStore(cfg.get('vector_store', {}))
        store.build_index(docs)
        
        store_stats = store.get_stats()
        print(f"✅ Index built. Documents: {store_stats['total_documents']}, Dimension: {store_stats['dimension']}")

        # Step 4: Query Processing and Reasoning
        print("\n🤔 Processing query and reasoning...")
        reasoner = Reasoner(store, embedder, cfg.get('reasoning', {}))
        subtasks = reasoner.decompose(query)
        
        print(f"📋 Decomposed into {len(subtasks)} subtasks:")
        for i, subtask in enumerate(subtasks, 1):
            print(f"   {i}. {subtask}")
        
        results = reasoner.solve(subtasks)
        
        # Display results summary
        total_evidence = sum(len(r.get('evidence', [])) for r in results)
        avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results) if results else 0
        print(f"✅ Reasoning complete. Total evidence: {total_evidence}, Avg confidence: {avg_confidence:.2f}")

        # Step 5: Synthesis
        print("\n📝 Synthesizing final answer...")
        synthesizer = Synthesizer(cfg.get('synthesizer', {}))
        final_answer = synthesizer.combine(results, query)

        # Step 6: Display Results
        print('\n' + '=' * 50)
        print('🎯 FINAL RESEARCH RESULTS')
        print('=' * 50)
        print(final_answer)

        # Step 7: Export
        print('\n💾 Exporting results...')
        exporter = Exporter(cfg.get('export', {}))
        export_path = exporter.export(query, final_answer, results)
        print(f'✅ Export complete: {export_path}')
        
        # Display summary statistics
        print('\n📊 Research Summary:')
        print(f"   Query: {query}")
        print(f"   Subtasks processed: {len(subtasks)}")
        print(f"   Total evidence pieces: {total_evidence}")
        print(f"   Average confidence: {avg_confidence:.2f}")
        print(f"   Export file: {export_path}")

    except Exception as e:
        logger.error(f"Error during research process: {e}")
        print(f"❌ Error: {e}")
        raise

def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Deep Researcher Agent - Search, analyze, and synthesize information from local documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "What are the main findings about climate change?"
  python main.py "Compare different machine learning approaches" --config custom_config.yml
        """
    )
    
    parser.add_argument('query', type=str, help='Research question to investigate')
    parser.add_argument('--config', type=str, default='config.yml', 
                       help='Path to configuration file (default: config.yml)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the research process
    run(args.query, args.config)

if __name__ == '__main__':
    main()
