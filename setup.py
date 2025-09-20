#!/usr/bin/env python3
"""
Setup script for the Deep Researcher Agent.

This script helps users set up the system by creating necessary directories,
downloading sample data, and validating the installation.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "data",
        "exports", 
        "cache",
        "cache/embeddings",
        "cache/vector_index",
        "logs",
        "templates"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "sentence-transformers",
        "faiss-cpu", 
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "PyPDF2",
        "python-docx",
        "reportlab",
        "markdown",
        "streamlit",
        "tqdm",
        "pyyaml",
        "beautifulsoup4",
        "lxml",
        "html5lib"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} (missing)")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def download_sample_data():
    """Download sample research paper for testing."""
    print("📄 Downloading sample data...")
    
    sample_url = "https://arxiv.org/pdf/2103.00020.pdf"  # Attention Is All You Need
    sample_file = Path("data") / "sample_paper.pdf"
    
    if sample_file.exists():
        print(f"  ✅ Sample file already exists: {sample_file}")
        return
    
    try:
        print(f"  📥 Downloading from: {sample_url}")
        urllib.request.urlretrieve(sample_url, sample_file)
        print(f"  ✅ Downloaded: {sample_file}")
    except Exception as e:
        print(f"  ⚠️  Could not download sample file: {e}")
        print("  💡 You can manually add your own documents to the 'data/' directory")


def create_sample_config():
    """Create a sample configuration file if it doesn't exist."""
    print("⚙️  Checking configuration...")
    
    config_file = Path("config.yml")
    if config_file.exists():
        print(f"  ✅ Configuration file exists: {config_file}")
        return
    
    print("  📝 Creating sample configuration file...")
    
    sample_config = """# Deep Researcher Agent Configuration

# Document Ingestion Settings
ingestion:
  data_dir: "data"
  chunk_size: 1000
  chunk_overlap: 200
  supported_formats: [".pdf", ".docx", ".txt", ".md", ".html"]
  max_file_size_mb: 50
  enable_metadata_extraction: true
  preserve_formatting: true

# Embedding Model Configuration
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32
  max_seq_length: 512
  normalize_embeddings: true
  cache_embeddings: true
  embedding_cache_dir: "cache/embeddings"

# Vector Store Configuration
vector_store:
  index_type: "faiss"
  faiss_index_type: "IndexFlatL2"
  persist_index: true
  index_path: "cache/vector_index"
  search_top_k: 10
  similarity_threshold: 0.7
  enable_metadata_filtering: true

# Reasoning and Query Processing
reasoning:
  decomposition_strategy: "intelligent"
  max_subtasks: 5
  subtask_overlap_threshold: 0.3
  top_k: 5
  rerank_results: true
  evidence_diversity_threshold: 0.8
  summarizer_model: "facebook/bart-large-cnn"
  summarizer_max_input_chars: 3000
  summarizer_min_length: 30
  summarizer_max_length: 200
  enable_abstractive_summarization: true
  fallback_to_extractive: true
  enable_llm_reasoning: false

# Synthesis Configuration
synthesizer:
  max_chars: 2000
  max_evidence: 5
  enable_citation_tracking: true
  citation_format: "apa"
  confidence_scoring: true
  source_verification: true
  enable_contradiction_detection: true

# Export Configuration
export:
  output_dir: "exports"
  formats: ["markdown", "pdf", "json", "html"]
  include_metadata: true
  include_citations: true
  include_reasoning_trace: true

# Web UI Configuration
web_ui:
  host: "localhost"
  port: 8501
  title: "Deep Researcher Agent"
  theme: "light"
  enable_file_upload: true
  max_upload_size_mb: 100
  enable_real_time_search: true
  show_confidence_scores: true
  enable_follow_up_questions: true

# AI Assistant Configuration
ai_assistant:
  enable_explanations: true
  explanation_depth: "detailed"
  show_reasoning_steps: true
  enable_insights: true
  suggest_follow_up_questions: true
  max_insights: 3

# Performance and Caching
performance:
  enable_caching: true
  cache_dir: "cache"
  max_cache_size_mb: 1000
  cache_ttl_hours: 24
  batch_size: 10
  max_concurrent_requests: 5
  max_memory_usage_mb: 2048
  enable_memory_monitoring: true

# Logging Configuration
logging:
  level: "INFO"
  log_file: "logs/researcher.log"
  max_log_size_mb: 10
  backup_count: 5
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Security and Privacy
security:
  enable_input_sanitization: true
  max_query_length: 1000
  block_malicious_patterns: true
  enable_rate_limiting: true
  max_requests_per_minute: 60

# Advanced Features
advanced:
  enable_semantic_search: true
  enable_hybrid_search: true
  enable_query_expansion: true
  enable_result_reranking: true
  enable_multi_modal_support: false
  enable_research_workflow: true
  auto_save_progress: true
  enable_collaboration: false
"""
    
    config_file.write_text(sample_config)
    print(f"  ✅ Created: {config_file}")


def run_quick_test():
    """Run a quick test to verify the system works."""
    print("🧪 Running quick test...")
    
    try:
        # Import main modules
        from utils import load_config
        from ingestion import DocumentLoader
        from embeddings import EmbeddingModel
        
        # Load configuration
        config = load_config("config.yml")
        print("  ✅ Configuration loaded")
        
        # Test document loader
        loader = DocumentLoader(config['ingestion'])
        docs = loader.load()
        print(f"  ✅ Document loader working ({len(docs)} documents found)")
        
        # Test embedding model (just initialization)
        embedder = EmbeddingModel(config['embeddings'])
        print("  ✅ Embedding model initialized")
        
        print("  🎉 Quick test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Quick test failed: {e}")
        return False


def print_next_steps():
    """Print instructions for next steps."""
    print("\n🚀 Setup Complete! Next Steps:")
    print("=" * 50)
    
    print("\n1. 📄 Add your documents:")
    print("   - Place PDF, DOCX, TXT, MD, or HTML files in the 'data/' directory")
    print("   - The system will automatically process them")
    
    print("\n2. 🔍 Run a research query:")
    print("   python main.py \"What are the main findings in your documents?\"")
    
    print("\n3. 🌐 Launch the web interface:")
    print("   streamlit run app.py")
    print("   Then open http://localhost:8501 in your browser")
    
    print("\n4. 🧪 Run the example script:")
    print("   python example_usage.py")
    
    print("\n5. 🧪 Run the test suite:")
    print("   python test_system.py")
    
    print("\n6. ⚙️  Customize configuration:")
    print("   - Edit 'config.yml' to adjust settings")
    print("   - See README.md for detailed configuration options")
    
    print("\n💡 Tips:")
    print("   - Check the 'exports/' directory for generated reports")
    print("   - Use '--verbose' flag for detailed output")
    print("   - Check 'logs/researcher.log' for system logs")


def main():
    """Main setup function."""
    print("🔧 Deep Researcher Agent - Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create directories
    create_directories()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Create sample configuration
    create_sample_config()
    
    # Download sample data
    download_sample_data()
    
    # Run quick test
    if deps_ok:
        test_ok = run_quick_test()
    else:
        test_ok = False
    
    # Print results
    if deps_ok and test_ok:
        print("\n🎉 Setup completed successfully!")
        print_next_steps()
    else:
        print("\n⚠️  Setup completed with issues:")
        if not deps_ok:
            print("   - Missing dependencies. Run: pip install -r requirements.txt")
        if not test_ok:
            print("   - System test failed. Check the error messages above")
        print("\n💡 After fixing issues, run: python test_system.py")


if __name__ == "__main__":
    main()
