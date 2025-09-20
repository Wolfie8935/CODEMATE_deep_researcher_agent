# 🔍 Deep Researcher Agent

A comprehensive, Python-based deep researcher agent that can search, analyze, and synthesize information from large-scale data sources without relying on external web search APIs. The system handles local embedding generation, intelligent reasoning, and provides advanced features for research automation.

## Working Link
- Demo video and Explaination : https://youtu.be/9EykW6EdIfk
- Website Link : https://wolfie8935-researcher.streamlit.app/

## ✨ Features

### Core Capabilities
- **Local Document Processing**: Supports PDF, DOCX, TXT, Markdown, and HTML files
- **Advanced Embedding Generation**: Uses sentence-transformers for local embedding generation with caching
- **Intelligent Query Decomposition**: Breaks down complex queries into manageable subtasks
- **Multi-step Reasoning**: Advanced reasoning engine with evidence retrieval and synthesis
- **Vector Search**: FAISS-based vector store with metadata filtering and hybrid search
- **Citation Tracking**: Comprehensive citation management with multiple format support

### Advanced Features
- **Confidence Scoring**: Automatic confidence assessment for all generated answers
- **Contradiction Detection**: Identifies conflicting information across sources
- **Source Verification**: Assesses source credibility and diversity
- **Multiple Export Formats**: PDF, Markdown, JSON, and HTML export capabilities
- **Interactive Web UI**: Streamlit-based interface with real-time search
- **AI Assistant**: Explains reasoning steps and provides insights
- **Performance Optimization**: Caching, batch processing, and memory management

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Wolfie8935/Deep_researcher_agent
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Add your documents**:
Place your research documents (PDF, DOCX, TXT, MD, HTML) in the `data/` directory.

4. **Run a research query**:
```bash
python main.py "What are the main findings about climate change?"
```

### Web Interface

Launch the interactive web interface:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
codemate/
├── main.py                 # CLI entry point
├── app.py                  # Streamlit web interface
├── config.yml              # Configuration file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/                  # Document storage directory
├── exports/               # Export output directory
├── cache/                 # Caching directory
├── logs/                  # Log files
├── ingestion.py           # Document loading and processing
├── embeddings.py          # Embedding generation and caching
├── vector_store.py        # Vector search and indexing
├── reasoning.py           # Query decomposition and reasoning
├── synthesizer.py         # Result synthesis and formatting
├── exporter.py            # Multi-format export system
├── utils.py               # Utility functions
└── summarizer.py          # Text summarization utilities
```

## ⚙️ Configuration

The system is highly configurable through `config.yml`. Key configuration sections:

### Document Ingestion
```yaml
ingestion:
  data_dir: "data"
  chunk_size: 1000
  chunk_overlap: 200
  supported_formats: [".pdf", ".docx", ".txt", ".md", ".html"]
  max_file_size_mb: 50
```

### Embedding Model
```yaml
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32
  cache_embeddings: true
  normalize_embeddings: true
```

### Vector Store
```yaml
vector_store:
  index_type: "faiss"
  faiss_index_type: "IndexFlatL2"
  persist_index: true
  search_top_k: 10
  similarity_threshold: 0.7
```

### Reasoning Engine
```yaml
reasoning:
  decomposition_strategy: "intelligent"  # simple, intelligent, llm_based
  max_subtasks: 5
  top_k: 5
  enable_abstractive_summarization: true
  enable_llm_reasoning: false
```

## 🔧 Usage Examples

### Command Line Interface

**Basic research query**:
```bash
python main.py "What are the key findings in machine learning research?"
```

**With custom configuration**:
```bash
python main.py "Compare different approaches to natural language processing" --config custom_config.yml
```

**Verbose output**:
```bash
python main.py "Analyze climate change impacts" --verbose
```

### Programmatic Usage

```python
from utils import load_config
from ingestion import DocumentLoader
from embeddings import EmbeddingModel
from vector_store import VectorStore
from reasoning import Reasoner
from synthesizer import Synthesizer
from exporter import Exporter

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
docs = loader.load()
docs = embedder.embed_documents(docs)
store.build_index(docs)

# Research query
query = "What are the main findings about renewable energy?"
subtasks = reasoner.decompose(query)
results = reasoner.solve(subtasks)
final_answer = synthesizer.combine(results, query)

# Export results
export_path = exporter.export(query, final_answer, results)
print(f"Results exported to: {export_path}")
```

## 🎯 Advanced Features

### Intelligent Query Decomposition

The system supports three decomposition strategies:

1. **Simple**: Basic splitting on delimiters
2. **Intelligent**: Keyword analysis and semantic understanding
3. **LLM-based**: Uses local language models for decomposition

### Hybrid Search

Combines semantic and keyword search for better results:
```python
# Enable hybrid search in vector store
results = store.hybrid_search(query_vector, query_text, alpha=0.7)
```

### Confidence Scoring

Automatic confidence assessment based on:
- Evidence quality and quantity
- Source diversity
- Answer completeness
- Contradiction detection

### Export Formats

**Markdown**: Human-readable format with citations
**PDF**: Professional reports with tables and formatting
**JSON**: Structured data for programmatic use
**HTML**: Web-ready format with styling

## 🔍 Research Workflow

1. **Document Ingestion**: Load and process documents from the data directory
2. **Embedding Generation**: Create vector representations with caching
3. **Index Building**: Build searchable vector index with metadata
4. **Query Processing**: Decompose complex queries into subtasks
5. **Evidence Retrieval**: Find relevant information using hybrid search
6. **Answer Generation**: Synthesize answers with confidence scoring
7. **Result Export**: Export in multiple formats with citations

## 🛠️ Customization

### Adding New Document Types

Extend the `DocumentLoader` class to support new file formats:

```python
def _read_custom_format(self, path: str) -> str:
    # Implement custom file reading logic
    pass
```

### Custom Embedding Models

Configure different embedding models in `config.yml`:

```yaml
embeddings:
  model_name: "sentence-transformers/all-mpnet-base-v2"  # Larger, more accurate
  # or
  model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual
```

### Custom Reasoning Strategies

Implement custom reasoning logic by extending the `Reasoner` class:

```python
def _custom_decomposition(self, query: str) -> List[str]:
    # Implement custom query decomposition
    pass
```

## 📊 Performance Optimization

### Caching
- Embedding cache for faster repeated queries
- Vector index persistence for large document collections
- Configuration-based cache management

### Memory Management
- Configurable batch sizes for processing
- Memory monitoring and optimization
- Efficient data structures for large datasets

### Parallel Processing
- Batch embedding generation
- Concurrent document processing
- Optimized search algorithms

## 🔒 Security and Privacy

- **Local Processing**: All processing happens locally, no external API calls
- **Input Validation**: Comprehensive query validation and sanitization
- **File Security**: Safe file handling with size limits and format validation
- **Data Privacy**: No data leaves your local environment

## 🐛 Troubleshooting

### Common Issues

**No documents found**:
- Ensure documents are in the `data/` directory
- Check supported file formats in configuration
- Verify file permissions

**Memory issues**:
- Reduce batch sizes in configuration
- Use smaller embedding models
- Enable memory monitoring

**Slow performance**:
- Enable caching for embeddings and vector index
- Use appropriate FAISS index types for your dataset size
- Optimize chunk sizes for your documents

### Logging

Enable detailed logging by setting the log level in `config.yml`:

```yaml
logging:
  level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR
  log_file: "logs/researcher.log"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Streamlit](https://streamlit.io/) for the web interface
- [ReportLab](https://www.reportlab.com/) for PDF generation

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation
- Review the configuration examples

---

**Deep Researcher Agent** - Empowering research with AI-driven local document analysis and synthesis.
