"""
Advanced Streamlit Web Interface for Deep Researcher Agent

Features:
- Interactive research interface with real-time search
- Advanced configuration options
- File upload capabilities
- Follow-up questions and query refinement
- Comprehensive result visualization
- Multiple export formats
- AI assistant with reasoning explanations
"""

import streamlit as st
import os
import logging
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Import our enhanced modules
from utils import load_config, validate_query, format_timestamp
from ingestion import DocumentLoader
from embeddings import EmbeddingModel
from vector_store import VectorStore
from reasoning import Reasoner
from synthesizer import Synthesizer
from exporter import Exporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and other non-serializable objects."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def clean_data_for_json(data):
    """Recursively clean data to make it JSON serializable."""
    if isinstance(data, dict):
        return {key: clean_data_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.floating, np.bool_)):
        return data.item()
    elif hasattr(data, '__dict__'):
        # Handle custom objects by converting to dict
        return clean_data_for_json(data.__dict__)
    else:
        return data

# Page configuration
st.set_page_config(
    page_title="Deep Researcher Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Session state will be handled with safe access patterns

def load_system_config():
    """Load and cache system configuration."""
    config = st.session_state.get('config', None)
    if config is None:
        try:
            config = load_config("config.yml")
            st.session_state.config = config
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            config = {}
            st.session_state.config = config
    return config

def initialize_components():
    """Initialize system components with caching."""
    components = st.session_state.get('components', {})
    if not components:
        config = load_system_config()
        
        try:
            components = {
                'loader': DocumentLoader(config.get('ingestion', {})),
                'embedder': EmbeddingModel(config.get('embeddings', {})),
                'store': VectorStore(config.get('vector_store', {})),
                'reasoner': None,  # Will be initialized with store and embedder
                'synthesizer': Synthesizer(config.get('synthesizer', {})),
                'exporter': Exporter(config.get('export', {}))
            }
            st.session_state.components = components
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            components = {}
            st.session_state.components = components
    
    return components

def display_header():
    """Display the main header and description."""
    st.markdown('<h1 class="main-header">🔍 Deep Researcher Agent</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Advanced AI-powered research assistant for local document analysis and synthesis
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_system_status():
    """Display system status and statistics."""
    components = initialize_components()
    config = load_system_config()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Check data directory
        data_dir = config.get('ingestion', {}).get('data_dir', 'data')
        data_files = list(Path(data_dir).glob('*')) if Path(data_dir).exists() else []
        st.metric("📄 Documents", len(data_files))
    
    with col2:
        # Check cache status
        cache_dir = config.get('embeddings', {}).get('embedding_cache_dir', 'cache/embeddings')
        cache_files = list(Path(cache_dir).glob('*')) if Path(cache_dir).exists() else []
        st.metric("💾 Cache Size", len(cache_files))
    
    with col3:
        # Check exports
        export_dir = config.get('export', {}).get('output_dir', 'exports')
        export_files = list(Path(export_dir).glob('*')) if Path(export_dir).exists() else []
        st.metric("📊 Exports", len(export_files))
    
    with col4:
        # System status
        status = "🟢 Ready" if components else "🔴 Not Ready"
        st.metric("Status", status)

def get_current_config():
    """Get current configuration from session state or return defaults."""
    config = load_system_config()
    
    # Return current config from session state if available, otherwise defaults
    return {
        'reasoning': {
            'decomposition_strategy': st.session_state.get('decomposition_strategy', 'intelligent'),
            'max_subtasks': st.session_state.get('max_subtasks', config.get('reasoning', {}).get('max_subtasks', 5)),
            'top_k': st.session_state.get('top_k', config.get('reasoning', {}).get('top_k', 5))
        },
        'export': {
            'formats': st.session_state.get('export_formats', config.get('export', {}).get('formats', ['markdown'])),
            'include_citations': st.session_state.get('include_citations', config.get('export', {}).get('include_citations', True))
        },
        'synthesizer': {
            'enable_contradiction_detection': st.session_state.get('enable_contradiction_detection', config.get('synthesizer', {}).get('enable_contradiction_detection', True)),
            'confidence_scoring': st.session_state.get('enable_confidence_scoring', config.get('synthesizer', {}).get('confidence_scoring', True))
        },
        'advanced': {
            'enable_hybrid_search': st.session_state.get('enable_hybrid_search', config.get('advanced', {}).get('enable_hybrid_search', True))
        }
    }

def sidebar_configuration():
    """Display configuration options in sidebar."""
    st.sidebar.header("⚙️ Configuration")
    
    config = load_system_config()
    
    # Research settings
    st.sidebar.subheader("🔍 Research Settings")
    
    decomposition_strategy = st.sidebar.selectbox(
        "Query Decomposition",
        ["simple", "intelligent", "llm_based"],
        index=1,
        help="How to break down complex queries",
        key="decomposition_strategy"
    )
    
    max_subtasks = st.sidebar.slider(
        "Max Subtasks",
        min_value=1,
        max_value=10,
        value=config.get('reasoning', {}).get('max_subtasks', 5),
        help="Maximum number of subtasks to create",
        key="max_subtasks"
    )
    
    top_k = st.sidebar.slider(
        "Evidence per Subtask",
        min_value=1,
        max_value=20,
        value=config.get('reasoning', {}).get('top_k', 5),
        help="Number of evidence pieces per subtask",
        key="top_k"
    )
    
    # Export settings
    st.sidebar.subheader("📄 Export Settings")
    
    export_formats = st.sidebar.multiselect(
        "Export Formats",
        ["markdown", "pdf", "json", "html"],
        default=config.get('export', {}).get('formats', ['markdown']),
        help="Formats to export results in",
        key="export_formats"
    )
    
    include_citations = st.sidebar.checkbox(
        "Include Citations",
        value=config.get('export', {}).get('include_citations', True),
        help="Include source citations in exports",
        key="include_citations"
    )
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        enable_contradiction_detection = st.checkbox(
            "Contradiction Detection",
            value=config.get('synthesizer', {}).get('enable_contradiction_detection', True),
            key="enable_contradiction_detection"
        )
        
        enable_confidence_scoring = st.checkbox(
            "Confidence Scoring",
            value=config.get('synthesizer', {}).get('confidence_scoring', True),
            key="enable_confidence_scoring"
        )
        
        enable_hybrid_search = st.checkbox(
            "Hybrid Search",
            value=config.get('advanced', {}).get('enable_hybrid_search', True),
            key="enable_hybrid_search"
        )
    
    # Return updated config
    return {
        'reasoning': {
            'decomposition_strategy': decomposition_strategy,
            'max_subtasks': max_subtasks,
            'top_k': top_k
        },
        'export': {
            'formats': export_formats,
            'include_citations': include_citations
        },
        'synthesizer': {
            'enable_contradiction_detection': enable_contradiction_detection,
            'confidence_scoring': enable_confidence_scoring
        },
        'advanced': {
            'enable_hybrid_search': enable_hybrid_search
        }
    }

def file_upload_section():
    """Handle file uploads."""
    st.subheader("📁 Document Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt', 'md', 'html'],
            accept_multiple_files=True,
            help="Upload research documents for analysis"
        )
    
    with col2:
        if st.button("🗑️ Clear All Documents", key="clear_documents_btn"):
            data_dir = Path("data")
            if data_dir.exists():
                for file in data_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
            st.success("All documents cleared!")
            # Clear session state to refresh the interface
            if 'research_results' in st.session_state:
                del st.session_state.research_results
            st.rerun()
    
    # Handle file uploads without automatic rerun
    if uploaded_files:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        uploaded_count = 0
        for uploaded_file in uploaded_files:
            file_path = data_dir / uploaded_file.name
            # Check if file already exists to avoid overwriting
            if not file_path.exists():
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                uploaded_count += 1
        
        if uploaded_count > 0:
            st.success(f"Uploaded {uploaded_count} new files!")
            # Clear any existing research results since we have new documents
            if st.session_state.get('research_results'):
                del st.session_state.research_results
        else:
            st.info("All selected files already exist in the data directory.")
    
    # Display current documents
    data_dir = Path("data")
    if data_dir.exists():
        existing_files = list(data_dir.glob("*"))
        if existing_files:
            st.markdown("**Current Documents:**")
            for file in existing_files:
                if file.is_file():
                    file_size = file.stat().st_size
                    size_mb = file_size / (1024 * 1024)
                    st.write(f"📄 {file.name} ({size_mb:.2f} MB)")
        else:
            st.info("No documents found. Upload some files to get started!")

def research_interface():
    """Main research interface."""
    st.markdown("---")
    st.subheader("🔍 Research Interface")
    
    # Query input with better styling
    st.markdown("#### 💭 Enter Your Research Question")
    query = st.text_area(
        "Research Query",
        placeholder="Enter your research question here...\n\nExample: What are the main findings about climate change and its impacts?",
        height=120,
        help="Enter a detailed research question. The system will break it down into subtasks and analyze your documents.",
        key="main_research_query"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        run_research = st.button("🚀 Run Research", type="primary", use_container_width=True, key="run_research_btn")
    
    with col2:
        if st.button("💡 Example Queries", use_container_width=True, key="example_queries_btn"):
            st.session_state.show_examples = True
    
    # Handle example queries display
    if st.session_state.get('show_examples', False):
        # Generate document-specific example queries
        example_queries = generate_document_specific_examples()
        selected_example = st.selectbox("Select example:", example_queries, key="example_selectbox")
        if st.button("Use Selected Example", key="use_example_btn"):
            st.session_state.example_query = selected_example
            st.session_state.show_examples = False
            st.rerun()
        if st.button("Cancel", key="cancel_example_btn"):
            st.session_state.show_examples = False
            st.rerun()
    
    with col3:
        if st.button("🔄 Clear Results", use_container_width=True, key="clear_results_btn"):
            if st.session_state.get('research_results'):
                del st.session_state.research_results
            st.rerun()
    
    # Handle example query selection
    example_query = st.session_state.get('example_query')
    if example_query:
        query = example_query
        del st.session_state.example_query
    
    return query, run_research

def process_research(query: str, config_updates: Dict[str, Any]):
    """Process the research query."""
    components = initialize_components()
    base_config = load_system_config()
    
    # Merge configurations
    config = {**base_config, **config_updates}
    
    try:
        # Step 1: Load documents
        with st.spinner("📄 Loading documents..."):
            docs = components['loader'].load()
            if not docs:
                st.error("No documents found! Please upload some documents first.")
                return None
        
        # Display document stats
        stats = components['loader'].get_document_stats(docs)
        st.info(f"📊 Loaded {stats['total_documents']} documents, {stats['total_chunks']} chunks")
        
        # Step 2: Generate embeddings
        with st.spinner("🧠 Generating embeddings..."):
            docs = components['embedder'].embed_documents(docs)
            cache_stats = components['embedder'].get_cache_stats()
            st.info(f"💾 Cache size: {cache_stats['cache_size']} embeddings")
        
        # Step 3: Build vector index
        with st.spinner("🔍 Building search index..."):
            components['store'].build_index(docs)
            store_stats = components['store'].get_stats()
            st.info(f"📈 Index built: {store_stats['total_documents']} documents, {store_stats['dimension']}D vectors")
        
        # Step 4: Initialize reasoner with updated config
        reasoner = Reasoner(components['store'], components['embedder'], config['reasoning'])
        
        # Step 5: Process query
        with st.spinner("🤔 Processing query..."):
            subtasks = reasoner.decompose(query)
            st.info(f"📋 Decomposed into {len(subtasks)} subtasks")
        
        # Step 6: Solve subtasks
        with st.spinner("🔬 Analyzing evidence..."):
            results = reasoner.solve(subtasks)
            total_evidence = sum(len(r.get('evidence', [])) for r in results)
            avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results) if results else 0
            st.info(f"📊 Found {total_evidence} evidence pieces, avg confidence: {avg_confidence:.2f}")
        
        # Step 7: Synthesize results
        with st.spinner("📝 Synthesizing results..."):
            synthesizer = Synthesizer(config['synthesizer'])
            final_answer = synthesizer.combine(results, query)
        
        # Step 8: Export results
        with st.spinner("💾 Exporting results..."):
            exporter = Exporter(config['export'])
            export_path = exporter.export(query, final_answer, results)
        
        # Clean results data to remove numpy arrays and make it JSON serializable
        cleaned_results = clean_data_for_json(results)
        
        return {
            'query': query,
            'subtasks': subtasks,
            'results': cleaned_results,
            'final_answer': final_answer,
            'export_path': export_path,
            'stats': {
                'total_documents': stats['total_documents'],
                'total_chunks': stats['total_chunks'],
                'total_evidence': total_evidence,
                'avg_confidence': avg_confidence
            }
        }
        
    except Exception as e:
        st.error(f"Error during research: {e}")
        logger.error(f"Research error: {e}")
        return None

def display_results(results_data: Dict[str, Any]):
    """Display research results in a comprehensive format."""
    if not results_data:
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Executive Summary", 
        "🔬 Detailed Analysis", 
        "📊 Analytics", 
        "📄 Evidence", 
        "💾 Export"
    ])
    
    with tab1:
        st.subheader("📋 Executive Summary")
        st.markdown(results_data['final_answer'])
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents", results_data['stats']['total_documents'])
        with col2:
            st.metric("Chunks", results_data['stats']['total_chunks'])
        with col3:
            st.metric("Evidence", results_data['stats']['total_evidence'])
        with col4:
            confidence = results_data['stats']['avg_confidence']
            confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.4 else "confidence-low"
            st.markdown(f'<div class="metric-card">Overall Confidence: <span class="{confidence_class}">{confidence:.2f}</span></div>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("🔬 Detailed Analysis")
        
        for i, result in enumerate(results_data['results'], 1):
            with st.expander(f"Subtask {i}: {result.get('subtask', 'Unknown')[:100]}..."):
                # Answer
                st.markdown("**Answer:**")
                st.write(result.get('answer', 'No answer generated'))
                
                # Confidence
                confidence = result.get('confidence', 0)
                confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.4 else "confidence-low"
                st.markdown(f'**Confidence:** <span class="{confidence_class}">{confidence:.2f}</span>', unsafe_allow_html=True)
                
                # Evidence count
                evidence_count = len(result.get('evidence', []))
                st.markdown(f"**Evidence Pieces:** {evidence_count}")
                
                # Metadata
                metadata = result.get('metadata', {})
                if metadata:
                    st.markdown("**Metadata:**")
                    st.json(metadata)
    
    with tab3:
        st.subheader("📊 Analytics")
        
        # Confidence distribution
        confidences = [r.get('confidence', 0) for r in results_data['results']]
        if confidences:
            df_conf = pd.DataFrame({'Confidence': confidences})
            st.bar_chart(df_conf)
        
        # Evidence per subtask
        evidence_counts = [len(r.get('evidence', [])) for r in results_data['results']]
        if evidence_counts:
            df_evidence = pd.DataFrame({
                'Subtask': list(range(1, len(evidence_counts) + 1)),
                'Evidence Count': evidence_counts
            })
            st.bar_chart(df_evidence.set_index('Subtask'))
        
        # Source diversity
        all_sources = set()
        for result in results_data['results']:
            for evidence in result.get('evidence', []):
                source = evidence.get('meta', {}).get('source', 'unknown')
                all_sources.add(source)
        
        if all_sources:
            st.markdown("**Source Diversity:**")
            st.write(f"Found evidence from {len(all_sources)} unique sources:")
            for source in sorted(all_sources):
                st.write(f"• {source}")
    
    with tab4:
        st.subheader("📄 Evidence Explorer")
        
        # Evidence search
        search_term = st.text_input("Search evidence:", placeholder="Enter keywords to search evidence...")
        
        all_evidence = []
        for i, result in enumerate(results_data['results']):
            for j, evidence in enumerate(result.get('evidence', [])):
                evidence['subtask_id'] = i + 1
                evidence['evidence_id'] = j + 1
                all_evidence.append(evidence)
        
        # Filter evidence
        if search_term:
            filtered_evidence = [
                ev for ev in all_evidence 
                if search_term.lower() in ev.get('text', '').lower()
            ]
        else:
            filtered_evidence = all_evidence
        
        st.write(f"Showing {len(filtered_evidence)} evidence pieces")
        
        # Display evidence
        for evidence in filtered_evidence:
            with st.expander(f"Subtask {evidence['subtask_id']} - Evidence {evidence['evidence_id']} (Score: {evidence.get('score', 0):.3f})"):
                st.markdown(f"**Source:** {evidence.get('meta', {}).get('source', 'unknown')}")
                st.markdown(f"**Relevance Score:** {evidence.get('score', 0):.3f}")
                st.markdown("**Content:**")
                st.write(evidence.get('text', ''))
    
    with tab5:
        st.subheader("💾 Export Results")
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Export Formats:**")
            export_path = results_data['export_path']
            if export_path and os.path.exists(export_path):
                with open(export_path, "rb") as f:
                    st.download_button(
                        label="📄 Download Markdown",
                        data=f.read(),
                        file_name=os.path.basename(export_path),
                        mime="text/markdown"
                    )
        
        with col2:
            # JSON export (data is already cleaned in process_research)
            json_data = {
                'query': results_data['query'],
                'final_answer': results_data['final_answer'],
                'results': results_data['results'],
                'stats': results_data['stats']
            }
            
            st.download_button(
                label="📊 Download JSON",
                data=json.dumps(json_data, indent=2, cls=NumpyEncoder),
                file_name=f"research_results_{format_timestamp()}.json",
                mime="application/json"
            )
        
        # Export path info
        if export_path:
            st.info(f"📁 Export saved to: {export_path}")

def generate_document_specific_examples() -> List[str]:
    """Generate document-specific example queries based on uploaded files."""
    try:
        data_dir = Path("data")
        if not data_dir.exists():
            return [
                "What are the key findings and conclusions?",
                "Compare and contrast different approaches mentioned",
                "What are the main challenges and limitations?",
                "Summarize the methodology and results",
                "What are the implications for future research?"
            ]
        
        files = list(data_dir.glob("*"))
        if not files:
            return [
                "What are the key findings and conclusions?",
                "Compare and contrast different approaches mentioned",
                "What are the main challenges and limitations?",
                "Summarize the methodology and results",
                "What are the implications for future research?"
            ]
        
        # Analyze document types and generate relevant examples
        doc_types = {}
        total_files = len(files)
        
        for file_path in files:
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in ['.pdf', '.docx', '.txt', '.md', '.html']:
                    doc_types[ext] = doc_types.get(ext, 0) + 1
        
        examples = []
        
        # Document type specific examples
        if '.pdf' in doc_types:
            examples.extend([
                "What are the main sections and key findings?",
                "Summarize the abstract and conclusions",
                "What methodology is used in this research?"
            ])
        
        if '.docx' in doc_types:
            examples.extend([
                "What are the key headings and main points?",
                "Extract the main arguments and evidence",
                "What are the recommendations provided?"
            ])
        
        # Multiple document examples
        if total_files > 1:
            examples.extend([
                "What are the common themes across all documents?",
                "How do these documents relate to each other?",
                "Compare the different perspectives presented"
            ])
        
        # General document analysis examples
        examples.extend([
            "What are the key takeaways and main points?",
            "What questions does this document raise?",
            "What are the practical implications?",
            "What future research directions are suggested?"
        ])
        
        # Remove duplicates and limit to 6 examples
        unique_examples = list(dict.fromkeys(examples))[:6]
        
        return unique_examples
        
    except Exception as e:
        st.error(f"Error generating example queries: {e}")
        return [
            "What are the key findings and conclusions?",
            "Compare and contrast different approaches mentioned",
            "What are the main challenges and limitations?",
            "Summarize the methodology and results",
            "What are the implications for future research?"
        ]

def generate_document_specific_followups(results: Dict[str, Any]) -> List[str]:
    """Generate document-specific follow-up questions based on content analysis."""
    follow_ups = []
    
    if not results or not results.get('subtasks'):
        return follow_ups
    
    # Extract key information from results
    final_answer = results.get('final_answer', '')
    subtasks = results.get('subtasks', [])
    evidence = results.get('evidence', [])
    
    # Analyze document content for intelligent follow-ups
    content_indicators = {
        'methodology': ['method', 'approach', 'technique', 'algorithm', 'model', 'framework'],
        'results': ['result', 'finding', 'outcome', 'conclusion', 'performance', 'accuracy'],
        'limitations': ['limitation', 'constraint', 'challenge', 'issue', 'problem', 'weakness'],
        'applications': ['application', 'use case', 'implementation', 'deployment', 'practical'],
        'comparison': ['compare', 'versus', 'vs', 'different', 'alternative', 'baseline'],
        'future_work': ['future', 'next step', 'improvement', 'enhancement', 'extension']
    }
    
    # Check what types of content are present
    content_types = []
    text_to_analyze = final_answer.lower()
    
    for content_type, keywords in content_indicators.items():
        if any(keyword in text_to_analyze for keyword in keywords):
            content_types.append(content_type)
    
    # Generate specific follow-ups based on content
    if 'methodology' in content_types:
        follow_ups.extend([
            "Can you explain the methodology in more detail?",
            "What are the key steps in the proposed approach?",
            "How does this method compare to existing techniques?"
        ])
    
    if 'results' in content_types:
        follow_ups.extend([
            "What are the quantitative results and metrics?",
            "How significant are these findings?",
            "What do the results tell us about the problem?"
        ])
    
    if 'limitations' in content_types:
        follow_ups.extend([
            "What are the main limitations and constraints?",
            "How do these limitations affect the conclusions?",
            "What could be done to address these limitations?"
        ])
    
    if 'applications' in content_types:
        follow_ups.extend([
            "What are the practical applications of this work?",
            "How can this be implemented in real-world scenarios?",
            "What are the potential use cases?"
        ])
    
    if 'comparison' in content_types:
        follow_ups.extend([
            "How does this compare to other approaches?",
            "What are the advantages and disadvantages?",
            "Which method performs better and why?"
        ])
    
    if 'future_work' in content_types:
        follow_ups.extend([
            "What future research directions are suggested?",
            "What improvements could be made?",
            "What are the next steps for this research?"
        ])
    
    # Add general document analysis questions
    follow_ups.extend([
        "What are the key takeaways from this document?",
        "What questions does this document raise?",
        "How does this relate to current trends in the field?",
        "What are the most important points to remember?"
    ])
    
    # Remove duplicates and limit to 6 questions
    unique_follow_ups = list(dict.fromkeys(follow_ups))[:6]
    
    return unique_follow_ups

def generate_document_structure_followups() -> List[str]:
    """Generate follow-ups based on document structure and content."""
    try:
        components = initialize_components()
        loader = components.get('loader')
        
        if not loader:
            return []
        
        # Get document statistics
        data_dir = Path("data")
        if not data_dir.exists():
            return []
        
        files = list(data_dir.glob("*"))
        if not files:
            return []
        
        # Analyze document types and generate relevant questions
        doc_types = {}
        total_pages = 0
        
        for file_path in files:
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in ['.pdf', '.docx', '.txt', '.md', '.html']:
                    doc_types[ext] = doc_types.get(ext, 0) + 1
                    # Estimate pages for PDFs
                    if ext == '.pdf':
                        try:
                            import PyPDF2
                            with open(file_path, 'rb') as f:
                                pdf_reader = PyPDF2.PdfReader(f)
                                total_pages += len(pdf_reader.pages)
                        except:
                            pass
        
        structure_followups = []
        
        # Document type specific questions
        if '.pdf' in doc_types:
            structure_followups.extend([
                "What are the main sections and chapters?",
                "Can you summarize each section?",
                "What figures and tables are included?"
            ])
        
        if '.docx' in doc_types:
            structure_followups.extend([
                "What are the key headings and structure?",
                "Can you extract the main points from each section?"
            ])
        
        if total_pages > 10:
            structure_followups.extend([
                "This is a long document - can you break it down by sections?",
                "What are the most important pages or sections?"
            ])
        
        # Multiple document questions
        if len(files) > 1:
            structure_followups.extend([
                "How do these documents relate to each other?",
                "What are the common themes across all documents?",
                "Can you compare the different documents?"
            ])
        
        return structure_followups[:4]  # Limit to 4 structure questions
        
    except Exception as e:
        st.error(f"Error generating structure follow-ups: {e}")
        return []

def follow_up_questions(results_data: Dict[str, Any]):
    """Generate and display intelligent follow-up questions."""
    if not results_data:
        return
    
    st.markdown("---")
    st.markdown("### 🤔 Intelligent Follow-up Questions")
    st.markdown("Continue your research with these contextually relevant questions:")
    
    # Generate document-specific follow-ups
    content_followups = generate_document_specific_followups(results_data)
    structure_followups = generate_document_structure_followups()
    
    # Combine and deduplicate
    all_followups = content_followups + structure_followups
    unique_followups = list(dict.fromkeys(all_followups))[:8]  # Limit to 8 questions
    
    # If no intelligent follow-ups, use fallback
    if not unique_followups:
        unique_followups = [
            "What are the main limitations mentioned?",
            "How do the results compare to previous studies?",
            "What are the practical applications?",
            "What future research directions are suggested?"
        ]
    
    # Display follow-ups in a grid with better styling
    st.markdown("#### 💡 Suggested Questions")
    cols = st.columns(2)
    
    for i, question in enumerate(unique_followups):
        col_idx = i % 2
        with cols[col_idx]:
            # Create a more attractive button with better styling
            button_text = f"❓ {question[:55]}{'...' if len(question) > 55 else ''}"
            if st.button(button_text, use_container_width=True, key=f"followup_{i}"):
                st.session_state.follow_up_query = question
                st.rerun()
    
    # Handle follow-up query
    follow_up_query = st.session_state.get('follow_up_query')
    if follow_up_query:
        st.markdown("---")
        st.markdown("#### 🎯 Selected Follow-up Question")
        st.info(f"**Query:** {follow_up_query}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Research Follow-up", use_container_width=True, key="research_followup_btn", type="primary"):
                # Set flag to process follow-up
                st.session_state.process_followup = True
                st.session_state.followup_query_to_process = follow_up_query
                st.rerun()
        with col2:
            if st.button("❌ Cancel Follow-up", use_container_width=True, key="cancel_followup_btn"):
                del st.session_state.follow_up_query
                st.rerun()
    
    # Add custom follow-up option
    st.markdown("---")
    st.markdown("#### 💬 Custom Question")
    st.markdown("**Or ask your own specific question:**")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        custom_query = st.text_input("Custom follow-up question:", placeholder="Enter your specific question...", key="custom_query_input")
    with col2:
        if st.button("🚀 Ask Question", use_container_width=True, key="ask_custom_btn", type="primary") and custom_query:
            # Set flag to process custom question
            st.session_state.process_custom = True
            st.session_state.custom_query_to_process = custom_query
            st.rerun()

def main():
    """Main application function."""
    display_header()
    display_system_status()
    
    # Sidebar configuration
    config_updates = sidebar_configuration()
    
    # File upload section
    file_upload_section()
    
    # Main research interface
    query, run_research = research_interface()
    
    # Process research if requested
    if run_research and query:
        # Validate query
        is_valid, error_msg = validate_query(query)
        if not is_valid:
            st.error(f"Invalid query: {error_msg}")
            return
        
        # Process research with visual feedback
        with st.spinner("🔍 Processing your research query..."):
            results_data = process_research(query, config_updates)
            if results_data:
                st.session_state.research_results = results_data
                st.success("✅ Research completed successfully!")
            else:
                st.error("❌ Failed to process research query")
    
    # Process follow-up research if requested
    if st.session_state.get('process_followup', False):
        followup_query = st.session_state.get('followup_query_to_process')
        if followup_query:
            with st.spinner("🔍 Processing follow-up research..."):
                config_updates = get_current_config()
                follow_up_results = process_research(followup_query, config_updates)
                if follow_up_results:
                    st.session_state.research_results = follow_up_results
                    st.success("✅ Follow-up research completed!")
                else:
                    st.error("❌ Failed to process follow-up research")
        
        # Clear flags
        del st.session_state.process_followup
        if 'followup_query_to_process' in st.session_state:
            del st.session_state.followup_query_to_process
        if 'follow_up_query' in st.session_state:
            del st.session_state.follow_up_query
    
    # Process custom question if requested
    if st.session_state.get('process_custom', False):
        custom_query = st.session_state.get('custom_query_to_process')
        if custom_query:
            with st.spinner("🔍 Processing custom question..."):
                config_updates = get_current_config()
                custom_results = process_research(custom_query, config_updates)
                if custom_results:
                    st.session_state.research_results = custom_results
                    st.success("✅ Custom question research completed!")
                else:
                    st.error("❌ Failed to process custom question")
        
        # Clear flags
        del st.session_state.process_custom
        if 'custom_query_to_process' in st.session_state:
            del st.session_state.custom_query_to_process
    
    # Display results if available
    research_results = st.session_state.get('research_results')
    if research_results:
        st.markdown("---")
        st.markdown("## 📊 Research Results")
        
        # Add a success banner
        st.success("🎉 Research completed! Explore your results below.")
        
        display_results(research_results)
        follow_up_questions(research_results)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Deep Researcher Agent - Powered by Local AI</p>
        <p>All processing happens locally - your data never leaves your machine</p>
        <p>Designed and Developed By Aman Goel</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
