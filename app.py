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
        help="How to break down complex queries"
    )
    
    max_subtasks = st.sidebar.slider(
        "Max Subtasks",
        min_value=1,
        max_value=10,
        value=config.get('reasoning', {}).get('max_subtasks', 5),
        help="Maximum number of subtasks to create"
    )
    
    top_k = st.sidebar.slider(
        "Evidence per Subtask",
        min_value=1,
        max_value=20,
        value=config.get('reasoning', {}).get('top_k', 5),
        help="Number of evidence pieces per subtask"
    )
    
    # Export settings
    st.sidebar.subheader("📄 Export Settings")
    
    export_formats = st.sidebar.multiselect(
        "Export Formats",
        ["markdown", "pdf", "json", "html"],
        default=config.get('export', {}).get('formats', ['markdown']),
        help="Formats to export results in"
    )
    
    include_citations = st.sidebar.checkbox(
        "Include Citations",
        value=config.get('export', {}).get('include_citations', True),
        help="Include source citations in exports"
    )
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        enable_contradiction_detection = st.checkbox(
            "Contradiction Detection",
            value=config.get('synthesizer', {}).get('enable_contradiction_detection', True)
        )
        
        enable_confidence_scoring = st.checkbox(
            "Confidence Scoring",
            value=config.get('synthesizer', {}).get('confidence_scoring', True)
        )
        
        enable_hybrid_search = st.checkbox(
            "Hybrid Search",
            value=config.get('advanced', {}).get('enable_hybrid_search', True)
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
        if st.button("🗑️ Clear All Documents"):
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
    st.subheader("🔍 Research Interface")
    
    # Query input
    query = st.text_area(
        "Research Query",
        placeholder="Enter your research question here...\n\nExample: What are the main findings about climate change and its impacts?",
        height=100,
        help="Enter a detailed research question. The system will break it down into subtasks and analyze your documents."
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        run_research = st.button("🚀 Run Research", type="primary", use_container_width=True)
    
    with col2:
        if st.button("💡 Example Queries", use_container_width=True):
            st.session_state.show_examples = True
    
    # Handle example queries display
    if st.session_state.get('show_examples', False):
        example_queries = [
            "What are the key findings and conclusions?",
            "Compare and contrast different approaches mentioned",
            "What are the main challenges and limitations?",
            "Summarize the methodology and results",
            "What are the implications for future research?"
        ]
        selected_example = st.selectbox("Select example:", example_queries)
        if st.button("Use Selected Example"):
            st.session_state.example_query = selected_example
            st.session_state.show_examples = False
            st.rerun()
        if st.button("Cancel"):
            st.session_state.show_examples = False
            st.rerun()
    
    with col3:
        if st.button("🔄 Clear Results", use_container_width=True):
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

def follow_up_questions(results_data: Dict[str, Any]):
    """Generate and display follow-up questions."""
    if not results_data:
        return
    
    st.subheader("💡 Follow-up Questions")
    
    # Generate follow-up questions based on results
    follow_ups = [
        "Can you provide more details about the methodology?",
        "What are the limitations mentioned in the research?",
        "Are there any conflicting findings?",
        "What are the practical implications?",
        "What future research directions are suggested?"
    ]
    
    col1, col2, col3 = st.columns(3)
    
    for i, question in enumerate(follow_ups):
        with [col1, col2, col3][i % 3]:
            if st.button(f"❓ {question[:50]}...", use_container_width=True):
                st.session_state.follow_up_query = question
    
    # Handle follow-up query
    follow_up_query = st.session_state.get('follow_up_query')
    if follow_up_query:
        st.info(f"Follow-up query: {follow_up_query}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Research Follow-up", use_container_width=True):
                # Process follow-up query
                config_updates = sidebar_configuration()
                follow_up_results = process_research(follow_up_query, config_updates)
                if follow_up_results:
                    st.session_state.research_results = follow_up_results
                del st.session_state.follow_up_query
                st.rerun()
        with col2:
            if st.button("❌ Cancel Follow-up", use_container_width=True):
                del st.session_state.follow_up_query
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
        
        # Process research
        results_data = process_research(query, config_updates)
        if results_data:
            st.session_state.research_results = results_data
    
    # Display results if available
    research_results = st.session_state.get('research_results')
    if research_results:
        display_results(research_results)
        follow_up_questions(research_results)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🔍 Deep Researcher Agent - Powered by Local AI</p>
        <p>All processing happens locally - your data never leaves your machine</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
