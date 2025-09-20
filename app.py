import streamlit as st
import os
import logging

from ingestion import DocumentLoader
from embeddings import EmbeddingModel
from vector_store import VectorStore
from reasoning import Reasoner
from synthesizer import Synthesizer
from exporter import Exporter

# --- Logging ---
logging.basicConfig(level=logging.INFO)

# --- Paths ---
DATA_DIR = "data"
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# --- Config ---
cfg = {
    "reasoning": {
        "top_k": 5,
        "summarizer_model": "facebook/bart-large-cnn",
        "summarizer_max_input_chars": 3000,
        "summarizer_min_length": 30,
        "summarizer_max_length": 200,
    },
    "synthesizer": {"max_chars": 1500, "max_evidence": 3},
}

# --- Streamlit UI ---
st.set_page_config(page_title="Deep Researcher Agent", layout="wide")
st.title("🔎 Deep Researcher Agent")
st.write("Search, analyze, and synthesize from your local research papers.")

query = st.text_input("Enter your research query:")
run_button = st.button("Run Research")

if run_button and query:
    with st.spinner("Ingesting documents..."):
        loader = DocumentLoader(DATA_DIR)
        docs = loader.load()
        st.success(f"Loaded {len(docs)} document chunks.")

    with st.spinner("Embedding documents..."):
        embedder = EmbeddingModel()
        docs = embedder.embed(docs)
        st.success("Documents embedded.")

    with st.spinner("Indexing with FAISS..."):
        store = VectorStore(cfg={})
        store.build_index(docs)
        st.success("Index built.")

    with st.spinner("Decomposing and reasoning..."):
        reasoner = Reasoner(store, embedder, cfg["reasoning"])
        subtasks = reasoner.decompose(query)
        results = reasoner.solve(subtasks)
        st.success("Reasoning completed.")

    with st.spinner("Synthesizing final answer..."):
        synthesizer = Synthesizer(cfg["synthesizer"])
        final_answer = synthesizer.combine(results, query)

    # Display results
    st.subheader("📄 Final Answer")
    st.text_area("Generated Answer", value=final_answer, height=300)

    st.subheader("🧩 Subtasks & Evidence")
    for i, r in enumerate(results, start=1):
        with st.expander(f"Subtask {i}: {r['subtask']}"):
            st.markdown(f"**Answer (generated):** {r['answer']}")
            st.markdown("**Evidence:**")
            for ev in r.get("evidence", [])[:3]:
                st.markdown(
                    f"- `{ev.get('meta', {}).get('source','unknown')}` "
                    f"(score={ev.get('score',0.0):.3f})"
                )
                st.caption(ev.get("text", "")[:300] + " ...")

    # Export
    with st.spinner("Exporting results..."):
        exporter = Exporter(EXPORT_DIR)
        export_path = exporter.export(query, final_answer, results)
    st.success(f"Results exported to: {export_path}")

    with open(export_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Export",
            data=f,
            file_name=os.path.basename(export_path),
            mime="text/plain",
        )
