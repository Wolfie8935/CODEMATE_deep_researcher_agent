# Entry point (CLI or web UI to run queries)

import argparse
from utils import load_config
from ingestion import ingest_documents
from embeddings import EmbeddingModel
from vector_store import VectorStore
from reasoning import Reasoner
from synthesizer import Synthesizer
from exporter import Exporter

def run(query: str, config_path: str = "config.yml"):
    cfg = load_config(config_path)

    print("Ingesting documents...")
    docs = ingest_documents(cfg.get('ingestion', {}))
    print(f"Got {len(docs)} chunks")

    print("Embedding documents...")
    embedder = EmbeddingModel(cfg.get('embeddings', {}))
    docs = embedder.embed_documents(docs)

    print("Indexing with FAISS...")
    store = VectorStore(cfg.get('vector_store', {}))
    store.build_index(docs)

    print("Decomposing and reasoning...")
    reasoner = Reasoner(store, embedder, cfg.get('reasoning', {}))
    subtasks = reasoner.decompose(query)
    results = reasoner.solve(subtasks)

    print("Synthesizing answer...")
    synth = Synthesizer(cfg.get('synthesizer', {}))
    answer = synth.combine(results, query)

    print('\n===== FINAL ANSWER =====\n')
    print(answer)

    print('\nExporting results...')
    Exporter(cfg.get('export', {})).export(answer, results)
    print('Export complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('query', type=str, help='Question to research')
    parser.add_argument('--config', type=str, default='config.yml')
    args = parser.parse_args()
    run(args.query, args.config)
