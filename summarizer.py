# (Optional) Summarization of multiple sources

"""Simple extractive summarizer using TF-IDF sentence scoring.
"""
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def summarize_docs(docs: List[Dict], max_sentences: int = 5) -> str:
    texts = [d['text'] for d in docs]
    if not texts:
        return ""
    # split into sentences
    sents = []
    import re
    for t in texts:
        parts = re.split(r'(?<=[.?!])\s+', t)
        for p in parts:
            if p.strip():
                sents.append(p.strip())
    if len(sents) == 0:
        return ""
    vec = TfidfVectorizer(stop_words='english').fit_transform(sents)
    scores = np.asarray(vec.sum(axis=1)).ravel()
    top_idx = np.argsort(-scores)[:max_sentences]
    summary = " ".join([sents[i] for i in sorted(top_idx)])
    return summary
