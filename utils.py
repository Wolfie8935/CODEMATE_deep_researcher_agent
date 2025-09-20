# Helper functions (logging, configs, etc.)
import os
import re
import json
import yaml
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(path: str = "config.yml") -> dict:
    if not os.path.exists(path):
        logging.warning(f"Config {path} not found — returning empty config")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Chunk text into overlapping windows trying to respect sentence boundaries."""
    text = clean_text(text)
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    cur = []
    cur_len = 0

    for s in sentences:
        s_len = len(s.split())
        if cur_len + s_len <= chunk_size or not cur:
            cur.append(s)
            cur_len += s_len
        else:
            chunks.append(" ".join(cur))
            # start new chunk with overlap
            overlap_sentences = []
            if overlap > 0:
                # take last sentences that approximately sum to overlap
                rev = cur[::-1]
                acc = 0
                for rs in rev:
                    acc += len(rs.split())
                    if acc <= overlap:
                        overlap_sentences.insert(0, rs)
                    else:
                        break
            cur = overlap_sentences + [s]
            cur_len = sum(len(x.split()) for x in cur)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
