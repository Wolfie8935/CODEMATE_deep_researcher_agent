"""
Multi-step reasoner that retrieves evidence and produces natural-language
answers by summarizing retrieved chunks with a local Hugging Face model.

Behavior:
- Decomposes the user query into subtasks (naive splitting).
- For each subtask, retrieves top-k chunks from the vector store.
- Attempts abstractive summarization of the combined top-k chunks using
  a local HF summarization model. If the input is too long or the model
  errors, falls back to summarizing each chunk separately then combining.
- Returns a list of dicts: {'subtask','answer','evidence'}.
"""

from typing import List, Dict
import re
import math
import logging

# transformers for local summarization
from transformers import pipeline, Pipeline
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Reasoner:
    def __init__(self, store, embedder, cfg: Dict):
        """
        :param store: VectorStore instance (must implement search(query_vector, top_k))
        :param embedder: EmbeddingModel instance (must implement embed_query)
        :param cfg: configuration dict; expected keys:
            - top_k (int)
            - summarizer_model (str) optional, e.g. "facebook/bart-large-cnn"
            - summarizer_max_input_chars (int) optional, e.g. 3000
            - summarizer_min_length, summarizer_max_length (ints)
        """
        self.store = store
        self.embedder = embedder
        self.cfg = cfg or {}
        self.top_k = int(self.cfg.get("top_k", 5))

        # summarizer config
        self.summarizer_model_name = self.cfg.get(
            "summarizer_model", "facebook/bart-large-cnn"
        )
        # max input chars before chunking (very conservative)
        self.max_input_chars = int(self.cfg.get("summarizer_max_input_chars", 3000))
        self.min_length = int(self.cfg.get("summarizer_min_length", 30))
        self.max_length = int(self.cfg.get("summarizer_max_length", 200))

        # initialize summarization pipeline (attempt)
        self.summarizer: Pipeline = None
        try:
            device = 0 if torch.cuda.is_available() else -1
            logger.info(
                f"Loading summarization model {self.summarizer_model_name} (device={device})"
            )
            self.summarizer = pipeline(
                "summarization",
                model=self.summarizer_model_name,
                device=device,
            )
        except Exception as e:
            logger.warning(
                f"Could not load summarization model '{self.summarizer_model_name}': {e}. "
                "Falling back to extractive behavior."
            )
            self.summarizer = None

    def decompose(self, query: str) -> List[str]:
        """
        Naive decomposition: split on 'and', ',', ';' and newline.
        This is deliberately simple — replace with a more advanced planner later.
        """
        parts = re.split(r"\band\b|,|;|\n", query)
        parts = [p.strip() for p in parts if p.strip()]
        return parts if parts else [query]

    def _concat_hits_text(self, hits: List[Dict]) -> str:
        """
        Concatenate hit texts into a single context string.
        """
        return "\n\n".join(h.get("text", "") for h in hits)

    def _summarize_text(self, text: str) -> str:
        """
        Summarize a single text (string) using the summarizer pipeline.
        Returns the summary or raises an exception if summarizer isn't available.
        """
        if not self.summarizer:
            raise RuntimeError("No summarizer pipeline available")
        # pipeline will manage truncation internally, but we prefer to control length
        # call summarizer with min_length/max_length taken from config
        out = self.summarizer(
            text,
            max_length=self.max_length,
            min_length=self.min_length,
            truncation=True,
        )
        # pipeline returns list of dicts with 'summary_text'
        if isinstance(out, list) and len(out) > 0 and "summary_text" in out[0]:
            return out[0]["summary_text"].strip()
        # fallback: join available text
        if isinstance(out, str):
            return out.strip()
        raise RuntimeError("Unexpected summarizer output")

    def _smart_summarize(self, hits: List[Dict]) -> str:
        """
        Summarize a list of retrieved hits robustly:
         - If the combined text is within max_input_chars, summarizer is called once.
         - Otherwise, summarize each hit individually then combine those summaries
           and run a final summarization pass (if summarizer available).
         - If summarizer not available, return a short extractive concatenation.
        """
        combined = self._concat_hits_text(hits)
        if not combined:
            return ""

        # if no summarizer is available, fallback to extractive combine (trim)
        if not self.summarizer:
            # simple heuristic: return first hit text (shortened)
            first = hits[0]["text"] if len(hits) > 0 else ""
            return first[: self.max_input_chars].strip() + (
                "..." if len(first) > self.max_input_chars else ""
            )

        # if combined is small, summarize directly
        if len(combined) <= self.max_input_chars:
            try:
                return self._summarize_text(combined)
            except Exception as e:
                logger.warning(f"Summarizer failed on combined text: {e}")

        # otherwise summarize each hit individually (robust to long docs)
        partial_summaries = []
        for h in hits:
            txt = h.get("text", "")
            if not txt:
                continue
            try:
                # if the chunk itself is long, allow truncation
                summary = self._summarize_text(txt)
                partial_summaries.append(summary)
            except Exception as e:
                # fallback to trimmed extract if summarizer fails on that chunk
                logger.warning(f"Chunk summarization failed: {e}")
                trimmed = txt[: self.max_input_chars].strip()
                partial_summaries.append(trimmed + ("..." if len(txt) > len(trimmed) else ""))

        # combine partial summaries
        combined_partial = "\n\n".join(partial_summaries)
        # final summarization pass if short enough, else return combined_partial trimmed
        try:
            if len(combined_partial) <= self.max_input_chars:
                return self._summarize_text(combined_partial)
            else:
                # attempt a final summarization with truncation
                return self._summarize_text(combined_partial[: self.max_input_chars])
        except Exception as e:
            logger.warning(f"Final summarization pass failed: {e}")
            return combined_partial[: self.max_input_chars].strip()

    def solve(self, subtasks: List[str]) -> List[Dict]:
        """
        For each subtask, embed the subtask, retrieve evidence from the store,
        and produce an 'answer' using summarization over the evidence.
        Returns: list of {'subtask','answer','evidence'}
        """
        results = []
        for s in subtasks:
            # embed the subtask text
            try:
                qv = self.embedder.embed_query(s)
            except Exception as e:
                logger.error(f"Embedding query failed for subtask '{s}': {e}")
                qv = None

            # retrieve top-k evidence
            hits = []
            try:
                if qv is not None:
                    hits = self.store.search(qv, top_k=self.top_k)
                else:
                    # if embedding failed, attempt a blind search (not available in this store)
                    hits = []
            except Exception as e:
                logger.warning(f"Search failed for subtask '{s}': {e}")
                hits = []

            # produce a summarized answer based on hits
            answer_text = ""
            try:
                if hits:
                    answer_text = self._smart_summarize(hits)
                else:
                    answer_text = ""  # no evidence found
            except Exception as e:
                logger.warning(f"Summarization error for subtask '{s}': {e}")
                # fallback to first hit extract if exists
                if hits:
                    answer_text = hits[0].get("text", "")
                else:
                    answer_text = ""

            results.append({"subtask": s, "answer": answer_text, "evidence": hits})
        return results
