"""
Combine retrieved results into a coherent final answer with citations,
and provide a short, human-friendly explanation.

This synthesizer expects each result to be:
    {'subtask': str, 'answer': str, 'evidence': List[{'id','meta','text','score'}]}

Output: single string suitable for printing / export.
"""
from typing import List, Dict
import textwrap


class Synthesizer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg or {}
        # max characters per extract shown in final answer
        self.max_chars = int(self.cfg.get("max_chars", 1500))
        # number of evidence items to show per subtask
        self.max_evidence = int(self.cfg.get("max_evidence", 3))

    def _format_evidence_ref(self, ev: Dict) -> str:
        src = ev.get("meta", {}).get("source", ev.get("id", "unknown"))
        score = ev.get("score", 0.0)
        return f"{src} (score={score:.3f})"

    def _shorten(self, text: str, max_chars: int) -> str:
        if text is None:
            return ""
        t = text.strip()
        if len(t) <= max_chars:
            return t
        return t[: max_chars - 3].rstrip() + "..."

    def combine(self, results: List[Dict], query: str) -> str:
        """
        Produce a final answer string:
         - A short declarative summary synthesizing subtask answers.
         - For each subtask: list the abstractive answer (from reasoning) and citations.
         - A short trace note pointing to the evidence file paths.
        """
        pieces = []
        pieces.append(f"Query: {query}\n")

        # Build a short synthetic summary by concatenating subtask answers
        synthetic_summary_parts = []
        for r in results:
            if r.get("answer"):
                synthetic_summary_parts.append(r["answer"])

        if synthetic_summary_parts:
            # try to keep summary concise
            summary = " ".join(synthetic_summary_parts)
            summary = self._shorten(summary, self.max_chars)
            pieces.append("Summary:\n")
            # wrap text for readability
            pieces.append(textwrap.fill(summary, width=100) + "\n")
        else:
            pieces.append("Summary: No concise answer could be generated from the indexed sources.\n")

        # Detailed per-subtask answers with citations
        pieces.append("\nDetails:\n")
        for i, r in enumerate(results, start=1):
            pieces.append(f"{i}. Subtask: {r['subtask']}\n")
            ans = r.get("answer", "")
            if ans:
                pieces.append("   Answer (generated):\n")
                pieces.append(textwrap.fill(self._shorten(ans, self.max_chars), width=100) + "\n")
            else:
                pieces.append("   Answer: (no generated answer)\n")

            # citations / evidence
            ev = r.get("evidence", []) or []
            if ev:
                pieces.append("   Citations:\n")
                for e in ev[: self.max_evidence]:
                    pieces.append("    - " + self._format_evidence_ref(e) + "\n")
            else:
                pieces.append("   Citations: (none)\n")
            pieces.append("\n")

        pieces.append("---\nResearch trace included in export.\n")
        return "".join(pieces)
