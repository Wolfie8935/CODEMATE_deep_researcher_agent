import os
from datetime import datetime

class Exporter:
    """
    Export research results to Markdown (or PDF) in a folder.
    """

    def __init__(self, export_dir: str):
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)
        # generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_path = os.path.join(self.export_dir, f"research_output_{timestamp}.md")

    def export(self, query: str, final_answer: str, results: list) -> str:
        """
        Export results to a Markdown file.
        Returns the file path.
        """
        with open(self.out_path, "w", encoding="utf-8") as f:
            f.write(f"# Research Query: {query}\n\n")
            f.write(f"## Final Answer\n{final_answer}\n\n")
            f.write("## Subtasks & Evidence\n")
            for i, r in enumerate(results, start=1):
                f.write(f"\n### Subtask {i}: {r['subtask']}\n")
                f.write(f"**Answer (generated):** {r['answer']}\n")
                f.write("**Evidence:**\n")
                for ev in r.get("evidence", []):
                    source = ev.get("meta", {}).get("source", "unknown") if isinstance(ev, dict) else "unknown"
                    text = ev.get("text", "")[:300] if isinstance(ev, dict) else str(ev)
                    score = ev.get("score", 0.0) if isinstance(ev, dict) else 0.0
                    f.write(f"- `{source}` (score={score:.3f}): {text} ...\n")
            f.write("\n---\nResearch trace included in export.\n")
        return self.out_path
