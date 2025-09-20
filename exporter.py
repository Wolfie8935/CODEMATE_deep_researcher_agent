"""
Comprehensive export system supporting multiple formats including PDF, Markdown, JSON, and HTML.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# PDF generation
try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from utils import ensure_directory, sanitize_filename, format_timestamp

logger = logging.getLogger(__name__)


class Exporter:
    """
    Export research results to Markdown (or PDF) in a folder.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize exporter with configuration.
        
        :param config: Configuration dictionary
        """
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'exports')
        self.formats = self.config.get('formats', ['markdown'])
        self.include_metadata = self.config.get('include_metadata', True)
        self.include_citations = self.config.get('include_citations', True)
        self.include_reasoning_trace = self.config.get('include_reasoning_trace', True)
        
        # Ensure output directory exists
        ensure_directory(self.output_dir)
        
        # Generate unique filename base
        timestamp = format_timestamp()
        self.filename_base = f"research_output_{timestamp}"
        
        logger.info(f"Exporter initialized. Output formats: {self.formats}")
        logger.info(f"Output directory: {self.output_dir}")

    def export(self, query: str, final_answer: str, results: List[Dict]) -> str:
        """
        Export research results in all configured formats.
        
        :param query: Original research query
        :param final_answer: Final synthesized answer
        :param results: List of research results
        :return: Path to the primary export file (Markdown)
        """
        exported_files = []
        
        # Export in each requested format
        for format_type in self.formats:
            try:
                if format_type == 'markdown':
                    file_path = self._export_markdown(query, final_answer, results)
                elif format_type == 'pdf':
                    file_path = self._export_pdf(query, final_answer, results)
                elif format_type == 'json':
                    file_path = self._export_json(query, final_answer, results)
                elif format_type == 'html':
                    file_path = self._export_html(query, final_answer, results)
                else:
                    logger.warning(f"Unsupported export format: {format_type}")
                    continue
                
                exported_files.append(file_path)
                logger.info(f"Exported {format_type.upper()} format: {file_path}")
                
            except Exception as e:
                logger.error(f"Error exporting {format_type} format: {e}")
                continue
        
        # Return the primary export file (Markdown)
        primary_file = next((f for f in exported_files if f.endswith('.md')), exported_files[0] if exported_files else None)
        return primary_file

    def _export_markdown(self, query: str, final_answer: str, results: List[Dict]) -> str:
        """Export results to Markdown format."""
        filename = f"{self.filename_base}.md"
        file_path = os.path.join(self.output_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            # Header
            f.write(f"# Research Report\n\n")
            f.write(f"**Query:** {query}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(final_answer)
            f.write("\n\n")
            
            # Detailed Analysis
            f.write("## Detailed Analysis\n\n")
            for i, result in enumerate(results, 1):
                f.write(f"### {i}. {result.get('subtask', 'Unknown subtask')}\n\n")
                
                # Answer
                answer = result.get('answer', 'No answer generated')
                confidence = result.get('confidence', 0)
                f.write(f"**Answer:** {answer}\n\n")
                f.write(f"**Confidence:** {confidence:.2f}\n\n")
                
                # Evidence
                evidence = result.get('evidence', [])
                if evidence and self.include_citations:
                    f.write("**Evidence:**\n\n")
                    for j, ev in enumerate(evidence, 1):
                        source = ev.get('meta', {}).get('source', 'unknown')
                        score = ev.get('score', 0)
                        text = ev.get('text', '')[:500] + "..." if len(ev.get('text', '')) > 500 else ev.get('text', '')
                        f.write(f"{j}. **{source}** (relevance: {score:.3f})\n")
                        f.write(f"   {text}\n\n")
                
                f.write("---\n\n")
        
        return file_path

    def _export_json(self, query: str, final_answer: str, results: List[Dict]) -> str:
        """Export results to JSON format."""
        filename = f"{self.filename_base}.json"
        file_path = os.path.join(self.output_dir, filename)
        
        export_data = {
            "metadata": {
                "query": query,
                "export_date": datetime.now().isoformat(),
                "total_subtasks": len(results),
                "total_evidence": sum(len(r.get('evidence', [])) for r in results),
                "average_confidence": sum(r.get('confidence', 0) for r in results) / len(results) if results else 0
            },
            "final_answer": final_answer,
            "results": results
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return file_path

    def _export_pdf(self, query: str, final_answer: str, results: List[Dict]) -> str:
        """Export results to PDF format."""
        if not PDF_AVAILABLE:
            raise ImportError("PDF export requires reportlab. Install with: pip install reportlab")
        
        filename = f"{self.filename_base}.pdf"
        file_path = os.path.join(self.output_dir, filename)
        
        # Set up PDF document
        doc = SimpleDocTemplate(file_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("Research Report", styles['Title']))
        story.append(Paragraph(f"Query: {query}", styles['Normal']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        story.append(Paragraph(final_answer, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Detailed Analysis
        story.append(Paragraph("Detailed Analysis", styles['Heading2']))
        
        for i, result in enumerate(results, 1):
            subtask = result.get('subtask', 'Unknown subtask')
            answer = result.get('answer', 'No answer generated')
            confidence = result.get('confidence', 0)
            
            story.append(Paragraph(f"{i}. {subtask}", styles['Heading3']))
            story.append(Paragraph(f"Answer: {answer}", styles['Normal']))
            story.append(Paragraph(f"Confidence: {confidence:.2f}", styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        return file_path

    def _export_html(self, query: str, final_answer: str, results: List[Dict]) -> str:
        """Export results to HTML format."""
        filename = f"{self.filename_base}.html"
        file_path = os.path.join(self.output_dir, filename)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        h3 {{ color: #888; }}
        .confidence {{ font-weight: bold; }}
        .confidence.high {{ color: green; }}
        .confidence.medium {{ color: orange; }}
        .confidence.low {{ color: red; }}
        .evidence {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-left: 4px solid #007acc; }}
    </style>
</head>
<body>
    <h1>Research Report</h1>
    <p><strong>Query:</strong> {query}</p>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Executive Summary</h2>
    <div>{final_answer.replace(chr(10), '<br>')}</div>
    
    <h2>Detailed Analysis</h2>
"""
        
        for i, result in enumerate(results, 1):
            subtask = result.get('subtask', 'Unknown subtask')
            answer = result.get('answer', 'No answer generated')
            confidence = result.get('confidence', 0)
            
            # Determine confidence class
            if confidence > 0.7:
                conf_class = "high"
            elif confidence > 0.4:
                conf_class = "medium"
            else:
                conf_class = "low"
            
            html_content += f"""
    <h3>{i}. {subtask}</h3>
    <p><strong>Answer:</strong> {answer}</p>
    <p><strong>Confidence:</strong> <span class="confidence {conf_class}">{confidence:.2f}</span></p>
"""
            
            # Evidence
            evidence = result.get('evidence', [])
            if evidence and self.include_citations:
                html_content += "<h4>Evidence:</h4>"
                for j, ev in enumerate(evidence, 1):
                    source = ev.get('meta', {}).get('source', 'unknown')
                    score = ev.get('score', 0)
                    text = ev.get('text', '')[:300] + "..." if len(ev.get('text', '')) > 300 else ev.get('text', '')
                    html_content += f"""
    <div class="evidence">
        <strong>{j}. {source}</strong> (relevance: {score:.3f})<br>
        {text}
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return file_path
