"""
Advanced synthesis engine that combines retrieved results into coherent final answers
with citation tracking, confidence scoring, and source verification.

Features:
- Citation tracking and formatting
- Confidence scoring and source verification
- Contradiction detection
- Multi-source synthesis
- Structured output formatting
"""

from typing import List, Dict, Any, Optional, Tuple
import textwrap
import re
import logging
from collections import defaultdict, Counter
from utils import calculate_similarity, extract_keywords

logger = logging.getLogger(__name__)


class Synthesizer:
    """
    Advanced synthesis engine for combining research results.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize synthesizer with configuration.
        
        :param config: Configuration dictionary
        """
        self.config = config or {}
        
        # Output formatting settings
        self.max_chars = int(self.config.get("max_chars", 2000))
        self.max_evidence = int(self.config.get("max_evidence", 5))
        
        # Citation settings
        self.enable_citation_tracking = self.config.get('enable_citation_tracking', True)
        self.citation_format = self.config.get('citation_format', 'apa')  # apa, mla, chicago, simple
        
        # Quality assessment settings
        self.confidence_scoring = self.config.get('confidence_scoring', True)
        self.source_verification = self.config.get('source_verification', True)
        self.enable_contradiction_detection = self.config.get('enable_contradiction_detection', True)
        
        # Citation tracking
        self.citation_counter = 0
        self.citations = {}
        self.source_credibility = {}
        
        logger.info(f"Synthesizer initialized with citation format: {self.citation_format}")
        logger.info(f"Contradiction detection: {self.enable_contradiction_detection}")

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
        Combine research results into a comprehensive final answer.
        
        :param results: List of research result dictionaries
        :param query: Original research query
        :return: Formatted final answer string
        """
        # Reset citation tracking for new synthesis
        self.citation_counter = 0
        self.citations = {}
        self.source_credibility = {}
        
        # Analyze results for contradictions and quality
        analysis = self._analyze_results(results)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(results, query, analysis)
        
        # Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(results, analysis)
        
        # Generate confidence assessment
        confidence_assessment = self._generate_confidence_assessment(results, analysis)
        
        # Generate source verification report
        source_report = self._generate_source_report(results) if self.source_verification else ""
        
        # Combine all sections
        final_answer = self._format_final_answer(
            query, executive_summary, detailed_analysis, 
            confidence_assessment, source_report, analysis
        )
        
        return final_answer

    def _analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze results for contradictions, quality, and patterns."""
        analysis = {
            'total_subtasks': len(results),
            'total_evidence': sum(len(r.get('evidence', [])) for r in results),
            'avg_confidence': sum(r.get('confidence', 0) for r in results) / len(results) if results else 0,
            'contradictions': [],
            'source_diversity': set(),
            'quality_issues': [],
            'key_findings': []
        }
        
        # Collect all evidence for analysis
        all_evidence = []
        for result in results:
            all_evidence.extend(result.get('evidence', []))
            analysis['source_diversity'].update([e.get('meta', {}).get('source', 'unknown') for e in result.get('evidence', [])])
        
        # Detect contradictions if enabled
        if self.enable_contradiction_detection:
            analysis['contradictions'] = self._detect_contradictions(results)
        
        # Assess source credibility
        if self.source_verification:
            analysis['source_credibility'] = self._assess_source_credibility(all_evidence)
        
        # Extract key findings
        analysis['key_findings'] = self._extract_key_findings(results)
        
        return analysis

    def _detect_contradictions(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """Detect contradictions between different results."""
        contradictions = []
        
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results[i+1:], i+1):
                # Compare answers for contradictions
                answer1 = result1.get('answer', '').lower()
                answer2 = result2.get('answer', '').lower()
                
                # Simple contradiction detection based on keywords
                contradiction_keywords = [
                    ('increase', 'decrease'), ('rise', 'fall'), ('positive', 'negative'),
                    ('support', 'oppose'), ('agree', 'disagree'), ('prove', 'disprove'),
                    ('confirm', 'refute'), ('valid', 'invalid'), ('effective', 'ineffective')
                ]
                
                for pos_word, neg_word in contradiction_keywords:
                    if (pos_word in answer1 and neg_word in answer2) or (neg_word in answer1 and pos_word in answer2):
                        contradictions.append({
                            'type': 'semantic_contradiction',
                            'subtask1': result1.get('subtask', ''),
                            'subtask2': result2.get('subtask', ''),
                            'keywords': (pos_word, neg_word),
                            'confidence': 0.7
                        })
        
        return contradictions

    def _assess_source_credibility(self, evidence: List[Dict]) -> Dict[str, float]:
        """Assess credibility of sources based on metadata and patterns."""
        source_scores = defaultdict(list)
        
        for ev in evidence:
            source = ev.get('meta', {}).get('source', 'unknown')
            score = ev.get('score', 0)
            source_scores[source].append(score)
        
        credibility = {}
        for source, scores in source_scores.items():
            # Calculate credibility based on average score and consistency
            avg_score = sum(scores) / len(scores)
            consistency = 1.0 - (max(scores) - min(scores)) if len(scores) > 1 else 1.0
            credibility[source] = (avg_score + consistency) / 2
        
        return credibility

    def _extract_key_findings(self, results: List[Dict]) -> List[str]:
        """Extract key findings from results."""
        findings = []
        
        for result in results:
            answer = result.get('answer', '')
            confidence = result.get('confidence', 0)
            
            # Only include high-confidence findings
            if confidence > 0.6 and len(answer) > 50:
                # Extract key sentences (simple heuristic)
                sentences = re.split(r'[.!?]+', answer)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20 and any(word in sentence.lower() for word in ['find', 'show', 'indicate', 'suggest', 'demonstrate']):
                        findings.append(sentence)
        
        return findings[:5]  # Limit to top 5 findings

    def _generate_executive_summary(self, results: List[Dict], query: str, analysis: Dict[str, Any]) -> str:
        """Generate executive summary of findings."""
        summary_parts = []
        
        # Overall assessment
        if analysis['avg_confidence'] > 0.7:
            summary_parts.append("Based on the analysis of available sources, the research provides strong evidence for the following findings:")
        elif analysis['avg_confidence'] > 0.4:
            summary_parts.append("The research reveals several findings with moderate confidence:")
        else:
            summary_parts.append("The research provides limited evidence, with the following tentative findings:")
        
        # Key findings
        if analysis['key_findings']:
            summary_parts.append("\nKey Findings:")
            for i, finding in enumerate(analysis['key_findings'], 1):
                summary_parts.append(f"{i}. {finding}")
        
        # Contradictions warning
        if analysis['contradictions']:
            summary_parts.append(f"\n⚠️  Note: {len(analysis['contradictions'])} potential contradictions were detected between different sources.")
        
        return "\n".join(summary_parts)

    def _generate_detailed_analysis(self, results: List[Dict], analysis: Dict[str, Any]) -> str:
        """Generate detailed analysis section."""
        sections = []
        
        sections.append("DETAILED ANALYSIS")
        sections.append("=" * 50)
        
        for i, result in enumerate(results, 1):
            subtask = result.get('subtask', '')
            answer = result.get('answer', '')
            confidence = result.get('confidence', 0)
            evidence = result.get('evidence', [])
            
            sections.append(f"\n{i}. {subtask}")
            sections.append("-" * 40)
            
            if answer:
                # Add confidence indicator
                confidence_indicator = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
                sections.append(f"Confidence: {confidence_indicator} {confidence:.2f}")
                sections.append(f"\nAnswer:\n{textwrap.fill(answer, width=80)}")
            else:
                sections.append("No answer generated for this subtask.")
            
            # Citations
            if evidence and self.enable_citation_tracking:
                sections.append(f"\nSources ({len(evidence)}):")
                for j, ev in enumerate(evidence[:self.max_evidence], 1):
                    citation = self._format_citation(ev, j)
                    sections.append(f"  [{j}] {citation}")
        
        return "\n".join(sections)

    def _generate_confidence_assessment(self, results: List[Dict], analysis: Dict[str, Any]) -> str:
        """Generate confidence assessment section."""
        if not self.confidence_scoring:
            return ""
        
        sections = []
        sections.append("\nCONFIDENCE ASSESSMENT")
        sections.append("=" * 50)
        
        # Overall confidence
        overall_confidence = analysis['avg_confidence']
        if overall_confidence > 0.8:
            confidence_level = "High"
            confidence_desc = "Strong evidence supports the findings"
        elif overall_confidence > 0.6:
            confidence_level = "Moderate"
            confidence_desc = "Good evidence with some limitations"
        elif overall_confidence > 0.4:
            confidence_level = "Low"
            confidence_desc = "Limited evidence, findings should be interpreted cautiously"
        else:
            confidence_level = "Very Low"
            confidence_desc = "Insufficient evidence for reliable conclusions"
        
        sections.append(f"Overall Confidence: {confidence_level} ({overall_confidence:.2f})")
        sections.append(f"Assessment: {confidence_desc}")
        
        # Evidence quality
        sections.append(f"\nEvidence Quality:")
        sections.append(f"  • Total evidence pieces: {analysis['total_evidence']}")
        sections.append(f"  • Source diversity: {len(analysis['source_diversity'])} unique sources")
        sections.append(f"  • Subtasks completed: {analysis['total_subtasks']}")
        
        return "\n".join(sections)

    def _generate_source_report(self, results: List[Dict]) -> str:
        """Generate source verification report."""
        if not self.source_verification:
            return ""
        
        sections = []
        sections.append("\nSOURCE VERIFICATION")
        sections.append("=" * 50)
        
        # Collect all sources
        all_sources = set()
        for result in results:
            for ev in result.get('evidence', []):
                source = ev.get('meta', {}).get('source', 'unknown')
                all_sources.add(source)
        
        sections.append(f"Sources analyzed: {len(all_sources)}")
        
        # Source credibility scores
        if hasattr(self, '_source_credibility') and self._source_credibility:
            sections.append("\nSource Credibility Scores:")
            for source, score in sorted(self._source_credibility.items(), key=lambda x: x[1], reverse=True):
                credibility_level = "High" if score > 0.7 else "Moderate" if score > 0.4 else "Low"
                sections.append(f"  • {source}: {credibility_level} ({score:.2f})")
        
        return "\n".join(sections)

    def _format_citation(self, evidence: Dict, citation_number: int) -> str:
        """Format citation according to specified format."""
        meta = evidence.get('meta', {})
        source = meta.get('source', 'unknown')
        score = evidence.get('score', 0)
        
        if self.citation_format == 'simple':
            return f"{source} (relevance: {score:.2f})"
        elif self.citation_format == 'apa':
            return f"{source} (n.d.). Retrieved from local database. Relevance score: {score:.2f}"
        else:
            return f"{source} (score: {score:.2f})"

    def _format_final_answer(self, query: str, executive_summary: str, 
                           detailed_analysis: str, confidence_assessment: str,
                           source_report: str, analysis: Dict[str, Any]) -> str:
        """Format the complete final answer."""
        sections = []
        
        # Header
        sections.append("🔍 DEEP RESEARCH RESULTS")
        sections.append("=" * 60)
        sections.append(f"Query: {query}")
        sections.append(f"Analysis Date: {self._get_timestamp()}")
        sections.append("")
        
        # Executive Summary
        sections.append("📋 EXECUTIVE SUMMARY")
        sections.append("-" * 30)
        sections.append(executive_summary)
        
        # Detailed Analysis
        sections.append(detailed_analysis)
        
        # Confidence Assessment
        if confidence_assessment:
            sections.append(confidence_assessment)
        
        # Source Report
        if source_report:
            sections.append(source_report)
        
        # Footer
        sections.append("\n" + "=" * 60)
        sections.append("Research completed by Deep Researcher Agent")
        sections.append("All sources are from local document collection")
        
        return "\n".join(sections)

    def _get_timestamp(self) -> str:
        """Get current timestamp for reports."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _shorten(self, text: str, max_length: int) -> str:
        """Shorten text to maximum length while preserving meaning."""
        if len(text) <= max_length:
            return text
        
        # Try to break at sentence boundary
        sentences = re.split(r'[.!?]+', text)
        result = ""
        for sentence in sentences:
            if len(result + sentence) <= max_length - 3:
                result += sentence + ". "
            else:
                break
        
        if result:
            return result.strip() + "..."
        else:
            return text[:max_length - 3] + "..."
