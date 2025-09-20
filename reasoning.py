"""
Advanced multi-step reasoner with intelligent query decomposition, evidence retrieval,
and natural-language answer generation using local models.

Features:
- Intelligent query decomposition using multiple strategies
- Advanced evidence retrieval with reranking
- Abstractive summarization with fallback mechanisms
- Confidence scoring and source verification
- Multi-step reasoning with LLM integration (optional)
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import math
import logging
import numpy as np
from collections import defaultdict

# transformers for local summarization and reasoning
from transformers import pipeline, Pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

from utils import calculate_similarity, extract_keywords, validate_query

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Reasoner:
    """
    Advanced reasoning engine with intelligent query decomposition and multi-step reasoning.
    """
    
    def __init__(self, store, embedder, config: Dict[str, Any]):
        """
        Initialize the reasoning engine.
        
        :param store: VectorStore instance
        :param embedder: EmbeddingModel instance
        :param config: Configuration dictionary
        """
        self.store = store
        self.embedder = embedder
        self.config = config or {}
        
        # Query decomposition settings
        self.decomposition_strategy = self.config.get('decomposition_strategy', 'intelligent')
        self.max_subtasks = self.config.get('max_subtasks', 5)
        self.subtask_overlap_threshold = self.config.get('subtask_overlap_threshold', 0.3)
        
        # Evidence retrieval settings
        self.top_k = int(self.config.get("top_k", 5))
        self.rerank_results = self.config.get('rerank_results', True)
        self.evidence_diversity_threshold = self.config.get('evidence_diversity_threshold', 0.8)
        
        # Summarization settings
        self.summarizer_model_name = self.config.get("summarizer_model", "facebook/bart-large-cnn")
        self.max_input_chars = int(self.config.get("summarizer_max_input_chars", 3000))
        self.min_length = int(self.config.get("summarizer_min_length", 30))
        self.max_length = int(self.config.get("summarizer_max_length", 200))
        self.enable_abstractive_summarization = self.config.get('enable_abstractive_summarization', True)
        self.fallback_to_extractive = self.config.get('fallback_to_extractive', True)
        
        # LLM integration settings
        self.enable_llm_reasoning = self.config.get('enable_llm_reasoning', False)
        self.llm_model = self.config.get('llm_model', 'microsoft/DialoGPT-medium')
        self.llm_max_tokens = self.config.get('llm_max_tokens', 500)
        self.llm_temperature = self.config.get('llm_temperature', 0.7)
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"Reasoner initialized with strategy: {self.decomposition_strategy}")
        logger.info(f"LLM reasoning enabled: {self.enable_llm_reasoning}")

    def _initialize_models(self):
        """Initialize summarization and LLM models."""
        # Initialize summarization pipeline
        self.summarizer: Optional[Pipeline] = None
        if self.enable_abstractive_summarization:
            try:
                device = 0 if torch.cuda.is_available() else -1
                logger.info(f"Loading summarization model {self.summarizer_model_name} (device={device})")
                self.summarizer = pipeline(
                    "summarization",
                    model=self.summarizer_model_name,
                    device=device,
                )
            except Exception as e:
                logger.warning(f"Could not load summarization model '{self.summarizer_model_name}': {e}")
                self.summarizer = None
        
        # Initialize LLM for reasoning (optional)
        self.llm_tokenizer = None
        self.llm_model_obj = None
        if self.enable_llm_reasoning:
            try:
                logger.info(f"Loading LLM model {self.llm_model}")
                self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
                self.llm_model_obj = AutoModelForCausalLM.from_pretrained(self.llm_model)
                
                # Set pad token if not present
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                    
            except Exception as e:
                logger.warning(f"Could not load LLM model '{self.llm_model}': {e}")
                self.llm_tokenizer = None
                self.llm_model_obj = None

    def decompose(self, query: str) -> List[str]:
        """
        Decompose query into subtasks using the configured strategy.
        
        :param query: Original query string
        :return: List of subtask strings
        """
        # Validate query
        is_valid, error_msg = validate_query(query)
        if not is_valid:
            logger.warning(f"Query validation failed: {error_msg}")
            return [query]
        
        if self.decomposition_strategy == 'simple':
            return self._simple_decomposition(query)
        elif self.decomposition_strategy == 'intelligent':
            return self._intelligent_decomposition(query)
        elif self.decomposition_strategy == 'llm_based':
            return self._llm_based_decomposition(query)
        else:
            logger.warning(f"Unknown decomposition strategy: {self.decomposition_strategy}")
            return self._simple_decomposition(query)

    def _simple_decomposition(self, query: str) -> List[str]:
        """Simple decomposition based on delimiters."""
        parts = re.split(r"\band\b|,|;|\n", query)
        parts = [p.strip() for p in parts if p.strip()]
        return parts if parts else [query]

    def _intelligent_decomposition(self, query: str) -> List[str]:
        """Intelligent decomposition using keyword analysis and semantic understanding."""
        # Extract keywords from the query
        keywords = extract_keywords(query, max_keywords=10)
        
        # Identify question words and action verbs
        question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which'}
        action_verbs = {'analyze', 'compare', 'explain', 'describe', 'evaluate', 'identify', 'summarize'}
        
        # Check if query contains multiple concepts
        sentences = re.split(r'[.!?]+', query)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            # Single sentence - try to break down by conjunctions and question words
            subtasks = self._break_down_single_query(query, keywords, question_words, action_verbs)
        else:
            # Multiple sentences - each could be a subtask
            subtasks = sentences
        
        # Remove duplicates and similar subtasks
        subtasks = self._deduplicate_subtasks(subtasks)
        
        # Limit number of subtasks
        if len(subtasks) > self.max_subtasks:
            subtasks = subtasks[:self.max_subtasks]
        
        return subtasks if subtasks else [query]

    def _break_down_single_query(self, query: str, keywords: List[str], 
                                question_words: set, action_verbs: set) -> List[str]:
        """Break down a single query into subtasks."""
        subtasks = []
        
        # Check for conjunctions that indicate multiple parts
        conjunction_patterns = [
            r'\band\b',
            r'\bor\b',
            r'\bbut\b',
            r'\bhowever\b',
            r'\bmoreover\b',
            r'\bfurthermore\b'
        ]
        
        for pattern in conjunction_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                parts = re.split(pattern, query, flags=re.IGNORECASE)
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) > 1:
                    subtasks.extend(parts)
                    break
        
        # If no conjunctions found, try to create subtasks based on keywords
        if not subtasks:
            # Group related keywords
            keyword_groups = self._group_related_keywords(keywords)
            
            if len(keyword_groups) > 1:
                for group in keyword_groups:
                    # Create a subtask for each keyword group
                    subtask = f"What is the relationship between {', '.join(group)} in the context of the query?"
                    subtasks.append(subtask)
        
        return subtasks

    def _group_related_keywords(self, keywords: List[str]) -> List[List[str]]:
        """Group related keywords together."""
        if len(keywords) <= 2:
            return [keywords]
        
        groups = []
        used_keywords = set()
        
        for keyword in keywords:
            if keyword in used_keywords:
                continue
            
            group = [keyword]
            used_keywords.add(keyword)
            
            # Find related keywords (simple similarity-based grouping)
            for other_keyword in keywords:
                if other_keyword in used_keywords:
                    continue
                
                similarity = calculate_similarity(keyword, other_keyword)
                if similarity > 0.3:  # Threshold for relatedness
                    group.append(other_keyword)
                    used_keywords.add(other_keyword)
            
            groups.append(group)
        
        return groups

    def _deduplicate_subtasks(self, subtasks: List[str]) -> List[str]:
        """Remove duplicate and overly similar subtasks."""
        if len(subtasks) <= 1:
            return subtasks
        
        unique_subtasks = []
        for subtask in subtasks:
            is_duplicate = False
            for existing in unique_subtasks:
                similarity = calculate_similarity(subtask, existing)
                if similarity > self.subtask_overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_subtasks.append(subtask)
        
        return unique_subtasks

    def _llm_based_decomposition(self, query: str) -> List[str]:
        """Decompose query using LLM reasoning."""
        if not self.enable_llm_reasoning or not self.llm_tokenizer or not self.llm_model_obj:
            logger.warning("LLM-based decomposition requested but LLM not available")
            return self._intelligent_decomposition(query)
        
        try:
            # Create prompt for query decomposition
            prompt = f"""Break down the following research query into {self.max_subtasks} specific subtasks:

Query: {query}

Subtasks:
1."""
            
            # Generate response using LLM
            inputs = self.llm_tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.llm_model_obj.generate(
                    inputs,
                    max_length=inputs.shape[1] + self.llm_max_tokens,
                    temperature=self.llm_temperature,
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id
                )
            
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract subtasks from response
            subtasks = self._extract_subtasks_from_llm_response(response, query)
            
            return subtasks if subtasks else self._intelligent_decomposition(query)
            
        except Exception as e:
            logger.error(f"Error in LLM-based decomposition: {e}")
            return self._intelligent_decomposition(query)

    def _extract_subtasks_from_llm_response(self, response: str, original_query: str) -> List[str]:
        """Extract subtasks from LLM response."""
        # Look for numbered list items
        pattern = r'\d+\.\s*([^\n]+)'
        matches = re.findall(pattern, response)
        
        if matches:
            subtasks = [match.strip() for match in matches if match.strip()]
            # Filter out subtasks that are too similar to the original query
            filtered_subtasks = []
            for subtask in subtasks:
                similarity = calculate_similarity(subtask, original_query)
                if similarity < 0.8:  # Not too similar to original
                    filtered_subtasks.append(subtask)
            return filtered_subtasks
        
        return []

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
        Solve subtasks by retrieving evidence and generating answers.
        
        :param subtasks: List of subtask strings
        :return: List of result dictionaries with subtask, answer, evidence, and metadata
        """
        results = []
        
        for i, subtask in enumerate(subtasks):
            logger.info(f"Processing subtask {i+1}/{len(subtasks)}: {subtask[:100]}...")
            
            try:
                # Generate embedding for subtask
                query_vector = self.embedder.embed_query(subtask)
                
                # Retrieve evidence with advanced search
                evidence = self._retrieve_evidence(subtask, query_vector)
                
                # Generate answer using evidence
                answer, confidence = self._generate_answer(subtask, evidence)
                
                # Create result dictionary
                result = {
                    "subtask": subtask,
                    "answer": answer,
                    "evidence": evidence,
                    "confidence": confidence,
                    "subtask_index": i,
                    "total_subtasks": len(subtasks),
                    "metadata": {
                        "evidence_count": len(evidence),
                        "avg_evidence_score": np.mean([e.get('score', 0) for e in evidence]) if evidence else 0,
                        "answer_length": len(answer),
                        "has_evidence": len(evidence) > 0
                    }
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing subtask '{subtask}': {e}")
                # Create error result
                error_result = {
                    "subtask": subtask,
                    "answer": f"Error processing subtask: {str(e)}",
                    "evidence": [],
                    "confidence": 0.0,
                    "subtask_index": i,
                    "total_subtasks": len(subtasks),
                    "metadata": {
                        "error": str(e),
                        "evidence_count": 0,
                        "avg_evidence_score": 0,
                        "answer_length": 0,
                        "has_evidence": False
                    }
                }
                results.append(error_result)
        
        logger.info(f"Completed processing {len(subtasks)} subtasks")
        return results

    def _retrieve_evidence(self, subtask: str, query_vector: np.ndarray) -> List[Dict]:
        """Retrieve evidence for a subtask with advanced search and reranking."""
        try:
            # Perform initial search
            if hasattr(self.store, 'hybrid_search'):
                # Use hybrid search if available
                evidence = self.store.hybrid_search(
                    query_vector, 
                    subtask, 
                    top_k=self.top_k * 2,  # Get more results for reranking
                    alpha=0.7
                )
            else:
                # Use regular search
                evidence = self.store.search(query_vector, top_k=self.top_k * 2)
            
            # Rerank results if enabled
            if self.rerank_results and evidence:
                evidence = self._rerank_evidence(subtask, evidence)
            
            # Ensure diversity in results
            evidence = self._ensure_evidence_diversity(evidence)
            
            # Limit to top_k results
            evidence = evidence[:self.top_k]
            
            return evidence
            
        except Exception as e:
            logger.error(f"Error retrieving evidence for subtask '{subtask}': {e}")
            return []

    def _rerank_evidence(self, subtask: str, evidence: List[Dict]) -> List[Dict]:
        """Rerank evidence based on relevance to subtask."""
        if not evidence:
            return evidence
        
        # Extract keywords from subtask
        subtask_keywords = set(extract_keywords(subtask, max_keywords=10))
        
        # Calculate reranking scores
        reranked_evidence = []
        for doc in evidence:
            # Original similarity score
            original_score = doc.get('score', 0)
            
            # Keyword overlap score
            doc_keywords = set([k.lower() for k in doc['meta'].get('keywords', [])])
            doc_text_words = set(doc['text'].lower().split())
            
            keyword_overlap = len(subtask_keywords.intersection(doc_keywords))
            text_overlap = len(subtask_keywords.intersection(doc_text_words))
            
            keyword_score = (keyword_overlap * 2 + text_overlap) / (len(subtask_keywords) + 1)
            
            # Combined reranking score
            rerank_score = 0.7 * original_score + 0.3 * keyword_score
            
            doc['rerank_score'] = rerank_score
            reranked_evidence.append(doc)
        
        # Sort by reranking score
        reranked_evidence.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked_evidence

    def _ensure_evidence_diversity(self, evidence: List[Dict]) -> List[Dict]:
        """Ensure diversity in evidence sources."""
        if not evidence:
            return evidence
        
        diverse_evidence = []
        used_sources = set()
        
        for doc in evidence:
            source = doc['meta'].get('source', 'unknown')
            
            # Check if we already have evidence from this source
            if source in used_sources:
                # Only add if it's significantly different from existing evidence
                is_diverse = True
                for existing_doc in diverse_evidence:
                    if existing_doc['meta'].get('source') == source:
                        similarity = calculate_similarity(doc['text'], existing_doc['text'])
                        if similarity > self.evidence_diversity_threshold:
                            is_diverse = False
                            break
                
                if not is_diverse:
                    continue
            
            diverse_evidence.append(doc)
            used_sources.add(source)
        
        return diverse_evidence

    def _generate_answer(self, subtask: str, evidence: List[Dict]) -> Tuple[str, float]:
        """Generate answer from evidence with confidence scoring."""
        if not evidence:
            return "No relevant evidence found for this subtask.", 0.0
        
        try:
            # Generate answer using summarization
            answer = self._smart_summarize(evidence)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(subtask, answer, evidence)
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Error generating answer for subtask '{subtask}': {e}")
            # Fallback to extractive answer
            if evidence:
                fallback_answer = evidence[0].get('text', '')[:500] + "..."
                return fallback_answer, 0.3
            else:
                return "Unable to generate answer.", 0.0

    def _calculate_confidence(self, subtask: str, answer: str, evidence: List[Dict]) -> float:
        """Calculate confidence score for the generated answer."""
        if not evidence or not answer:
            return 0.0
        
        # Base confidence from evidence quality
        avg_evidence_score = np.mean([e.get('score', 0) for e in evidence])
        evidence_confidence = min(avg_evidence_score, 1.0)
        
        # Answer quality indicators
        answer_length = len(answer)
        has_evidence = len(evidence) > 0
        
        # Length-based confidence (not too short, not too long)
        if 50 <= answer_length <= 1000:
            length_confidence = 1.0
        elif answer_length < 50:
            length_confidence = answer_length / 50
        else:
            length_confidence = max(0.5, 1000 / answer_length)
        
        # Evidence count confidence
        evidence_count_confidence = min(len(evidence) / 3, 1.0)  # Optimal at 3+ evidence pieces
        
        # Combined confidence
        confidence = (
            0.4 * evidence_confidence +
            0.3 * length_confidence +
            0.3 * evidence_count_confidence
        )
        
        return min(confidence, 1.0)
