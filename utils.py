# Helper functions (logging, configs, etc.)
import os
import re
import json
import yaml
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import time
from datetime import datetime

# Configure logging
def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration based on config file."""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO').upper())
    log_file = log_config.get('log_file', 'logs/researcher.log')
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(path: str = "config.yml") -> dict:
    """Load configuration from YAML file with validation."""
    if not os.path.exists(path):
        logging.warning(f"Config {path} not found — creating default config")
        create_default_config(path)
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # Setup logging based on config
        setup_logging(config)
        
        # Validate and set defaults
        config = validate_and_set_defaults(config)
        
        logging.info(f"Configuration loaded successfully from {path}")
        return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return get_default_config()

def create_default_config(path: str) -> None:
    """Create a default configuration file."""
    default_config = get_default_config()
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)

def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'ingestion': {
            'data_dir': 'data',
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'supported_formats': ['.pdf', '.docx', '.txt', '.md'],
            'max_file_size_mb': 50
        },
        'embeddings': {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'batch_size': 32,
            'cache_embeddings': True
        },
        'vector_store': {
            'index_type': 'faiss',
            'search_top_k': 10,
            'persist_index': True
        },
        'reasoning': {
            'top_k': 5,
            'summarizer_model': 'facebook/bart-large-cnn',
            'summarizer_max_input_chars': 3000
        },
        'synthesizer': {
            'max_chars': 2000,
            'max_evidence': 5
        },
        'export': {
            'output_dir': 'exports',
            'formats': ['markdown', 'pdf']
        },
        'logging': {
            'level': 'INFO',
            'log_file': 'logs/researcher.log'
        }
    }

def validate_and_set_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration and set missing defaults."""
    default_config = get_default_config()
    
    def merge_configs(base: Dict, override: Dict) -> Dict:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    return merge_configs(default_config, config)

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

def get_file_hash(file_path: str) -> str:
    """Generate MD5 hash of a file for caching purposes."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logging.error(f"Error generating hash for {file_path}: {e}")
        return ""

def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage."""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    return filename

def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """Format timestamp for consistent use across the application."""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y%m%d_%H%M%S")

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using Jaccard similarity."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0.0

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis."""
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    # Simple word frequency analysis
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_freq = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]

def validate_query(query: str, max_length: int = 1000) -> tuple[bool, str]:
    """Validate user query for security and length."""
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    if len(query) > max_length:
        return False, f"Query too long (max {max_length} characters)"
    
    # Check for potentially malicious patterns
    malicious_patterns = [
        r'<script.*?>',
        r'javascript:',
        r'data:',
        r'vbscript:',
        r'onload=',
        r'onerror='
    ]
    
    for pattern in malicious_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False, "Query contains potentially malicious content"
    
    return True, ""

def create_cache_key(*args) -> str:
    """Create a cache key from multiple arguments."""
    key_string = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()

def measure_time(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"
