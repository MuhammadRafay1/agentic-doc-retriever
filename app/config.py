"""
Configuration module for AI Research Assistant
Contains all configurable settings including embedding models, vector stores, and paths.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
VECTOR_STORE_DIR = PROJECT_ROOT / "Vector_Store"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Create directories if they don't exist
for dir_path in [DATA_DIR, EMBEDDINGS_DIR, VECTOR_STORE_DIR, EXPERIMENTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Supported Hugging Face Embedding Models
# These are sentence-transformers models optimized for semantic search
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Fast and efficient, good for general use"
    },
    "all-mpnet-base-v2": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "dimension": 768,
        "description": "High quality, balanced speed/performance"
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "dimension": 384,
        "description": "Optimized for question-answering"
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "description": "Supports 50+ languages"
    },
    "all-distilroberta-v1": {
        "name": "sentence-transformers/all-distilroberta-v1",
        "dimension": 768,
        "description": "High quality RoBERTa-based model"
    }
}

# Vector Store Options
VECTOR_STORES = {
    "FAISS": "Facebook AI Similarity Search - Fast in-memory search",
    "ChromaDB": "Chroma - Open-source embedding database"
}

# Text Processing Settings
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Search Settings
DEFAULT_TOP_K = 5  # Default number of results to retrieve
MAX_TOP_K = 20  # Maximum retrievable results

# Supported document formats
SUPPORTED_FORMATS = ['.txt', '.pdf', '.docx', '.md']

# GUI Settings
WINDOW_TITLE = "AI Research Assistant - Semantic Search"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
