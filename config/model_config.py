
from pathlib import Path

class ModelConfig:
    """Central configuration for all models"""
    
    # Base directories
    BASE_DIR = Path(__file__).parent.parent
    MODEL_DIR = BASE_DIR / "models"
    CACHE_DIR = BASE_DIR / "cache"
    
    # Legal-BERT Configuration (for clause extraction)
    LEGAL_BERT = {
        "model_name": "nlpaueb/legal-bert-base-uncased",
        "task": "clause-extraction",
        "max_length": 512,
        "batch_size": 16,
        "local_path": MODEL_DIR / "legal-bert"
    }
    
    # Embedding Model for Semantic Search
    EMBEDDING_MODEL = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "local_path": MODEL_DIR / "embeddings"
    }
    
    # LLM for Analysis (Ollama)
    LLM_CONFIG = {
        "base_url": "http://localhost:11434",
        "model": "mistral:7b",
        "temperature": 0.1,
        "max_tokens": 2500,
        "timeout": 120
    }

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)