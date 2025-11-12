# DEPENDENCIES
from pathlib import Path
from pydantic import Field
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application-wide settings: primary configuration source
    """
    # Application Info
    APP_NAME               : str           = "AI Contract Risk Analyzer"
    APP_VERSION            : str           = "1.0.0"
    API_PREFIX             : str           = "/api/v1/"
    
    # Server Configuration
    HOST                   : str           = "0.0.0.0"
    PORT                   : int           = 8000
    RELOAD                 : bool          = True
    WORKERS                : int           = 1
    
    # CORS Settings
    CORS_ORIGINS           : list          = ["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"]
    CORS_ALLOW_CREDENTIALS : bool          = True
    CORS_ALLOW_METHODS     : list          = ["*"]
    CORS_ALLOW_HEADERS     : list          = ["*"]
    
    # File Upload Settings
    MAX_UPLOAD_SIZE        : int           = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS     : list          = [".pdf", ".docx", ".txt"]
    UPLOAD_DIR             : Path          = Path("uploads")
    
    # Model Management Settings
    MODEL_CACHE_SIZE       : int           = 3     # Number of models to keep in memory
    MODEL_DOWNLOAD_TIMEOUT : int           = 1800  # 30 minutes
    USE_GPU                : bool          = True  # Automatically detect and use GPU if available
    
    # External API Settings
    OLLAMA_BASE_URL        : str           = "http://localhost:11434"
    OLLAMA_MODEL           : str           = "llama3:8b"
    OLLAMA_TIMEOUT         : int           = 300
    OLLAMA_TEMPERATURE     : float         = 0.1
    
    # External API Keys
    OPENAI_API_KEY         : Optional[str] = None
    ANTHROPIC_API_KEY      : Optional[str] = None
    
    # Analysis Limits
    MIN_CONTRACT_LENGTH    : int           = 300    # Minimum characters for valid contract
    MAX_CONTRACT_LENGTH    : int           = 500000 # Maximum characters (500KB text)
    MAX_CLAUSES_TO_ANALYZE : int           = 15
    
    # Logging Settings
    LOG_LEVEL              : str           = "INFO"
    LOG_FORMAT             : str           = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE               : Optional[Path] = Path("logs/app.log")
    
    # Cache Settings
    ENABLE_CACHE           : bool          = True
    CACHE_TTL              : int           = 3600 # 1 hour
    CACHE_DIR              : Path          = Path("cache")
    
    # Rate Limiting Settings
    RATE_LIMIT_ENABLED     : bool          = True
    RATE_LIMIT_REQUESTS    : int           = 10
    RATE_LIMIT_PERIOD      : int           = 60  # seconds
    
    # PDF Report Settings
    PDF_FONT_SIZE          : int           = 10
    PDF_MARGIN             : float         = 0.5 # inches
    PDF_PAGE_SIZE          : str           = "letter"
    

    class Config:
        env_file          = ".env"
        env_file_encoding = "utf-8"
        case_sensitive    = True
    

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.UPLOAD_DIR.mkdir(parents = True, exist_ok = True)
        self.CACHE_DIR.mkdir(parents = True, exist_ok = True)
        
        if self.LOG_FILE:
            self.LOG_FILE.parent.mkdir(parents = True, exist_ok = True)


# Global settings instance
settings = Settings()