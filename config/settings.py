# DEPENDENCIES
import os
from pathlib import Path
from pydantic import Field
from typing import Literal
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application-wide settings: primary configuration source
    """
    # Application Info
    APP_NAME               : str                                                = "AI Contract Risk Analyzer"
    APP_VERSION            : str                                                = "1.0.0"
    API_PREFIX             : str                                                = "/api/v1/"
    
    # Server Configuration
    HOST                   : str                                                = "0.0.0.0"
    PORT                   : int                                                = 8000
    RELOAD                 : bool                                               = True
    WORKERS                : int                                                = 1
    
    # CORS Settings
    CORS_ORIGINS           : list                                               = ["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"]
    CORS_ALLOW_CREDENTIALS : bool                                               = True
    CORS_ALLOW_METHODS     : list                                               = ["*"]
    CORS_ALLOW_HEADERS     : list                                               = ["*"]
    
    # File Upload Settings
    MAX_UPLOAD_SIZE        : int                                                = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS     : list                                               = [".pdf", ".docx", ".txt"]
    UPLOAD_DIR             : Path                                               = Path("uploads")
    
    # Model Management Settings
    MODEL_CACHE_SIZE       : int                                                = 3     # Number of models to keep in memory
    MODEL_DOWNLOAD_TIMEOUT : int                                                = 1800  # 30 minutes
    USE_GPU                : bool                                               = True  # Automatically detect and use GPU if available

    # Environment Detection Settings
    IS_HUGGINGFACE_SPACE   : bool                                               = False  # Auto-detected
    IS_LOCAL               : bool                                               = True   # Auto-detected
    DEPLOYMENT_ENV         : Literal["local", "huggingface", "docker", "cloud"] = "local"

    # LLAMA.CPP Settings (For HF Spaces)
    LLAMA_CPP_ENABLED      : bool                                               = False  # Auto-enabled in HF Spaces
    LLAMA_CPP_MODEL_PATH   : Optional[Path]                                     = None   # Local path to GGUF model
    LLAMA_CPP_MODEL_REPO   : str                                                = "NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF"
    LLAMA_CPP_MODEL_FILE   : str                                                = "Hermes-2-Pro-Llama-3-8B-GGUF.Q4_K_M.gguf"
    LLAMA_CPP_N_CTX        : int                                                = 4096   # Context window
    LLAMA_CPP_N_GPU_LAYERS : int                                                = -1     # -1 = all layers on GPU
    LLAMA_CPP_N_BATCH      : int                                                = 512    # Batch size for prompt processing
    LLAMA_CPP_N_THREADS    : int                                                = 4      # CPU threads (0 = auto)
    
    # Ollama Settings (For Local)
    OLLAMA_BASE_URL        : str                                                = "http://localhost:11434"
    OLLAMA_MODEL           : str                                                = "llama3:8b"
    OLLAMA_TIMEOUT         : int                                                = 300
    OLLAMA_TEMPERATURE     : float                                              = 0.1

    # External API Settings
    OPENAI_API_KEY         : Optional[str]                                      = None
    OPENAI_MODEL           : str                                                = "gpt-3.5-turbo"
    OPENAI_TIMEOUT         : int                                                = 30
    OPENAI_TEMPERATURE     : float                                              = 0.1
    OPENAI_MAX_TOKENS      : int                                                = 1024
    
    ANTHROPIC_API_KEY      : Optional[str]                                      = None
    ANTHROPIC_MODEL        : str                                                = "claude-3-haiku-20240307"
    ANTHROPIC_TIMEOUT      : int                                                = 30
    ANTHROPIC_TEMPERATURE  : float                                              = 0.1
    ANTHROPIC_MAX_TOKENS   : int                                                = 1024

    # Priority order for LLM providers
    LLM_PROVIDER_PRIORITY  : list                                               = ["llama_cpp", "ollama", "openai", "anthropic", ]
    
    # Which providers are available
    ENABLE_OLLAMA          : bool                                               = True
    ENABLE_LLAMA_CPP       : bool                                               = False  # Auto-enabled in HF Spaces
    ENABLE_OPENAI          : bool                                               = False
    ENABLE_ANTHROPIC       : bool                                               = False
    ENABLE_HF_INFERENCE    : bool                                               = False  # HuggingFace Inference API

    # Default provider (auto-selected based on environment)
    LLM_DEFAULT_PROVIDER   : str                                                = "llama_cpp"
    
    # Huggingface Inference Settings (Optional)
    HF_MODEL_ID            : Optional[str]                                      = None   # e.g. "meta-llama/Llama-2-7b-chat-hf"
    HF_API_TOKEN           : Optional[str]                                      = None   # HF token for gated models
    
    # LLM Generation Settings (Shared across providers)
    LLM_TEMPERATURE        : float                                              = 0.1      # Default for all providers
    LLM_MAX_TOKENS         : int                                                = 1024     # Default for all providers
    LLM_TOP_P              : float                                              = 0.95     # Default top-p sampling
    LLM_REPEAT_PENALTY     : float                                              = 1.1      # Default repeat penalty
    LLM_SYSTEM_PROMPT      : str                                                = "You are a helpful legal assistant specializing in contract analysis and risk assessment."
    
    # Analysis Limits
    MIN_CONTRACT_LENGTH    : int                                                = 300    # Minimum characters for valid contract
    MAX_CONTRACT_LENGTH    : int                                                = 500000 # Maximum characters (500KB text)
    MAX_CLAUSES_TO_ANALYZE : int                                                = 100
    
    # Logging Settings
    LOG_LEVEL              : str                                                = "INFO"
    LOG_FORMAT             : str                                                = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE               : Optional[Path]                                     = Path("logs/app.log")
    
    # Cache Settings
    ENABLE_CACHE           : bool                                               = True
    CACHE_TTL              : int                                                = 3600 # 1 hour
    CACHE_DIR              : Path                                               = Path("cache")
    
    # Model Cache Directory (for llama.cpp models)
    MODEL_CACHE_DIR        : Path                                               = Path("data/models")
    
    # Rate Limiting Settings
    RATE_LIMIT_ENABLED     : bool                                               = False
    RATE_LIMIT_REQUESTS    : int                                                = 10
    RATE_LIMIT_PERIOD      : int                                                = 60  # seconds
    

    class Config:
        env_file          = ".env"
        env_file_encoding = "utf-8"
        case_sensitive    = True
    

    @field_validator('IS_HUGGINGFACE_SPACE', 'IS_LOCAL', 'DEPLOYMENT_ENV', mode = 'before')
    def detect_environment(cls, v, info):
        """
        Auto-detect deployment environment
        """
        field_name = info.field_name
        
        if (field_name == 'IS_HUGGINGFACE_SPACE'):
            return bool(os.getenv('SPACE_ID'))
        
        elif (field_name == 'IS_LOCAL'):
            # Check if not in any container/cloud environment
            return not any([os.getenv('SPACE_ID'),
                            os.getenv('DOCKER_CONTAINER'),
                            os.getenv('KUBERNETES_SERVICE_HOST'),
                            os.getenv('AWS_EXECUTION_ENV')
                          ])
        
        elif (field_name == 'DEPLOYMENT_ENV'):
            if os.getenv('SPACE_ID'):
                return "huggingface"
            
            elif os.getenv('DOCKER_CONTAINER'):
                return "docker"
            
            elif os.getenv('KUBERNETES_SERVICE_HOST'):
                return "kubernetes"
            
            elif os.getenv('AWS_EXECUTION_ENV'):
                return "aws"
            
            else:
                return "local"
        
        return v
    

    @field_validator('ENABLE_LLAMA_CPP', 'LLAMA_CPP_ENABLED', mode = 'after')
    def enable_llama_cpp_for_hf(cls, v, info):
        """
        Auto-enable llama.cpp for HuggingFace Spaces
        """
        values = info.data
        
        if values.get('IS_HUGGINGFACE_SPACE'):
            return True

        return v
    

    @field_validator('ENABLE_OLLAMA', mode = 'after')
    def disable_ollama_for_hf(cls, v, info):
        """
        Auto-disable Ollama for HuggingFace Spaces
        """
        values = info.data
        
        if values.get('IS_HUGGINGFACE_SPACE'):
            return False

        return v

    
    @field_validator('LLM_PROVIDER_PRIORITY', mode='after')
    def adjust_provider_priority(cls, v, info):
        """
        Adjust provider priority based on environment
        """
        values = info.data
        
        if values.get('IS_HUGGINGFACE_SPACE'):
            # For HF Spaces: llama_cpp first, then external APIs
            priority = []
            
            if (values.get('ENABLE_LLAMA_CPP')):
                priority.append("llama_cpp")

            if (values.get('ENABLE_HF_INFERENCE') and values.get('HF_API_TOKEN')):
                priority.append("hf_inference")

            if (values.get('ENABLE_OPENAI') and values.get('OPENAI_API_KEY')):
                priority.append("openai")

            if (values.get('ENABLE_ANTHROPIC') and values.get('ANTHROPIC_API_KEY')):
                priority.append("anthropic")
            
            return priority if priority else ["llama_cpp"]
        
        else:
            # For local: Ollama first
            priority = list()
            
            if values.get('ENABLE_OLLAMA'):
                priority.append("ollama")

            if values.get('ENABLE_LLAMA_CPP'):
                priority.append("llama_cpp")
            
            if values.get('ENABLE_OPENAI') and values.get('OPENAI_API_KEY'):
                priority.append("openai")
            
            if values.get('ENABLE_ANTHROPIC') and values.get('ANTHROPIC_API_KEY'):
                priority.append("anthropic")
            
            return priority if priority else ["ollama"]
    

    @field_validator('LLM_DEFAULT_PROVIDER', mode = 'after')
    def set_default_provider(cls, v, info):
        """
        Set default provider based on availability
        """
        values   = info.data
        
        # Get the priority list (after adjustments)
        priority = values.get('LLM_PROVIDER_PRIORITY', [])
        
        if priority:
            # First available provider is default
            return priority[0]  
        
        # Fallback
        return "ollama"  
    

    @field_validator('MODEL_CACHE_DIR')
    def set_model_cache_dir(cls, v, info):
        """
        Set appropriate model cache directory based on environment
        """
        values = info.data
        
        if (values.get('IS_HUGGINGFACE_SPACE')):
            # HF Spaces have persistent /data directory
            return Path("/data/models")

        elif (values.get('DEPLOYMENT_ENV') == "docker"):
            # Docker containers
            return Path("/app/models")

        else:
            # Local development
            return Path("models")
    

    @field_validator('LLAMA_CPP_N_GPU_LAYERS')
    def optimize_gpu_layers(cls, v, info):
        """
        Auto-optimize GPU layers for different environments
        """
        values = info.data
        
        if values.get('IS_HUGGINGFACE_SPACE'):
            # HF Spaces: T4 GPU with 15-16GB VRAM
            # For 8B Q4 model: ~20 layers is safe
            return 20

        elif v == -1:  # -1 means "use all layers"
            # For local with sufficient GPU
            return -1

        else:
            # Explicit value from config
            return v

    
    @field_validator('LLAMA_CPP_MODEL_PATH')
    def set_default_model_path(cls, v, info):
        """
        Set default model path if not specified
        """
        values = info.data
        
        if v is None and values.get('LLAMA_CPP_MODEL_FILE'):
            cache_dir = values.get('MODEL_CACHE_DIR', Path("models"))
            return cache_dir / values['LLAMA_CPP_MODEL_FILE']
        
        return v


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure Directories Exist
        self.UPLOAD_DIR.mkdir(parents = True, exist_ok = True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok = True)
        self.MODEL_CACHE_DIR.mkdir(parents = True, exist_ok = True)
        
        if self.LOG_FILE:
            self.LOG_FILE.parent.mkdir(parents = True, exist_ok = True)
        


# Global settings instance
settings = Settings()