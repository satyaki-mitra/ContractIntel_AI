# DEPENDENCIES
import sys
import threading
from enum import Enum
from typing import Any
from typing import Dict
from pathlib import Path
from typing import Optional
from dataclasses import field
from datetime import datetime
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log_info
from utils.logger import log_error
from utils.logger import ContractAnalyzerLogger 


class ModelType(Enum):
    """
    Enum for model types
    """
    LEGAL_BERT = "legal-bert"
    EMBEDDING  = "embedding"
    TOKENIZER  = "tokenizer"


class ModelStatus(Enum):
    """
    Model loading status
    """
    NOT_LOADED = "not_loaded"
    LOADING    = "loading"
    LOADED     = "loaded"
    ERROR      = "error"


@dataclass
class ModelInfo:
    """
    Model metadata and state
    """
    name           : str
    type           : ModelType
    status         : ModelStatus        = ModelStatus.NOT_LOADED
    model          : Optional[Any]      = None
    tokenizer      : Optional[Any]      = None
    loaded_at      : Optional[datetime] = None
    error_message  : Optional[str]      = None
    memory_size_mb : float              = 0.0
    access_count   : int                = 0
    last_accessed  : Optional[datetime] = None
    metadata       : Dict[str, Any]     = field(default_factory = dict)
    

    def mark_accessed(self):
        """
        Update access statistics
        """
        self.access_count += 1
        self.last_accessed = datetime.now()
    

    def get_age_seconds(self) -> float:
        """
        Get seconds since last access
        """
        if self.last_accessed:
            return (datetime.now() - self.last_accessed).total_seconds()
        
        return float('inf')


class ModelRegistry:
    """
    Thread-safe singleton model registry : manages all loaded models with LRU eviction
    """
    _instance = None
    _lock     = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance              = super().__new__(cls)
                    cls._instance._initialized = False
        
        return cls._instance
    

    def __init__(self):
        if self._initialized:
            return
        
        self._registry : Dict[ModelType, ModelInfo] = dict()
        self._model_lock                            = threading.Lock()
        self._max_models                            = 3  # LRU cache size
        self._initialized                           = True
        self.logger                                 = ContractAnalyzerLogger.get_logger()
        
        log_info("ModelRegistry initialized", max_models = self._max_models)
    

    def register(self, model_type: ModelType, model_info: ModelInfo):
        """
        Register a model (thread-safe)
        """
        with self._model_lock:
            self._registry[model_type] = model_info
            self._enforce_cache_limit()
            
            log_info(f"Model registered: {model_type.value}",
                     model_name = model_info.name,
                     status     = model_info.status.value,
                     memory_mb  = model_info.memory_size_mb,
                    )
    

    def get(self, model_type: ModelType) -> Optional[ModelInfo]:
        """
        Get model info (thread-safe)
        """
        with self._model_lock:
            info = self._registry.get(model_type)
            
            if info:
                info.mark_accessed()
                log_info(f"Model accessed: {model_type.value}",
                         access_count = info.access_count,
                         status       = info.status.value,
                        )

            return info
    

    def is_loaded(self, model_type: ModelType) -> bool:
        """
        Check if model is loaded
        """
        info = self.get(model_type)
        return info is not None and (info.status == ModelStatus.LOADED)
    

    def unload(self, model_type: ModelType):
        """
        Unload a model from memory
        """
        with self._model_lock:
            if model_type in self._registry:
                info = self._registry[model_type]
                
                log_info(f"Unloading model: {model_type.value}",
                         memory_freed_mb = info.memory_size_mb,
                         access_count    = info.access_count,
                        )
                
                # Clear references to allow garbage collection
                info.model     = None
                info.tokenizer = None
                info.status    = ModelStatus.NOT_LOADED

                del self._registry[model_type]
    

    def get_all_loaded(self) -> list[ModelInfo]:
        """
        Get all loaded models
        """
        with self._model_lock:
            loaded = [info for info in self._registry.values() if (info.status == ModelStatus.LOADED)]
            
            if loaded:
                log_info(f"Retrieved {len(loaded)} loaded models", models = [info.name for info in loaded])
            
            return loaded
    

    def get_memory_usage(self) -> float:
        """
        Get total memory usage in MB
        """
        with self._model_lock:
            total = sum(info.memory_size_mb for info in self._registry.values() if (info.status == ModelStatus.LOADED))

            return total
    

    def _enforce_cache_limit(self):
        """
        Enforce LRU cache limit
        """
        loaded_models = [(model_type, info) for model_type, info in self._registry.items() if (info.status == ModelStatus.LOADED)]
        
        if (len(loaded_models) > self._max_models):
            # Sort by last access time (oldest first)
            loaded_models.sort(key = lambda x: x[1].get_age_seconds(), reverse = True)
            
            # Unload oldest models
            for model_type, info in loaded_models[self._max_models:]:
                log_info(f"LRU eviction: {model_type.value}",
                         reason      = "cache_limit_exceeded",
                         age_seconds = info.get_age_seconds(),
                         max_models  = self._max_models,
                        )

                self.unload(model_type)
    

    def clear_all(self):
        """
        Clear all models from registry
        """
        with self._model_lock:
            model_count  = len(self._registry)
            total_memory = self.get_memory_usage()
            
            log_info("Clearing all models from registry",
                     models_cleared  = model_count,
                     memory_freed_mb = total_memory,
                    )
            
            for model_type in list(self._registry.keys()):
                self.unload(model_type)
    

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics
        """
        with self._model_lock:
            stats = {"total_models"    : len(self._registry),
                     "loaded_models"   : sum(1 for info in self._registry.values() if info.status == ModelStatus.LOADED),
                     "total_memory_mb" : self.get_memory_usage(),
                     "models"          : {model_type.value: {"status"        : info.status.value,
                                                             "access_count"  : info.access_count,
                                                             "memory_mb"     : info.memory_size_mb,
                                                             "last_accessed" : info.last_accessed.isoformat() if info.last_accessed else None,
                                                            }
                                          for model_type, info in self._registry.items()
                                         },
                    }
            
            log_info("Registry stats retrieved", **stats)
            
            return stats
