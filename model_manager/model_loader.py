# DEPENDENCIES
import sys
import torch
from pathlib import Path
from transformers import AutoModel
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log_info
from utils.logger import log_error
from config.model_config import ModelConfig
from utils.logger import ContractAnalyzerLogger
from model_manager.model_registry import ModelInfo
from model_manager.model_registry import ModelType
from model_manager.model_registry import ModelStatus
from model_manager.model_registry import ModelRegistry


class ModelLoader:
    """
    Smart model loader with automatic download, caching, and GPU support
    """
    def __init__(self):
        self.registry = ModelRegistry()
        self.config   = ModelConfig()
        self.logger   = ContractAnalyzerLogger.get_logger()
        
        # Detect device
        self.device   = "cuda" if torch.cuda.is_available() else "cpu"

        log_info(f"ModelLoader initialized", device = self.device, gpu_available = torch.cuda.is_available())
        
        # Ensure directories exist
        ModelConfig.ensure_directories()
        log_info("Model directories ensured", 
                 model_dir = str(self.config.MODEL_DIR),
                 cache_dir = str(self.config.CACHE_DIR),
                )

    
    def load_legal_bert(self) -> tuple:
        """
        Load Legal-BERT model and tokenizer (nlpaueb/legal-bert-base-uncased)
        """
        # Check if already loaded
        if self.registry.is_loaded(ModelType.LEGAL_BERT):
            info = self.registry.get(ModelType.LEGAL_BERT)

            log_info("Legal-BERT already loaded from cache",
                     memory_mb    = info.memory_size_mb,
                     access_count = info.access_count,
                    )

            return info.model, info.tokenizer
        
        # Mark as loading
        self.registry.register(ModelType.LEGAL_BERT, ModelInfo(name   = "legal-bert", 
                                                               type   = ModelType.LEGAL_BERT, 
                                                               status = ModelStatus.LOADING,
                                                              )
                              )
        
        try:
            config = self.config.LEGAL_BERT
            
            # Try loading from local cache first
            if config["local_path"].exists():
                log_info(f"Loading Legal-BERT from local cache", path = str(config["local_path"]))

                model     = AutoModel.from_pretrained(pretrained_model_name_or_path = config["local_path"])
                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = config["local_path"])
            
            else:
                log_info(f"Downloading Legal-BERT from HuggingFace", model_name = config["model_name"])

                model     = AutoModel.from_pretrained(pretrained_model_name_or_path = config["model_name"])
                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = config["model_name"])
                
                # Save to local cache
                log_info(f"Saving Legal-BERT to local cache", path = str(config["local_path"]))

                model.save_pretrained(save_directory = config["local_path"])
                tokenizer.save_pretrained(save_directory = config["local_path"])
            
            # Move to device
            model.to(self.device)
            model.eval()
            
            # Calculate memory size
            memory_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            
            # Register as loaded
            self.registry.register(ModelType.LEGAL_BERT,
                                   ModelInfo(name           = "legal-bert",
                                             type           = ModelType.LEGAL_BERT,
                                             status         = ModelStatus.LOADED,
                                             model          = model,
                                             tokenizer      = tokenizer,
                                             memory_size_mb = memory_mb,
                                             metadata       = {"device"     : self.device, 
                                                               "model_name" : config["model_name"],
                                                              },
                                            ),
                                  )
            
            log_info("Legal-BERT loaded successfully",
                     memory_mb  = round(memory_mb, 2),
                     device     = self.device,
                     parameters = sum(p.numel() for p in model.parameters()),
                    )
            
            return model, tokenizer
            
        except Exception as e:
            log_error(e, context = {"component" : "ModelLoader", "operation" : "load_legal_bert", "model_name" : self.config.LEGAL_BERT["model_name"]})
            
            self.registry.register(ModelType.LEGAL_BERT,
                                   ModelInfo(name          = "legal-bert",
                                             type          = ModelType.LEGAL_BERT,
                                             status        = ModelStatus.ERROR,
                                             error_message = str(e),
                                            )
                                  )
            raise

    
    def load_embedding_model(self) -> SentenceTransformer:
        """
        Load sentence transformer for embeddings
        """
        # Check if already loaded
        if self.registry.is_loaded(ModelType.EMBEDDING):
            info = self.registry.get(ModelType.EMBEDDING)
            
            log_info("Embedding model already loaded from cache",
                     memory_mb    = info.memory_size_mb,
                     access_count = info.access_count,
                    )

            return info.model
        
        # Mark as loading
        self.registry.register(ModelType.EMBEDDING,
                               ModelInfo(name   = "embedding", 
                                         type   = ModelType.EMBEDDING, 
                                         status = ModelStatus.LOADING,
                                        ),
                              )
        
        try:
            config = self.config.EMBEDDING_MODEL
            
            # Load model
            if config["local_path"].exists():
                log_info("Loading embedding model from local cache",
                         path = str(config["local_path"]),
                        )

                model = SentenceTransformer(model_name_or_path = str(config["local_path"]))
            
            else:
                log_info("Downloading embedding model from HuggingFace", model_name = config["model_name"])

                model = SentenceTransformer(model_name_or_path = config["model_name"])
                
                # Save to local cache
                log_info("Saving embedding model to local cache", path = str(config["local_path"]))
                
                model.save(str(config["local_path"]))
            
            # Move to device
            if (self.device == "cuda"):
                model = model.to(self.device)
            
            # Estimate memory size : approximate for sentence transformers
            memory_mb = 100  
            
            # Register as loaded
            self.registry.register(ModelType.EMBEDDING,
                                   ModelInfo(name           = "embedding",
                                             type           = ModelType.EMBEDDING,
                                             status         = ModelStatus.LOADED,
                                             model          = model,
                                             memory_size_mb = memory_mb,
                                             metadata       = {"device"     : self.device,
                                                               "model_name" : config["model_name"],
                                                               "dimension"  : config["dimension"],
                                                              }
                                            ),
                                  )
            
            log_info("Embedding model loaded successfully",
                     memory_mb = memory_mb,
                     device    = self.device,
                     dimension = config["dimension"],
                    )
            
            return model
            
        except Exception as e:
            log_error(e, context = {"component" : "ModelLoader", "operation" : "load_embedding_model", "model_name" : self.config.EMBEDDING_MODEL["model_name"]})
            
            self.registry.register(ModelType.EMBEDDING,
                                   ModelInfo(name          = "embedding",
                                             type          = ModelType.EMBEDDING,
                                             status        = ModelStatus.ERROR,
                                             error_message = str(e),
                                            )
                                  )
            raise

    
    def get_registry_stats(self) -> dict:
        """
        Get statistics about loaded models
        """
        stats = self.registry.get_stats()
        log_info("Retrieved registry statistics",
                 total_models    = stats["total_models"],
                 loaded_models   = stats["loaded_models"],
                 total_memory_mb = stats["total_memory_mb"],
                )

        return stats
    

    def clear_cache(self):
        """
        Clear all models from memory
        """
        log_info("Clearing all models from cache")
        self.registry.clear_all()

        log_info("All models cleared from cache")
