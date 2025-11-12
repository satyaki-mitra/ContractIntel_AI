# DEPENDENCIES
from pathlib import Path


class ModelConfig:
    """
    Model-specific configurations - FOR AI MODEL SETTINGS ONLY
    """
    # Directory Settings
    MODEL_DIR = Path("models")
    CACHE_DIR = Path("cache/models")
    
    # Model Architecture Settings
    LEGAL_BERT        = {"model_name"      : "nlpaueb/legal-bert-base-uncased",
                         "local_path"      : MODEL_DIR / "legal-bert",
                         "task"            : "clause-extraction",
                         "max_length"      : 512,
                         "batch_size"      : 16,
                         "hidden_dim"      : 768,
                         "num_layers"      : 12,
                         "attention_heads" : 12,
                        }
    
    # Embedding Model Settings
    EMBEDDING_MODEL   = {"model_name"           : "sentence-transformers/all-MiniLM-L6-v2",
                         "local_path"           : MODEL_DIR / "embeddings",
                         "dimension"            : 384,
                         "pooling"              : "mean",
                         "normalize"            : True,
                         "similarity_threshold" : 0.7,
                        }
    
    # Classification Model Settings
    CLASSIFIER_MODEL  = {"embedding_dim"    : 384,
                         "hidden_dim"       : 256,
                         "num_categories"   : 12,
                         "dropout_rate"     : 0.1,
                         "learning_rate"    : 2e-5,
                         "max_seq_length"   : 512,
                        }
    
    # Clause Extraction Settings
    CLAUSE_EXTRACTION = {"min_clause_length"    : 50,
                         "max_clause_length"    : 2000,
                         "confidence_threshold" : 0.7,
                         "overlap_threshold"    : 0.3,
                         "max_clauses_per_doc"  : 50,
                        }
    
    # Risk Analysis Settings
    RISK_ANALYSIS     = {"score_ranges"     : {"low"      : (0, 40),
                                               "medium"   : (40, 60),
                                               "high"     : (60, 80),
                                               "critical" : (80, 100),
                                              },
                         "weight_decay"     : 0.1,
                         "smoothing_factor" : 0.5,
                        }
    
    # Market Comparison Settings
    MARKET_COMPARISON = {"similarity_threshold" : 0.75,
                         "min_matches_required" : 3,
                         "max_comparisons"      : 20,
                         "embedding_cache_size" : 1000,
                        }
    
    # LLM Generation Settings
    LLM_GENERATION    = {"max_tokens"        : 5000,
                         "temperature"       : 0.1,
                         "top_p"             : 0.9,
                         "frequency_penalty" : 0.1,
                         "presence_penalty"  : 0.1,
                         "stop_sequences"    : ["\n\n", "###", "---"],
                        }
    
    # Text Processing Settings
    TEXT_PROCESSING   = {"chunk_size"          : 512,
                         "chunk_overlap"       : 50,
                         "min_sentence_length" : 10,
                         "max_sentence_length" : 200,
                         "entity_confidence"   : 0.8,
                        }

    @classmethod
    def ensure_directories(cls):
        """
        Ensure all required directories exist
        """
        directories = [cls.MODEL_DIR,
                       cls.CACHE_DIR,
                       cls.MODEL_DIR / "legal-bert",
                       cls.MODEL_DIR / "embeddings",
                      ]
                    
        for directory in directories:
            directory.mkdir(parents = True, exist_ok = True)


    @classmethod
    def get_model_config(cls, model_type: str) -> dict:
        """
        Get configuration for specific model type
        """
        config_map = {"legal_bert"        : cls.LEGAL_BERT,
                      "embedding"         : cls.EMBEDDING_MODEL,
                      "classifier"        : cls.CLASSIFIER_MODEL,
                      "clause_extraction" : cls.CLAUSE_EXTRACTION,
                      "risk_analysis"     : cls.RISK_ANALYSIS,
                      "market_comparison" : cls.MARKET_COMPARISON,
                      "llm_generation"    : cls.LLM_GENERATION,
                      "text_processing"   : cls.TEXT_PROCESSING,
                     }
        
        return config_map.get(model_type, {})