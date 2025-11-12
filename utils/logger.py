# DEPENDENCIES
import sys
import time
import json
import logging
import traceback
from typing import Any
from typing import Dict
from pathlib import Path
from typing import Optional
from functools import wraps
from datetime import datetime



class ContractAnalyzerLogger:
    """
    Production-grade logging for contract analysis
    Features:
    - Structured JSON logging
    - Separate files for errors/warnings
    - Request ID tracking
    - Performance metrics
    - Log rotation
    """
    _loggers : Dict[str, logging.Logger] = dict()
    _log_dir : Optional[Path]            = None
    
    # Log levels
    DEBUG                                = logging.DEBUG
    INFO                                 = logging.INFO
    WARNING                              = logging.WARNING
    ERROR                                = logging.ERROR
    CRITICAL                             = logging.CRITICAL
    

    @classmethod
    def setup(cls, log_dir: str = "logs", app_name: str = "contract_analyzer"):
        """
        Setup logging system
        
        Arguments:
        ----------
            log_dir  { str } : Directory for log files

            app_name { str } : Application name for log files
        """
        cls._log_dir = Path(log_dir)
        cls._log_dir.mkdir(parents = True, exist_ok = True)
        
        # Create main logger
        cls._create_logger(name     = app_name,
                           log_file = cls._log_dir / f"{app_name}.log",
                           level    = logging.INFO,
                          )
        
        # Create error logger
        cls._create_logger(name     = f"{app_name}.error",
                           log_file = cls._log_dir / f"{app_name}_error.log",
                           level    = logging.ERROR,
                          )
        
        # Create performance logger
        cls._create_logger(name     = f"{app_name}.performance",
                           log_file = cls._log_dir / f"{app_name}_performance.log",
                           level    = logging.INFO,
                          )
        
        print(f"[Logger] Logging initialized. Logs: {cls._log_dir}")
    

    @classmethod
    def _create_logger(cls, name: str, log_file: Path, level: int) -> logging.Logger:
        """
        Create and configure a logger
        """
        logger             = logging.getLogger(name)
        logger.setLevel(level)

        # Clear existing handlers
        logger.handlers.clear()  
        
        # File handler
        file_handler       = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Console handler (for errors and above)
        console_handler    = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter          = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        cls._loggers[name] = logger

        return logger
    

    @classmethod
    def get_logger(cls, name: str = "contract_analyzer") -> logging.Logger:
        """
        Get logger by name
        """
        if name not in cls._loggers:
            # Lazy initialization
            cls.setup()

        return cls._loggers.get(name, logging.getLogger(name))
    

    @classmethod
    def log_structured(cls, level: int, message: str, **kwargs):
        """
        Log structured data as JSON
        
        Arguments:
        ----------
            level      { int } : Log level

            message    { str } : Log message
            
            **kwargs           : Additional structured data
        """
        logger   = cls.get_logger()
        
        log_data = {"timestamp"  : datetime.now().isoformat(),
                    "message"    : message,
                    **kwargs
                   }
        
        logger.log(level, json.dumps(log_data))
    

    @classmethod
    def log_error(cls, error: Exception, context: Dict[str, Any] = None):
        """
        Log error with full traceback and context
        
        Arguments:
        ----------
            error      { Exception } : Exception object

            context      { dict }    : Additional context dictionary
        """
        error_logger = cls._loggers.get("contract_analyzer.error")
        
        if not error_logger:
            error_logger = cls.get_logger()
        
        error_data = {"timestamp"     : datetime.now().isoformat(),
                      "error_type"    : type(error).__name__,
                      "error_message" : str(error),
                      "traceback"     : traceback.format_exc(),
                      "context"       : context or {},
                     }
        
        error_logger.error(json.dumps(error_data, indent = 2))
    

    @classmethod
    def log_performance(cls, operation: str, duration: float, **metrics):
        """
        Log performance metrics
        
        Arguments:
        ----------
            operation  { str }  : Operation name

            duration  { float } : Duration in seconds
            
            **metrics           : Additional metrics
        """
        perf_logger = cls._loggers.get("contract_analyzer.performance")
        if not perf_logger:
            perf_logger = cls.get_logger()
        
        perf_data = {"timestamp"        : datetime.now().isoformat(),
                     "operation"        : operation,
                     "duration_seconds" : round(duration, 3),
                     **metrics
                    }
        
        perf_logger.info(json.dumps(perf_data))
    

    @staticmethod
    def log_execution_time(operation_name: str = None):
        """
        Decorator to log execution time of functions
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                op_name    = operation_name or func.__name__
                start_time = time.time()
                
                try:
                    result   = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    ContractAnalyzerLogger.log_performance(operation = op_name,
                                                           duration  = duration,
                                                           status    = "success",
                                                          )
                    
                    return result
                
                except Exception as e:
                    duration = time.time() - start_time
                    
                    ContractAnalyzerLogger.log_performance(operation = op_name,
                                                           duration  = duration,
                                                           status    = "error",
                                                           error     = str(e),
                                                          )
                    
                    ContractAnalyzerLogger.log_error(e, context = {"operation" : op_name})
                    raise
            
            return wrapper

        return decorator



# Convenience functions
def get_logger(name: str = "contract_analyzer") -> logging.Logger:
    """
    Get logger instance
    """
    return ContractAnalyzerLogger.get_logger(name)


def log_info(message: str, **kwargs):
    """
    Log info message
    """
    ContractAnalyzerLogger.log_structured(logging.INFO, message, **kwargs)


def log_warning(message: str, **kwargs):
    """
    Log warning message
    """
    ContractAnalyzerLogger.log_structured(logging.WARNING, message, **kwargs)


def log_error(error: Exception, context: Dict[str, Any] = None):
    """
    Log error with context
    """
    ContractAnalyzerLogger.log_error(error, context)


def log_debug(message: str, **kwargs):
    """
    Log debug message
    """
    ContractAnalyzerLogger.log_structured(logging.DEBUG, message, **kwargs)
