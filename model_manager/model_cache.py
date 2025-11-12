# DEPENDENCIES
import os
import sys
import pickle
import hashlib
from typing import Any
from pathlib import Path
from typing import Callable
from typing import Optional
from functools import wraps
from datetime import datetime
from datetime import timedelta

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log_info
from utils.logger import log_error
from utils.logger import ContractAnalyzerLogger



class ModelCache:
    """
    Smart caching for model outputs and embeddings : uses disk-based caching with TTL
    """
    def __init__(self, cache_dir: Path, ttl_seconds: int = 3600):
        self.cache_dir   = Path(cache_dir)
        self.cache_dir.mkdir(parents = True, exist_ok = True)
        
        self.ttl_seconds = ttl_seconds
        self.logger      = ContractAnalyzerLogger.get_logger()
        
        log_info("ModelCache initialized",
                 cache_dir   = str(self.cache_dir),
                 ttl_seconds = ttl_seconds,
                )
    

    def _get_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate cache key from function arguments
        """
        # Create a unique key from arguments
        key_data  = f"{prefix}_{args}_{sorted(kwargs.items())}"
        cache_key = hashlib.md5(key_data.encode()).hexdigest()

        return cache_key
    

    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get file path for cache key
        """
        return self.cache_dir / f"{cache_key}.pkl"
    

    def _is_expired(self, cache_path: Path) -> bool:
        """
        Check if cache file is expired
        """
        if not cache_path.exists():
            return True
        
        file_time  = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age        = datetime.now() - file_time
        is_expired = age > timedelta(seconds = self.ttl_seconds)
        
        return is_expired
    

    def get(self, prefix: str, *args, **kwargs) -> Optional[Any]:
        """
        Get cached value
        """
        cache_key  = self._get_cache_key(prefix, *args, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_expired(cache_path):
            log_info(f"Cache miss (expired): {prefix}",
                     cache_key = cache_key,
                    )

            return None
        
        try:
            with open(cache_path, 'rb') as f:
                result = pickle.load(f)
            
            log_info(f"Cache hit: {prefix}",
                     cache_key    = cache_key,
                     file_size_kb = cache_path.stat().st_size / 1024,
                    )
            
            return result
            
        except Exception as e:
            log_error(e, context = {"component" : "ModelCache", "operation" : "get", "prefix" : prefix, "cache_key" : cache_key})
            
            return None
    

    def set(self, prefix: str, value: Any, *args, **kwargs):
        """
        Set cached value
        """
        cache_key  = self._get_cache_key(prefix, *args, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            file_size = cache_path.stat().st_size
            
            log_info(f"Cache set: {prefix}",
                     cache_key    = cache_key,
                     file_size_kb = file_size / 1024,
                     ttl_seconds  = self.ttl_seconds,
                    )
            
        except Exception as e:
            log_error(e, context = {"component" : "ModelCache", "operation" : "set", "prefix" : prefix, "cache_key" : cache_key})
    

    def clear_expired(self):
        """
        Clear all expired cache files
        """
        expired_count = 0
        freed_bytes   = 0
        
        log_info("Starting cache cleanup (expired files)")
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            if self._is_expired(cache_file):
                file_size      = cache_file.stat().st_size
                cache_file.unlink()

                expired_count += 1
                freed_bytes   += file_size
        
        log_info("Cache cleanup completed",
                 expired_files = expired_count,
                 freed_mb      = freed_bytes / (1024 * 1024),
                )

    
    def clear_all(self):
        """
        Clear all cache files
        """
        file_count  = 0
        freed_bytes = 0
        
        log_info("Clearing all cache files")
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            file_size    = cache_file.stat().st_size
            cache_file.unlink()

            file_count  += 1
            freed_bytes += file_size
        
        log_info("All cache cleared",
                 files_deleted = file_count,
                 freed_mb      = freed_bytes / (1024 * 1024),
                )

    
    def get_size_mb(self) -> float:
        """
        Get total cache size in MB
        """
        total_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        size_mb     = total_bytes / (1024 * 1024)
        
        log_info("Cache size calculated",
                 size_mb    = round(size_mb, 2),
                 file_count = len(list(self.cache_dir.glob("*.pkl"))),
                )
        
        return size_mb
    

    def get_stats(self) -> dict:
        """
        Get cache statistics
        """
        cache_files   = list(self.cache_dir.glob("*.pkl"))
        total_size    = sum(f.stat().st_size for f in cache_files)
        
        expired_files = [f for f in cache_files if self._is_expired(f)]
        
        stats         = {"total_files"   : len(cache_files),
                         "expired_files" : len(expired_files),
                         "total_size_mb" : total_size / (1024 * 1024),
                         "cache_dir"     : str(self.cache_dir),
                         "ttl_seconds"   : self.ttl_seconds,
                        }
        
        log_info("Cache statistics retrieved", **stats)
        
        return stats


def cached(prefix: str, cache: ModelCache):
    """
    Decorator for caching function results
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get from cache
            cached_result = cache.get(prefix, *args, **kwargs)
            
            if cached_result is not None:
                return cached_result
            
            # Compute and cache
            log_info(f"Computing (cache miss): {func.__name__}", prefix=prefix)
            result = func(*args, **kwargs)

            cache.set(prefix, result, *args, **kwargs)
            
            return result
        
        return wrapper

    return decorator