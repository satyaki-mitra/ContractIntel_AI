# DEPENDENCIES
import sys
import json
import time
import requests
import threading
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from pathlib import Path
from typing import Literal
from typing import Optional
from dataclasses import dataclass
from config.settings import settings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log_info
from utils.logger import log_error
from config.model_config import ModelConfig
from utils.logger import ContractAnalyzerLogger


# Optional imports for API providers
try:
    import openai
    OPENAI_AVAILABLE = True

except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True

except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
    
except ImportError:
    LLAMA_CPP_AVAILABLE = False


# Enums and models
class LLMProvider(Enum):
    """
    Supported LLM providers
    """
    OLLAMA     = "ollama"
    OPENAI     = "openai"
    ANTHROPIC  = "anthropic"
    LLAMA_CPP  = "llama_cpp"
    HF_INFER   = "hf_inference"


@dataclass
class LLMResponse:
    """
    Standardized LLM response
    """
    text            : str
    provider        : str
    model           : str
    tokens_used     : int
    latency_seconds : float
    success         : bool
    error_message   : Optional[str]            = None
    raw_response    : Optional[Dict[str, Any]] = None
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        """
        return {"text"            : self.text,
                "provider"        : self.provider,
                "model"           : self.model,
                "tokens_used"     : self.tokens_used,
                "latency_seconds" : round(self.latency_seconds, 3),
                "success"         : self.success,
                "error_message"   : self.error_message,
               }


class LLMManager:
    """
    Unified LLM manager for multiple providers : handles Ollama (local), OpenAI API, Anthropic API, and Llama.cpp
    """
    def __init__(self, default_provider: Optional[LLMProvider] = None, ollama_base_url: Optional[str] = None,
                 openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None):
        """
        Initialize LLM Manager
        
        Arguments:
        ----------
            default_provider  : Default LLM provider to use (if None, uses settings.LLM_DEFAULT_PROVIDER)
            
            ollama_base_url   : Ollama server URL (default: from settings)
            
            openai_api_key    : OpenAI API key (or set OPENAI_API_KEY env var)
            
            anthropic_api_key : Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        self.default_provider   = default_provider or LLMProvider(settings.LLM_DEFAULT_PROVIDER)
        self.logger             = ContractAnalyzerLogger.get_logger()
        
        # Configuration Variables Initialization
        self.config             = ModelConfig()
        
        # Ollama configuration 
        self.ollama_base_url    = ollama_base_url or settings.OLLAMA_BASE_URL
        self.ollama_model       = settings.OLLAMA_MODEL
        self.ollama_timeout     = settings.OLLAMA_TIMEOUT
        self.ollama_temperature = settings.OLLAMA_TEMPERATURE
        
        # OpenAI configuration 
        self.openai_api_key     = openai_api_key or settings.OPENAI_API_KEY
        
        if (OPENAI_AVAILABLE and self.openai_api_key):
            openai.api_key = self.openai_api_key
        
        # Anthropic configuration  
        self.anthropic_api_key = anthropic_api_key or settings.ANTHROPIC_API_KEY

        if (ANTHROPIC_AVAILABLE and self.anthropic_api_key):
            self.anthropic_client = anthropic.Anthropic(api_key = self.anthropic_api_key)
        
        else:
            self.anthropic_client = None
        
        # Llama.cpp configuration (lazy loaded)
        self.llama_cpp_model    = None
        self.llama_cpp_lock     = threading.Lock()
        
        # HuggingFace Inference configuration
        self.hf_client          = None
       
        if (settings.ENABLE_HF_INFERENCE and settings.HF_API_TOKEN):
            try:
                from huggingface_hub import InferenceClient

                self.hf_client = InferenceClient(model = settings.HF_MODEL_ID,
                                                 token = settings.HF_API_TOKEN,
                                                )
            except ImportError:
                log_error("huggingface_hub not installed, HF Inference disabled")
        
        # Rate limiting (simple token bucket)
        self._rate_limit_tokens      = settings.RATE_LIMIT_REQUESTS
        self._rate_limit_last_refill = time.time()
        self._rate_limit_refill_rate = settings.RATE_LIMIT_REQUESTS / settings.RATE_LIMIT_PERIOD
        
        # Generation settings from settings (not ModelConfig)
        self.generation_config       = {"max_tokens"     : settings.LLM_MAX_TOKENS,
                                        "temperature"    : settings.LLM_TEMPERATURE,
                                        "top_p"          : settings.LLM_TOP_P,
                                        "repeat_penalty" : settings.LLM_REPEAT_PENALTY,
                                       }
        
        log_info("LLMManager initialized",
                 default_provider      = self.default_provider.value,
                 deployment_env        = settings.DEPLOYMENT_ENV,
                 ollama_enabled        = settings.ENABLE_OLLAMA,
                 llama_cpp_enabled     = settings.ENABLE_LLAMA_CPP,
                 openai_available      = OPENAI_AVAILABLE and bool(self.openai_api_key),
                 anthropic_available   = ANTHROPIC_AVAILABLE and bool(self.anthropic_api_key),
                 llama_cpp_available   = LLAMA_CPP_AVAILABLE,
                 provider_priority     = settings.LLM_PROVIDER_PRIORITY,
                )


    # Provider Availability Check
    def _check_ollama_available(self) -> bool:
        """
        Check if Ollama server is available
        """
        if not settings.ENABLE_OLLAMA:
            return False
            
        try:
            response  = requests.get(f"{self.ollama_base_url}/api/tags", timeout = 30)
            available = (response.status_code == 200)

            if available:
                log_info("Ollama server is available", base_url = self.ollama_base_url)

            return available

        except Exception as e:
            log_error(e, context = {"component" : "LLMManager", "operation" : "check_ollama"})

            return False

    
    def get_available_providers(self) -> List[LLMProvider]:
        """
        Get list of available providers based on settings and environment
        """
        available = list()
        
        # Check each provider based on settings
        if (settings.ENABLE_OLLAMA and self._check_ollama_available()):
            available.append(LLMProvider.OLLAMA)
        
        if (settings.ENABLE_OPENAI and OPENAI_AVAILABLE and self.openai_api_key):
            available.append(LLMProvider.OPENAI)
        
        if (settings.ENABLE_ANTHROPIC and ANTHROPIC_AVAILABLE and self.anthropic_api_key):
            available.append(LLMProvider.ANTHROPIC)
        
        if (settings.ENABLE_LLAMA_CPP and LLAMA_CPP_AVAILABLE):
            available.append(LLMProvider.LLAMA_CPP)
        
        if (settings.ENABLE_HF_INFERENCE and self.hf_client):
            available.append(LLMProvider.HF_INFER)
        
        # Sort by priority from settings
        priority_order = settings.LLM_PROVIDER_PRIORITY

        available.sort(key = lambda p: priority_order.index(p.value) if p.value in priority_order else len(priority_order))
        
        log_info("Available LLM providers", 
                 providers = [p.value for p in available],
                 priority  = priority_order,
                )
        
        return available
    

    # Rate Limiting
    def _check_rate_limit(self) -> bool:
        """
        Check if rate limit allows request (simple token bucket)
        """
        if not settings.RATE_LIMIT_ENABLED:
            return True
            
        now                          = time.time()
        time_passed                  = now - self._rate_limit_last_refill
        
        # Refill tokens
        self._rate_limit_tokens      = min(settings.RATE_LIMIT_REQUESTS, self._rate_limit_tokens + time_passed * self._rate_limit_refill_rate)
        self._rate_limit_last_refill = now
        
        if (self._rate_limit_tokens >= 1):
            self._rate_limit_tokens -= 1
            return True
        
        log_info("Rate limit hit, waiting...", tokens_remaining = self._rate_limit_tokens)
        
        return False
    

    def _wait_for_rate_limit(self):
        """
        Wait until rate limit allows request
        """
        while not self._check_rate_limit():
            time.sleep(0.5)
    

    # UNIFIED COMPLETION METHOD
    @ContractAnalyzerLogger.log_execution_time("llm_complete")
    def complete(self, prompt: str, provider: Optional[LLMProvider] = None, model: Optional[str] = None, temperature: Optional[float] = None, 
                 max_tokens: Optional[int] = None, system_prompt: Optional[str] = None, json_mode: bool = False, retry_on_error: bool = True,
                 max_retries: int = 3) -> LLMResponse:
        """
        Unified completion method for all providers with automatic fallback
        
        Arguments:
        ----------
            prompt             : User prompt
            
            provider           : LLM provider (default: self.default_provider)
            
            model              : Model name (provider-specific)
            
            temperature        : Sampling temperature (0.0-1.0, default from settings)
            
            max_tokens         : Maximum tokens to generate (default from settings)
            
            system_prompt      : System prompt (if supported)
            
            json_mode          : Force JSON output (if supported)
            
            retry_on_error     : Retry with fallback providers on error
            
            max_retries        : Maximum number of retry attempts
        
        Returns:
        --------
            { LLMResponse }    : LLMResponse object
        """
        provider      = provider or self.default_provider
        temperature   = temperature or settings.LLM_TEMPERATURE
        max_tokens    = max_tokens or settings.LLM_MAX_TOKENS
        system_prompt = system_prompt or settings.LLM_SYSTEM_PROMPT
        
        log_info("LLM completion request",
                 provider      = provider.value,
                 prompt_length = len(prompt),
                 temperature   = temperature,
                 max_tokens    = max_tokens,
                 json_mode     = json_mode,
                )
        
        # Rate limiting
        self._wait_for_rate_limit()
        
        # Try primary provider with retries
        for attempt in range(max_retries if retry_on_error else 1):
            try:
                if (provider == LLMProvider.OLLAMA):
                    return self._complete_ollama(prompt        = prompt,
                                                 model         = model, 
                                                 temperature   = temperature,
                                                 max_tokens    = max_tokens, 
                                                 system_prompt = system_prompt, 
                                                 json_mode     = json_mode,
                                                )

                elif (provider == LLMProvider.OPENAI):
                    return self._complete_openai(prompt        = prompt, 
                                                 model         = model, 
                                                 temperature   = temperature, 
                                                 max_tokens    = max_tokens, 
                                                 system_prompt = system_prompt, 
                                                 json_mode     = json_mode,
                                                )

                elif (provider == LLMProvider.ANTHROPIC):
                    return self._complete_anthropic(prompt        = prompt, 
                                                    model         = model, 
                                                    temperature   = temperature, 
                                                    max_tokens    = max_tokens, 
                                                    system_prompt = system_prompt,
                                                   )

                elif (provider == LLMProvider.LLAMA_CPP):
                    return self._complete_llama_cpp(prompt        = prompt,
                                                    model         = model,
                                                    temperature   = temperature,
                                                    max_tokens    = max_tokens,
                                                    system_prompt = system_prompt,
                                                    json_mode     = json_mode,
                                                   )

                elif (provider == LLMProvider.HF_INFER):
                    return self._complete_hf_inference(prompt        = prompt,
                                                       model         = model,
                                                       temperature   = temperature,
                                                       max_tokens    = max_tokens,
                                                       system_prompt = system_prompt,
                                                      )

                else:
                    raise ValueError(f"Unsupported provider: {provider}")
            
            except Exception as e:
                log_error(e, context = {"component" : "LLMManager", 
                                        "operation" : "complete", 
                                        "provider"  : provider.value,
                                        "attempt"   : attempt + 1,
                                       }
                         )
                
                if (attempt < max_retries - 1):
                    log_info(f"Retrying attempt {attempt + 2}/{max_retries}")
                    # Exponential backoff
                    time.sleep(1 * (attempt + 1))  
                    continue
                
                # If retries exhausted, try fallback providers
                if retry_on_error:
                    available_providers = self.get_available_providers()
                    # Remove current provider from fallback list
                    fallback_providers  = [p for p in available_providers if p != provider]
                    
                    for fallback_provider in fallback_providers:
                        try:
                            log_info(f"Attempting fallback to {fallback_provider.value}")
                            # Prevent infinite recursion by disabling further fallbacks
                            return self.complete(prompt         = prompt,
                                                 provider       = fallback_provider,
                                                 model          = model,
                                                 temperature    = temperature,
                                                 max_tokens     = max_tokens,
                                                 system_prompt  = system_prompt,
                                                 json_mode      = json_mode,
                                                 retry_on_error = False,  # No more fallbacks
                                                )

                        except Exception as fallback_error:
                            log_error(fallback_error, context = {"component" : "LLMManager", 
                                                                 "operation" : "fallback_complete", 
                                                                 "provider"  : fallback_provider.value,
                                                                }
                                     )
                            continue
                
                # All attempts failed
                return LLMResponse(text            = "",
                                   provider        = provider.value,
                                   model           = model or "unknown",
                                   tokens_used     = 0,
                                   latency_seconds = 0.0,
                                   success         = False,
                                   error_message   = str(e),
                                  )
        
        # Should never reach here
        return LLMResponse(text            = "",
                           provider        = provider.value,
                           model           = model or "unknown",
                           tokens_used     = 0,
                           latency_seconds = 0.0,
                           success         = False,
                           error_message   = "Unknown error",
                          )


    # OLLAMA Provider
    def _complete_ollama(self, prompt: str, model: Optional[str], temperature: float, max_tokens: int, system_prompt: Optional[str], json_mode: bool) -> LLMResponse:
        """
        Complete using local Ollama
        """
        if not settings.ENABLE_OLLAMA:
            raise ValueError("Ollama is disabled in settings")
            
        start_time  = time.time()
        model       = model or self.ollama_model
        
        # Construct full prompt with system prompt
        full_prompt = prompt
        
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
        
        payload = {"model"   : model,
                   "prompt"  : full_prompt,
                   "stream"  : False,
                   "options" : {"temperature": temperature, "num_predict": max_tokens},
                  }
        
        if json_mode:
            payload["format"] = "json"
        
        log_info("Calling Ollama API",
                 model     = model,
                 base_url  = self.ollama_base_url,
                 json_mode = json_mode,
                )
        
        response       = requests.post(f"{self.ollama_base_url}/api/generate", 
                                       json    = payload, 
                                       timeout = self.ollama_timeout,
                                      )

        response.raise_for_status()
        
        result         = response.json()
        generated_text = result.get('response', '')
        
        latency        = time.time() - start_time
        
        # Estimate tokens (rough approximation)
        tokens_used    = len(prompt.split()) + len(generated_text.split())
        
        log_info("Ollama completion successful",
                 model           = model,
                 tokens_used     = tokens_used,
                 latency_seconds = round(latency, 3),
                )
        
        return LLMResponse(text            = generated_text,
                           provider        = "ollama",
                           model           = model,
                           tokens_used     = tokens_used,
                           latency_seconds = latency,
                           success         = True,
                           raw_response    = result,
                          )
    

    # Open-AI Provider
    def _complete_openai(self, prompt: str, model: Optional[str], temperature: float, max_tokens: int, system_prompt: Optional[str], json_mode: bool) -> LLMResponse:
        """
        Complete using OpenAI API
        """
        if not settings.ENABLE_OPENAI:
            raise ValueError("OpenAI is disabled in settings")
            
        if not OPENAI_AVAILABLE or not self.openai_api_key:
            raise ValueError("OpenAI not available. Install with: pip install openai")
        
        start_time = time.time()
        model      = model or settings.OPENAI_MODEL
        
        # Construct messages
        messages   = list()

        if system_prompt:
            messages.append({"role"    : "system", 
                             "content" : system_prompt,
                           })

        messages.append({"role"    : "user",
                         "content" : prompt,
                       })

        
        log_info("Calling OpenAI API", model = model, json_mode = json_mode)
        
        # API call parameters
        api_params = {"model"       : model,
                      "messages"    : messages,
                      "temperature" : temperature,
                      "max_tokens"  : max_tokens,
                     }
        
        if json_mode:
            api_params["response_format"] = {"type": "json_object"}
        
        response       = openai.ChatCompletion.create(**api_params)
        generated_text = response.choices[0].message.content
        tokens_used    = response.usage.total_tokens
        latency        = time.time() - start_time
        
        log_info("OpenAI completion successful", model = model, tokens_used = tokens_used, latency_seconds = round(latency, 3))
        
        return LLMResponse(text            = generated_text,
                           provider        = "openai",
                           model           = model,
                           tokens_used     = tokens_used,
                           latency_seconds = latency,
                           success         = True,
                           raw_response    = response.to_dict(),
                          )
    
    # Anthropic Provider
    def _complete_anthropic(self, prompt: str, model: Optional[str], temperature: float, max_tokens: int, system_prompt: Optional[str]) -> LLMResponse:
        """
        Complete using Anthropic (Claude) API
        """
        if not settings.ENABLE_ANTHROPIC:
            raise ValueError("Anthropic is disabled in settings")
            
        if not ANTHROPIC_AVAILABLE or not self.anthropic_client:
            raise ValueError("Anthropic not available. Install with: pip install anthropic")
        
        start_time     = time.time()
        model          = model or settings.ANTHROPIC_MODEL
        
        log_info("Calling Anthropic API", model = model)
        
        # API call
        message        = self.anthropic_client.messages.create(model       = model,
                                                               max_tokens  = max_tokens,
                                                               temperature = temperature,
                                                               system      = system_prompt or settings.LLM_SYSTEM_PROMPT,
                                                               messages    = [{"role": "user", "content": prompt}],
                                                              )
        
        generated_text = message.content[0].text
        tokens_used    = message.usage.input_tokens + message.usage.output_tokens
        latency        = time.time() - start_time
        
        log_info("Anthropic completion successful", model = model, tokens_used = tokens_used, latency_seconds = round(latency, 3))
        
        return LLMResponse(text            = generated_text,
                           provider        = "anthropic",
                           model           = model,
                           tokens_used     = tokens_used,
                           latency_seconds = latency,
                           success         = True,
                           raw_response    = message.dict(),
                          )
    

    # Llama.cpp Provider
    def _complete_llama_cpp(self, prompt: str, model: Optional[str], temperature: float, max_tokens: int, system_prompt: Optional[str], json_mode: bool) -> LLMResponse:
        """
        Complete using Llama.cpp (GGUF models)
        """
        if not settings.ENABLE_LLAMA_CPP:
            raise ValueError("Llama.cpp is disabled in settings")
            
        if not LLAMA_CPP_AVAILABLE:
            raise ValueError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        
        start_time    = time.time()
        
        # Lazy load the model
        with self.llama_cpp_lock:
            if self.llama_cpp_model is None:
                self._load_llama_cpp_model()
        
        # Construct full prompt
        system_prompt  = system_prompt or settings.LLM_SYSTEM_PROMPT

        full_prompt    = f"""
                            {system_prompt}

                            {prompt}

                            Response:
                         """
        
        log_info("Calling Llama.cpp",
                 model_path = str(settings.LLAMA_CPP_MODEL_PATH),
                 n_ctx      = settings.LLAMA_CPP_N_CTX,
                 json_mode  = json_mode,
                )
        
        # Generate response
        response       = self.llama_cpp_model(prompt         = full_prompt,
                                              max_tokens     = max_tokens,
                                              temperature    = temperature,
                                              top_p          = settings.LLM_TOP_P,
                                              repeat_penalty = settings.LLM_REPEAT_PENALTY,
                                              stop           = ["\n\n", "###", "Human:", "Assistant:", "</s>"],
                                              echo           = False,
                                             )
        
        generated_text = response['choices'][0]['text'].strip()
        latency        = time.time() - start_time
        
        # Rough token estimation
        tokens_used    = len(full_prompt.split()) + len(generated_text.split())
        
        log_info("Llama.cpp completion successful",
                 tokens_used     = tokens_used,
                 latency_seconds = round(latency, 3),
                )
        
        return LLMResponse(text            = generated_text,
                           provider        = "llama_cpp",
                           model           = str(settings.LLAMA_CPP_MODEL_PATH),
                           tokens_used     = tokens_used,
                           latency_seconds = latency,
                           success         = True,
                           raw_response    = response,
                          )
    

    def _load_llama_cpp_model(self):
        """
        Lazy load the Llama.cpp model
        """
        log_info("Loading Llama.cpp model", model_path=str(settings.LLAMA_CPP_MODEL_PATH))
        
        # Ensure model exists, download if needed
        if( not settings.LLAMA_CPP_MODEL_PATH.exists()):
            self._download_llama_cpp_model()
        
        # Load model with appropriate GPU layers / CPU loading
        n_gpu_layers         = settings.LLAMA_CPP_N_GPU_LAYERS
        
        if settings.IS_HUGGINGFACE_SPACE:
            n_gpu_layers = 0 
        
        self.llama_cpp_model = Llama(model_path   = str(settings.LLAMA_CPP_MODEL_PATH),
                                     n_ctx        = settings.LLAMA_CPP_N_CTX,
                                     n_gpu_layers = n_gpu_layers,  
                                     n_batch      = settings.LLAMA_CPP_N_BATCH,
                                     n_threads    = settings.LLAMA_CPP_N_THREADS,
                                     verbose      = False,
                                    )

        log_info("Llama.cpp model loaded successfully")
    

    def _download_llama_cpp_model(self):
        """
        Download GGUF model from HuggingFace Hub
        """
        log_info("Downloading GGUF model", repo = settings.LLAMA_CPP_MODEL_REPO, filename = settings.LLAMA_CPP_MODEL_FILE)
        
        try:
            from huggingface_hub import hf_hub_download
            
            # Ensure cache directory exists
            settings.MODEL_CACHE_DIR.mkdir(parents = True, exist_ok = True)
            
            # Download the model
            downloaded_path = hf_hub_download(repo_id         = settings.LLAMA_CPP_MODEL_REPO,
                                              filename        = settings.LLAMA_CPP_MODEL_FILE,
                                              cache_dir       = str(settings.MODEL_CACHE_DIR),
                                              force_download  = False,
                                              resume_download = True,
                                             )
            
            # Create symlink to expected path
            if (downloaded_path != str(settings.LLAMA_CPP_MODEL_PATH)):
                import shutil
                shutil.copy(downloaded_path, settings.LLAMA_CPP_MODEL_PATH)
            
            log_info("GGUF model downloaded successfully", path = str(settings.LLAMA_CPP_MODEL_PATH))
            
        except Exception as e:
            log_error(e, context = {"component" : "LLMManager",
                                    "operation" : "download_llama_cpp_model",
                                    "repo"      : settings.LLAMA_CPP_MODEL_REPO,
                                    "filename"  : settings.LLAMA_CPP_MODEL_FILE,
                                   }
                     )
            raise
    

    # HuggingFace Inference Provider
    def _complete_hf_inference(self, prompt: str, model: Optional[str], temperature: float, max_tokens: int, system_prompt: Optional[str]) -> LLMResponse:
        """
        Complete using HuggingFace Inference API
        """
        if not settings.ENABLE_HF_INFERENCE or not self.hf_client:
            raise ValueError("HF Inference is disabled or not configured")
        
        start_time     = time.time()
        
        # Construct full prompt
        full_prompt    = f"""
                             {system_prompt or settings.LLM_SYSTEM_PROMPT}

                             {prompt}

                             Response:
                          """
        
        log_info("Calling HuggingFace Inference API")
        
        # Generate response
        response       = self.hf_client.text_generation(full_prompt,
                                                        max_new_tokens   = max_tokens,
                                                        temperature      = temperature,
                                                        do_sample        = True,
                                                        return_full_text = False,
                                                       )
        
        generated_text = response
        latency        = time.time() - start_time
        
        # Rough token estimation
        tokens_used    = len(full_prompt.split()) + len(generated_text.split())
        
        log_info("HF Inference completion successful",
                 tokens_used     = tokens_used,
                 latency_seconds = round(latency, 3),
                )
        
        return LLMResponse(text            = generated_text,
                           provider        = "hf_inference",
                           model           = settings.HF_MODEL_ID or "hf_inference",
                           tokens_used     = tokens_used,
                           latency_seconds = latency,
                           success         = True,
                           raw_response    = {"text": generated_text},
                          )
    

    # Specialized Methods 
    def generate_structured_json(self, prompt: str, schema_description: str, provider: Optional[LLMProvider] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate structured JSON output
        
        Arguments:
        ----------
            prompt             : User prompt

            schema_description : Description of expected JSON schema
            
            provider           : LLM provider
            
            **kwargs           : Additional arguments for complete()
        
        Returns:
        --------
               { dict }        : Parsed JSON dictionary
        """
        system_prompt = (f"You are a helpful assistant that returns valid JSON.\n"
                         f"Expected schema:\n{schema_description}\n\n"
                         f"Return ONLY valid JSON, no markdown, no explanation."
                        )
        
        response      = self.complete(prompt        = prompt,
                                      provider      = provider,
                                      system_prompt = system_prompt,
                                      json_mode     = True,
                                      **kwargs,
                                     )
        
        if not response.success:
            raise ValueError(f"LLM completion failed: {response.error_message}")
        
        # Parse JSON
        try:
            # Clean response (remove markdown code blocks if present)
            text   = response.text.strip()
            text   = text.replace("```json", "").replace("```", "").strip()
            
            parsed = json.loads(text)

            log_info("JSON parsing successful", keys = list(parsed.keys()))
            
            return parsed
            
        except json.JSONDecodeError as e:
            log_error(e, context = {"component" : "LLMManager", "operation" : "parse_json", "response_text" : response.text})
            raise ValueError(f"Failed to parse JSON response: {e}")
    

    # Utility Methods
    def get_provider_info(self, provider: LLMProvider) -> Dict[str, Any]:
        """
        Get information about a provider
        """
        info = {"provider"  : provider.value,
                "available" : False,
                "models"    : [],
               }
        
        if (provider == LLMProvider.OLLAMA):
            info["available"] = settings.ENABLE_OLLAMA and self._check_ollama_available()

            if info["available"]:
                info["models"]   = self.list_ollama_models()
                info["base_url"] = self.ollama_base_url
        
        elif (provider == LLMProvider.OPENAI):
            info["available"] = settings.ENABLE_OPENAI and OPENAI_AVAILABLE and bool(self.openai_api_key)

            if info["available"]:
                info["models"] = [settings.OPENAI_MODEL, "gpt-4", "gpt-4-turbo-preview"]
        
        elif (provider == LLMProvider.ANTHROPIC):
            info["available"] = settings.ENABLE_ANTHROPIC and ANTHROPIC_AVAILABLE and bool(self.anthropic_client)

            if info["available"]:
                info["models"] = [settings.ANTHROPIC_MODEL, "claude-3-sonnet-20240229", "claude-3-opus-20240229"]
        
        elif (provider == LLMProvider.LLAMA_CPP):
            info["available"]  = settings.ENABLE_LLAMA_CPP and LLAMA_CPP_AVAILABLE
            info["model_path"] = str(settings.LLAMA_CPP_MODEL_PATH) if settings.LLAMA_CPP_MODEL_PATH else None
            info["model_repo"] = settings.LLAMA_CPP_MODEL_REPO
        
        elif (provider == LLMProvider.HF_INFER):
            info["available"] = settings.ENABLE_HF_INFERENCE and self.hf_client is not None
            info["model_id"]  = settings.HF_MODEL_ID
        
        return info
    

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, provider: LLMProvider, model: str) -> float:
        """
        Estimate API cost in USD
        
        Arguments:
        ----------
            prompt_tokens     : Number of prompt tokens
            
            completion_tokens : Number of completion tokens
            
            provider          : LLM provider
            
            model             : Model name
        
        Returns:
        --------
                { float }     : Estimated cost in USD
        """
        # Local models (Ollama, Llama.cpp) are free
        if provider in [LLMProvider.OLLAMA, LLMProvider.LLAMA_CPP, LLMProvider.HF_INFER]:
            return 0.0
        
        # Pricing per 1K tokens (as of 2025)
        pricing          = {"openai"    : {"gpt-3.5-turbo"       : {"prompt": 0.0015, "completion": 0.002},
                                           "gpt-4"               : {"prompt": 0.03, "completion": 0.06},
                                           "gpt-4-turbo-preview" : {"prompt": 0.01, "completion": 0.03},
                                          },
                            "anthropic" : {"claude-3-opus-20240229"   : {"prompt": 0.015, "completion": 0.075},
                                           "claude-3-sonnet-20240229" : {"prompt": 0.003, "completion": 0.015},
                                           "claude-3-haiku-20240307"  : {"prompt": 0.00025, "completion": 0.00125},
                                          }
                           }
        
        provider_pricing = pricing.get(provider.value, {}).get(model)
        
        if not provider_pricing:
            return 0.0
        
        cost             = ((prompt_tokens / 1000) * provider_pricing["prompt"] + (completion_tokens / 1000) * provider_pricing["completion"])
        
        return round(cost, 6)