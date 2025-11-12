# DEPENDENCIES
import sys
import json
import time
import requests
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


class LLMProvider(Enum):
    """
    Supported LLM providers
    """
    OLLAMA    = "ollama"
    OPENAI    = "openai"
    ANTHROPIC = "anthropic"


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
    Unified LLM manager for multiple providers : handles Ollama (local), OpenAI API, and Anthropic API
    """
    def __init__(self, default_provider: LLMProvider = LLMProvider.OLLAMA, ollama_base_url: Optional[str] = None,
             openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None):
        """
        Initialize LLM Manager
        
        Arguments:
        ----------
            default_provider  : Default LLM provider to use
            
            ollama_base_url   : Ollama server URL (default: http://localhost:11434)
            
            openai_api_key    : OpenAI API key (or set OPENAI_API_KEY env var)
            
            anthropic_api_key : Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        self.default_provider  = default_provider
        self.logger            = ContractAnalyzerLogger.get_logger()
        
        # Configuration
        self.config            = ModelConfig()
        
        # Ollama configuration
        self.ollama_base_url   = ollama_base_url or "http://localhost:11434"  # Default Ollama URL
        self.ollama_model      = "mistral:7b"  # Default model
        self.ollama_timeout    = 300           # Default timeout
        
        # Get settings from environment or use defaults
        try:
            
            self.ollama_base_url = ollama_base_url or settings.OLLAMA_BASE_URL
            self.ollama_model    = settings.OLLAMA_MODEL
            self.ollama_timeout = settings.OLLAMA_TIMEOUT
        
        except ImportError:
            # Fallback to defaults if settings not available
            pass
        
        # OpenAI configuration
        self.openai_api_key    = openai_api_key
        
        if (OPENAI_AVAILABLE and self.openai_api_key):
            openai.api_key = self.openai_api_key
        
        # Anthropic configuration
        self.anthropic_api_key = anthropic_api_key

        if (ANTHROPIC_AVAILABLE and self.anthropic_api_key):
            self.anthropic_client = anthropic.Anthropic(api_key = self.anthropic_api_key)
        
        else:
            self.anthropic_client = None
        
        # Rate limiting (simple token bucket)
        self._rate_limit_tokens      = 10
        self._rate_limit_last_refill = time.time()

        # Tokens per second
        self._rate_limit_refill_rate = 1.0  
        
        log_info("LLMManager initialized",
                 default_provider    = default_provider.value,
                 ollama_available    = self._check_ollama_available(),
                 openai_available    = OPENAI_AVAILABLE and bool(self.openai_api_key),
                 anthropic_available = ANTHROPIC_AVAILABLE and bool(self.anthropic_api_key),
                )
        

    # PROVIDER AVAILABILITY CHECKS
    def _check_ollama_available(self) -> bool:
        """
        Check if Ollama server is available
        """
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
        Get list of available providers
        """
        available = list()
        
        if (self._check_ollama_available()):
            available.append(LLMProvider.OLLAMA)
        
        if (OPENAI_AVAILABLE and self.openai_api_key):
            available.append(LLMProvider.OPENAI)
        
        if (ANTHROPIC_AVAILABLE and self.anthropic_api_key):
            available.append(LLMProvider.ANTHROPIC)
        
        log_info("Available LLM providers", providers = [p.value for p in available])
        
        return available
    

    # RATE LIMITING
    def _check_rate_limit(self) -> bool:
        """
        Check if rate limit allows request (simple token bucket)
        """
        now         = time.time()
        time_passed = now - self._rate_limit_last_refill
        
        # Refill tokens
        self._rate_limit_tokens      = min(10, self._rate_limit_tokens + time_passed * self._rate_limit_refill_rate)
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
    def complete(self, prompt: str, provider: Optional[LLMProvider] = None, model: Optional[str] = None, temperature: float = 0.1, 
                 max_tokens: int = 2000, system_prompt: Optional[str] = None, json_mode: bool = False, retry_on_error: bool = True, 
                 fallback_providers: Optional[List[LLMProvider]] = None) -> LLMResponse:
        """
        Unified completion method for all providers
        
        Arguments:
        ----------
            prompt             : User prompt
            
            provider           : LLM provider (default: self.default_provider)
            
            model              : Model name (provider-specific)
            
            temperature        : Sampling temperature (0.0-1.0)
            
            max_tokens         : Maximum tokens to generate
            
            system_prompt      : System prompt (if supported)
            
            json_mode          : Force JSON output (if supported)
            
            retry_on_error     : Retry with fallback providers on error
            
            fallback_providers : List of fallback providers to try
        
        Returns:
        --------
            { LLMResponse }    : LLMResponse object
        """
        provider = provider or self.default_provider
        
        log_info("LLM completion request",
                 provider      = provider.value,
                 prompt_length = len(prompt),
                 temperature   = temperature,
                 max_tokens    = max_tokens,
                )
        
        # Rate limiting
        self._wait_for_rate_limit()
        
        # Try primary provider
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

            else:
                raise ValueError(f"Unsupported provider: {provider}")
        
        except Exception as e:
            log_error(e, context = {"component" : "LLMManager", "operation" : "complete", "provider" : provider.value})
            
            # Try fallback providers
            if retry_on_error and fallback_providers:
                log_info("Trying fallback providers", fallbacks = [p.value for p in fallback_providers])
                
                for fallback_provider in fallback_providers:
                    if (fallback_provider == provider):
                        continue
                    
                    try:
                        log_info(f"Attempting fallback to {fallback_provider.value}")
                        return self.complete(prompt         = prompt,
                                             provider       = fallback_provider,
                                             model          = model,
                                             temperature    = temperature,
                                             max_tokens     = max_tokens,
                                             system_prompt  = system_prompt,
                                             json_mode      = json_mode,
                                             retry_on_error = False,  # Prevent infinite recursion
                                            )

                    except Exception as fallback_error:
                        log_error(fallback_error, context = {"component" : "LLMManager", "operation" : "fallback_complete", "provider" : fallback_provider.value})
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
    
    # OLLAMA PROVIDER
    def _complete_ollama(self, prompt: str, model: Optional[str], temperature: float, max_tokens: int, system_prompt: Optional[str], json_mode: bool) -> LLMResponse:
        """
        Complete using local Ollama
        """
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
        
        response       = requests.post(f"{self.ollama_base_url}/api/generate", json = payload, timeout = self.ollama_timeout)
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
    

    # OPENAI PROVIDER
    def _complete_openai(self, prompt: str, model: Optional[str], temperature: float, max_tokens: int, system_prompt: Optional[str], json_mode: bool) -> LLMResponse:
        """
        Complete using OpenAI API
        """
        if not OPENAI_AVAILABLE or not self.openai_api_key:
            raise ValueError("OpenAI not available. Install with: pip install openai")
        
        start_time = time.time()
        model      = model or "gpt-3.5-turbo"
        
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
    
    # ANTHROPIC PROVIDER
    def _complete_anthropic(self, prompt: str, model: Optional[str], temperature: float, max_tokens: int, system_prompt: Optional[str]) -> LLMResponse:
        """
        Complete using Anthropic (Claude) API
        """
        if (not ANTHROPIC_AVAILABLE or not self.anthropic_client):
            raise ValueError("Anthropic not available. Install with: pip install anthropic")
        
        start_time = time.time()
        model      = model or "claude-3-sonnet-20240229"
        
        log_info("Calling Anthropic API", model = model)
        
        # API call
        message        = self.anthropic_client.messages.create(model       = model,
                                                               max_tokens  = max_tokens,
                                                               temperature = temperature,
                                                               system      = system_prompt or "",
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
    
    # SPECIALIZED METHODS
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
    

    def batch_complete(self, prompts: List[str], provider: Optional[LLMProvider] = None, **kwargs) -> List[LLMResponse]:
        """
        Complete multiple prompts (sequential for now)
        
        Arguments:
        ----------
            prompts   : List of prompts
            
            provider  : LLM provider

            **kwargs  : Additional arguments for complete()
        
        Returns:
        --------
            { list }  : List of LLMResponse objects
        """
        log_info("Batch completion started", batch_size=len(prompts))
        
        responses = list()

        for i, prompt in enumerate(prompts):
            log_info(f"Processing prompt {i+1}/{len(prompts)}")

            response = self.complete(prompt   = prompt, 
                                     provider = provider, 
                                     **kwargs,
                                    )

            responses.append(response)
        
        successful = sum(1 for r in responses if r.success)
        
        log_info("Batch completion finished",
                 total       = len(prompts),
                 successful  = successful,
                 failed      = len(prompts) - successful,
                )
        
        return responses
    

    # OLLAMA-SPECIFIC METHODS
    def list_ollama_models(self) -> List[str]:
        """
        List available local Ollama models
        """
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout = 30)
            response.raise_for_status()
            
            models   = [model['name'] for model in response.json().get('models', [])]

            log_info("Ollama models listed", count = len(models), models = models)

            return models
            
        except Exception as e:
            log_error(e, context = {"component" : "LLMManager", "operation" : "list_ollama_models"})
            return []

    
    def pull_ollama_model(self, model_name: str) -> bool:
        """
        Pull/download an Ollama model
        """
        try:
            log_info(f"Pulling Ollama model: {model_name}")
            
            response = requests.post(f"{self.ollama_base_url}/api/pull",
                                     json    = {"name": model_name},
                                     stream  = True,
                                     timeout = 600,  # 10 minutes for download
                                    )

            response.raise_for_status()
            
            # Stream response to track progress
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    
                    if 'status' in data:
                        log_info(f"Pull status: {data['status']}")
            
            log_info(f"Model pulled successfully: {model_name}")
            return True
            
        except Exception as e:
            log_error(e, context = {"component" : "LLMManager", "operation" : "pull_ollama_model", "model" : model_name})
            return False
    
    # UTILITY METHODS
    def get_provider_info(self, provider: LLMProvider) -> Dict[str, Any]:
        """
        Get information about a provider
        """
        info = {"provider"  : provider.value,
                "available" : False,
                "models"    : [],
               }
        
        if (provider == LLMProvider.OLLAMA):
            info["available"] = self._check_ollama_available()

            if info["available"]:
                info["models"]   = self.list_ollama_models()
                info["base_url"] = self.ollama_base_url
        
        elif (provider == LLMProvider.OPENAI):
            info["available"] = OPENAI_AVAILABLE and bool(self.openai_api_key)

            if info["available"]:
                info["models"] = ["gpt-3.5-turbo", 
                                  "gpt-4", 
                                  "gpt-4-turbo-preview",
                                 ]
        
        elif (provider == LLMProvider.ANTHROPIC):
            info["available"] = ANTHROPIC_AVAILABLE and bool(self.anthropic_client)

            if info["available"]:
                info["models"] = ["claude-3-opus-20240229",
                                  "claude-3-sonnet-20240229",
                                  "claude-3-haiku-20240307",
                                 ]
        
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
        # Pricing per 1K tokens (as of 2024)
        pricing = {"openai"    : {"gpt-3.5-turbo"       : {"prompt": 0.0015, "completion": 0.002},
                                  "gpt-4"               : {"prompt": 0.03, "completion": 0.06},
                                  "gpt-4-turbo-preview" : {"prompt": 0.01, "completion": 0.03},
                                 },
                   "anthropic" : {"claude-3-opus-20240229"   : {"prompt": 0.015, "completion": 0.075},
                                  "claude-3-sonnet-20240229" : {"prompt": 0.003, "completion": 0.015},
                                  "claude-3-haiku-20240307"  : {"prompt": 0.00025, "completion": 0.00125},
                                 }
                  }
        
        if (provider == LLMProvider.OLLAMA):
            # Local models are free
            return 0.0  
        
        provider_pricing = pricing.get(provider.value, {}).get(model)
        
        if not provider_pricing:
            return 0.0
        
        cost = ((prompt_tokens / 1000) * provider_pricing["prompt"] + (completion_tokens / 1000) * provider_pricing["completion"])
        
        return round(cost, 6)
