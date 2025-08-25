"""OpenAI API client implementation."""

import requests
import json
from typing import Dict, Any
from src.llm_client import LLMClient, LLMRequest, LLMResponse, APIError, RateLimitError, AuthenticationError, ContextLengthError


class OpenAIClient(LLMClient):
    """OpenAI API client implementation."""
    
    def __init__(self, api_key: str, model: str):
        """Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model name (gpt-4, gpt-3.5-turbo, etc.)
        """
        super().__init__(api_key, model)
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _get_provider_name(self) -> str:
        """Return provider name."""
        return "openai"
    
    def _make_api_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Make request to OpenAI API.
        
        Args:
            request: Standardized request object
            
        Returns:
            Raw API response
            
        Raises:
            APIError: For API-specific errors
        """
        # Build messages with system and user prompts
        messages = [
            {
                "role": "system",
                "content": request.system_prompt
            },
            {
                "role": "user", 
                "content": request.prompt
            }
        ]
        
        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages
        }

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            # Handle HTTP errors
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 429:
                retry_after = response.headers.get("retry-after")
                retry_after = int(retry_after) if retry_after else None
                raise RateLimitError("Rate limit exceeded", response.status_code, retry_after)
            elif response.status_code == 400:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error_message = error_data.get("error", {}).get("message", "Unknown error")
                if "maximum context length" in error_message.lower():
                    raise ContextLengthError("Context length exceeded")
                elif "invalid_request_error" in error_data.get("error", {}).get("type", ""):
                    raise APIError(f"Invalid request: {error_message}")
                else:
                    raise APIError(f"Bad request: {error_message}")
            elif response.status_code >= 500:
                raise APIError(f"Server error: {response.status_code}")
            elif not response.ok:
                raise APIError(f"HTTP error: {response.status_code}")
            
            return response.json()
            
        except requests.exceptions.Timeout:
            raise APIError("Request timeout")
        except requests.exceptions.ConnectionError:
            raise APIError("Connection error")
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed: {str(e)}")
        except json.JSONDecodeError:
            raise APIError("Invalid JSON response")
    
    def _parse_response(self, raw_response: Dict[str, Any]) -> LLMResponse:
        """Parse OpenAI API response.
        
        Args:
            raw_response: Raw response from OpenAI API
            
        Returns:
            Standardized LLMResponse object
        """
        try:
            # Extract content from response
            choices = raw_response.get("choices", [])
            if not choices:
                raise APIError("No choices in OpenAI response")
            
            content = choices[0].get("message", {}).get("content", "")
            
            # Extract token usage
            usage = raw_response.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
            
            return LLMResponse(
                content=content.strip(),
                provider=self.provider_name,
                model=self.model,
                tokens_used=tokens_used,
                response_time=0,  # Will be set by base class
                is_final=False,   # Will be set by base class
                raw_response=raw_response
            )
            
        except (KeyError, TypeError) as e:
            raise APIError(f"Failed to parse OpenAI response: {str(e)}")
    
    def get_context_limit(self) -> int:
        """Get context limit for OpenAI models."""
        if "gpt-4" in self.model.lower():
            if "32k" in self.model.lower():
                return 32768
            elif "turbo" in self.model.lower():
                return 128000  # GPT-4 Turbo has 128k context
            else:
                return 8192    # Standard GPT-4
        elif "gpt-3.5-turbo" in self.model.lower():
            if "16k" in self.model.lower():
                return 16384
            else:
                return 4096    # Standard GPT-3.5-turbo
        else:
            # Default for unknown models
            return 4096
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens for OpenAI (roughly 4 chars per token)."""
        return len(text) // 4
    
    def _validate_request_specific(self, request: LLMRequest) -> None:
        """OpenAI-specific request validation."""
        if request.max_tokens and request.max_tokens > 4096:
            # Allow higher limits for models that support it
            if "32k" in self.model.lower() and request.max_tokens > 32768:
                raise ValueError("Max tokens exceeds model limit")
            elif "turbo" in self.model.lower() and request.max_tokens > 4096:
                # GPT-4 Turbo can handle more, but we'll be conservative with output
                pass
            elif "gpt-4" not in self.model.lower():
                raise ValueError("Max tokens too high for this model")
        
        if request.temperature and (request.temperature < 0 or request.temperature > 2):
            raise ValueError("OpenAI temperature must be between 0 and 2")