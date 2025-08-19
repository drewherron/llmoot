"""Claude API client implementation."""

import requests
import json
from typing import Dict, Any
from src.llm_client import LLMClient, LLMRequest, LLMResponse, APIError, RateLimitError, AuthenticationError, ContextLengthError


class ClaudeClient(LLMClient):
    """Claude API client implementation."""
    
    def __init__(self, api_key: str, model: str):
        """Initialize Claude client.
        
        Args:
            api_key: Anthropic API key
            model: Claude model name
        """
        super().__init__(api_key, model)
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    def _get_provider_name(self) -> str:
        """Return provider name."""
        return "claude"
    
    def _make_api_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Make request to Claude API.
        
        Args:
            request: Standardized request object
            
        Returns:
            Raw API response
            
        Raises:
            APIError: For API-specific errors
        """
        # Build messages
        messages = [
            {
                "role": "user",
                "content": request.prompt
            }
        ]
        
        # Build request payload
        payload = {
            "model": self.model,
            "max_tokens": request.max_tokens or 1000,
            "messages": messages,
            "system": request.system_prompt
        }
        
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
                error_type = error_data.get("error", {}).get("type", "")
                if "context_length" in error_type.lower():
                    raise ContextLengthError("Context length exceeded")
                else:
                    raise APIError(f"Bad request: {error_data.get('error', {}).get('message', 'Unknown error')}")
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
        """Parse Claude API response.
        
        Args:
            raw_response: Raw response from Claude API
            
        Returns:
            Standardized LLMResponse object
        """
        try:
            # Extract content from response
            content = ""
            if "content" in raw_response and raw_response["content"]:
                # Claude returns content as a list of content blocks
                for block in raw_response["content"]:
                    if block.get("type") == "text":
                        content += block.get("text", "")
            
            # Extract token usage
            usage = raw_response.get("usage", {})
            tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            
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
            raise APIError(f"Failed to parse Claude response: {str(e)}")
    
    def get_context_limit(self) -> int:
        """Get context limit for Claude models."""
        # Claude 3 models have different context limits
        if "opus" in self.model.lower():
            return 200000
        elif "sonnet" in self.model.lower():
            return 200000
        elif "haiku" in self.model.lower():
            return 200000
        else:
            # Default for unknown models
            return 100000
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens for Claude (roughly 3.5 chars per token)."""
        return len(text) // 4
    
    def _validate_request_specific(self, request: LLMRequest) -> None:
        """Claude-specific request validation."""
        if request.max_tokens and request.max_tokens > 4096:
            raise ValueError("Claude max_tokens cannot exceed 4096")
        
        if request.temperature and (request.temperature < 0 or request.temperature > 1):
            raise ValueError("Claude temperature must be between 0 and 1")