"""Google Gemini API client implementation."""

import requests
import json
from typing import Dict, Any
from src.llm_client import LLMClient, LLMRequest, LLMResponse, APIError, RateLimitError, AuthenticationError, ContextLengthError


class GeminiClient(LLMClient):
    """Google Gemini API client implementation."""
    
    def __init__(self, api_key: str, model: str):
        """Initialize Gemini client.
        
        Args:
            api_key: Google API key
            model: Gemini model name (gemini-pro, etc.)
        """
        super().__init__(api_key, model)
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def _get_provider_name(self) -> str:
        """Return provider name."""
        return "gemini"
    
    def _make_api_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Make request to Gemini API.
        
        Args:
            request: Standardized request object
            
        Returns:
            Raw API response
            
        Raises:
            APIError: For API-specific errors
        """
        # Combine system prompt and user prompt for Gemini
        # Gemini doesn't have separate system prompts like Claude/OpenAI
        combined_prompt = f"{request.system_prompt}\n\nUser request: {request.prompt}"
        
        # Build request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": combined_prompt
                        }
                    ]
                }
            ]
        }
        
        # Add generation config if specified
        generation_config = {}
        if request.max_tokens:
            generation_config["maxOutputTokens"] = request.max_tokens
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        
        if generation_config:
            payload["generationConfig"] = generation_config
        
        # Add API key to URL
        url = f"{self.base_url}?key={self.api_key}"
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            # Handle HTTP errors
            if response.status_code == 401 or response.status_code == 403:
                raise AuthenticationError("Invalid API key or insufficient permissions")
            elif response.status_code == 429:
                retry_after = response.headers.get("retry-after")
                retry_after = int(retry_after) if retry_after else None
                raise RateLimitError("Rate limit exceeded", response.status_code, retry_after)
            elif response.status_code == 400:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error_message = error_data.get("error", {}).get("message", "Unknown error")
                if "context length" in error_message.lower() or "token limit" in error_message.lower():
                    raise ContextLengthError("Context length exceeded")
                elif "INVALID_ARGUMENT" in error_message:
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
        """Parse Gemini API response.
        
        Args:
            raw_response: Raw response from Gemini API
            
        Returns:
            Standardized LLMResponse object
        """
        try:
            # Extract content from response
            content = ""
            candidates = raw_response.get("candidates", [])
            if candidates:
                candidate = candidates[0]
                content_parts = candidate.get("content", {}).get("parts", [])
                for part in content_parts:
                    if "text" in part:
                        content += part["text"]
            
            if not content:
                # Check for content filtering or other issues
                if candidates and "finishReason" in candidates[0]:
                    finish_reason = candidates[0]["finishReason"]
                    if finish_reason == "SAFETY":
                        raise APIError("Content filtered by safety settings")
                    elif finish_reason == "RECITATION":
                        raise APIError("Content blocked due to recitation")
                    else:
                        raise APIError(f"Generation stopped: {finish_reason}")
                else:
                    raise APIError("No content generated")
            
            # Extract token usage if available
            # Note: Gemini doesn't always provide detailed token usage
            usage = raw_response.get("usageMetadata", {})
            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            tokens_used = input_tokens + output_tokens
            
            # If no usage data, estimate
            if tokens_used == 0:
                tokens_used = self.estimate_tokens(content)
            
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
            raise APIError(f"Failed to parse Gemini response: {str(e)}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens for Gemini (roughly 4 chars per token)."""
        return len(text) // 4
    
    def _validate_request_specific(self, request: LLMRequest) -> None:
        """Gemini-specific request validation."""
        # Loosened this restriction to allow for larger outputs from models like Gemini 1.5 Pro
        if request.max_tokens and request.max_tokens > 1_000_000:
            raise ValueError("Gemini max_tokens cannot exceed 1,000,000")
        
        if request.temperature and (request.temperature < 0 or request.temperature > 1):
            raise ValueError("Gemini temperature must be between 0 and 1")