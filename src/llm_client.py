"""Base LLM client interface and common functionality."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class LLMResponse:
    """Standardized response from an LLM."""
    content: str
    provider: str
    model: str
    tokens_used: int
    response_time: float
    is_final: bool
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class LLMRequest:
    """Standardized request to an LLM."""
    prompt: str
    system_prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    is_final: bool = False


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class APIError(LLMError):
    """API-specific errors (rate limits, auth, etc.)."""
    def __init__(self, message: str, status_code: Optional[int] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class RateLimitError(APIError):
    """Rate limit exceeded error."""
    pass


class AuthenticationError(APIError):
    """Authentication/API key error."""
    pass


class ContextLengthError(LLMError):
    """Context length exceeded error."""
    pass


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, api_key: str, model: str):
        """Initialize the client.
        
        Args:
            api_key: API key for the provider
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model
        self.provider_name = self._get_provider_name()
    
    @abstractmethod
    def _get_provider_name(self) -> str:
        """Return the provider name for this client."""
        pass
    
    @abstractmethod
    def _make_api_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Make the actual API request.
        
        Args:
            request: Standardized request object
            
        Returns:
            Raw API response
            
        Raises:
            APIError: For API-specific errors
            LLMError: For other LLM-related errors
        """
        pass
    
    @abstractmethod
    def _parse_response(self, raw_response: Dict[str, Any]) -> LLMResponse:
        """Parse the raw API response into standardized format.
        
        Args:
            raw_response: Raw response from API
            
        Returns:
            Standardized LLMResponse object
        """
        pass
    
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            request: Standardized request object
            
        Returns:
            Standardized response object
            
        Raises:
            LLMError: For various LLM-related errors
        """
        start_time = time.time()
        
        try:
            raw_response = self._make_api_request(request)
            response = self._parse_response(raw_response)
            
            # Set timing and metadata
            response.response_time = time.time() - start_time
            response.provider = self.provider_name
            response.model = self.model
            response.is_final = request.is_final
            response.raw_response = raw_response
            
            return response
            
        except Exception as e:
            # Return error response
            return LLMResponse(
                content="",
                provider=self.provider_name,
                model=self.model,
                tokens_used=0,
                response_time=time.time() - start_time,
                is_final=request.is_final,
                error=str(e)
            )
    
    def validate_request(self, request: LLMRequest) -> None:
        """Validate a request before sending.
        
        Args:
            request: Request to validate
            
        Raises:
            ValueError: If request is invalid
        """
        if not request.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if not request.system_prompt.strip():
            raise ValueError("System prompt cannot be empty")
        
        # Provider-specific validation can be overridden
        self._validate_request_specific(request)
    
    def _validate_request_specific(self, request: LLMRequest) -> None:
        """Provider-specific request validation. Override in subclasses."""
        pass
    
    def get_context_limit(self) -> int:
        """Get the context limit for this model. Override in subclasses."""
        # Default conservative limit
        return 4000
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation. Override in subclasses for accuracy."""
        # Very rough estimate: ~4 characters per token
        return len(text) // 4