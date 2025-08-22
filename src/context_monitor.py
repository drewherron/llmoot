"""Context length monitoring and management system."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from .llm_client import LLMRequest, LLMResponse
from .order_parser import ExecutionStep


class ContextWarningLevel(Enum):
    """Context warning levels."""
    SAFE = "safe"           # Under 60% of limit
    CAUTION = "caution"     # 60-75% of limit
    WARNING = "warning"     # 75-90% of limit
    CRITICAL = "critical"   # 90%+ of limit


@dataclass
class ContextUsage:
    """Tracks context usage for a provider."""
    provider: str
    model: str
    context_limit: int
    current_usage: int
    warning_level: ContextWarningLevel
    percentage_used: float
    remaining_tokens: int


@dataclass
class TokenEstimate:
    """Token count estimate for text."""
    text: str
    estimated_tokens: int
    method: str  # "simple", "tiktoken", "approximate"


class TokenEstimator:
    """Estimates token counts for different providers."""
    
    def __init__(self):
        """Initialize token estimator."""
        self.provider_ratios = {
            # Characters per token ratios (approximate)
            'claude': 3.5,      # Anthropic models
            'openai': 4.0,      # OpenAI models
            'gemini': 3.8,      # Google models
            'default': 4.0      # Conservative default
        }
    
    def estimate_tokens(self, text: str, provider: str = None) -> TokenEstimate:
        """
        Estimate token count for text.
        
        Args:
            text: Text to analyze
            provider: Provider name for model-specific estimation
            
        Returns:
            TokenEstimate with count and method used
        """
        if not text:
            return TokenEstimate(text="", estimated_tokens=0, method="empty")
        
        # Use provider-specific ratio or default
        char_per_token = self.provider_ratios.get(provider, self.provider_ratios['default'])
        
        # Simple character-based estimation
        estimated = max(1, len(text) // char_per_token)
        
        return TokenEstimate(
            text=text,
            estimated_tokens=estimated,
            method="simple"
        )
    
    def estimate_conversation_tokens(self, messages: List[Dict[str, str]], provider: str = None) -> int:
        """Estimate tokens for a conversation with multiple messages."""
        total_tokens = 0
        
        for message in messages:
            content = message.get('content', '')
            role = message.get('role', '')
            
            # Add content tokens
            content_estimate = self.estimate_tokens(content, provider)
            total_tokens += content_estimate.estimated_tokens
            
            # Add overhead for message formatting
            total_tokens += 3  # Rough estimate for role and formatting tokens
        
        return total_tokens


class ContextMonitor:
    """Monitors context usage and provides warnings."""
    
    def __init__(self):
        """Initialize context monitor."""
        self.token_estimator = TokenEstimator()
        self.provider_limits = {
            # Context limits for different models (in tokens)
            'claude-3-opus-20240229': 200000,
            'claude-3-sonnet-20240229': 200000,
            'claude-3-haiku-20240307': 200000,
            'gpt-4': 8192,
            'gpt-4-32k': 32768,
            'gpt-4-turbo-preview': 128000,
            'gpt-3.5-turbo': 4096,
            'gpt-3.5-turbo-16k': 16384,
            'gemini-pro': 32768,
            'gemini-pro-vision': 16384
        }
        
        # Warning thresholds (as percentage of context limit)
        self.warning_thresholds = {
            ContextWarningLevel.SAFE: 0.0,
            ContextWarningLevel.CAUTION: 0.6,
            ContextWarningLevel.WARNING: 0.75,
            ContextWarningLevel.CRITICAL: 0.9
        }
        
        # Current context tracking
        self.context_usage: Dict[str, ContextUsage] = {}
        self.total_conversation_tokens = 0
    
    def get_context_limit(self, model: str) -> int:
        """Get context limit for a model."""
        return self.provider_limits.get(model, 8192)  # Conservative default
    
    def _get_warning_level(self, usage_percentage: float) -> ContextWarningLevel:
        """Determine warning level based on usage percentage."""
        if usage_percentage >= self.warning_thresholds[ContextWarningLevel.CRITICAL]:
            return ContextWarningLevel.CRITICAL
        elif usage_percentage >= self.warning_thresholds[ContextWarningLevel.WARNING]:
            return ContextWarningLevel.WARNING
        elif usage_percentage >= self.warning_thresholds[ContextWarningLevel.CAUTION]:
            return ContextWarningLevel.CAUTION
        else:
            return ContextWarningLevel.SAFE
    
    def estimate_request_tokens(self, request: LLMRequest, provider: str, model: str) -> int:
        """Estimate total tokens for a request."""
        # Estimate system prompt tokens
        system_tokens = self.token_estimator.estimate_tokens(
            request.system_prompt or "", provider
        ).estimated_tokens
        
        # Estimate user prompt tokens
        user_tokens = self.token_estimator.estimate_tokens(
            request.prompt or "", provider
        ).estimated_tokens
        
        # Add small overhead for message formatting
        overhead = 10
        
        return system_tokens + user_tokens + overhead
    
    def update_context_usage(self, provider: str, model: str, request: LLMRequest, 
                           response: Optional[LLMResponse] = None) -> ContextUsage:
        """
        Update context usage tracking for a provider.
        
        Args:
            provider: Provider name
            model: Model name
            request: Request being sent
            response: Response received (optional)
            
        Returns:
            Updated ContextUsage information
        """
        context_limit = self.get_context_limit(model)
        
        # Estimate tokens for this request
        request_tokens = self.estimate_request_tokens(request, provider, model)
        
        # Add response tokens if available
        response_tokens = 0
        if response and hasattr(response, 'tokens_used') and response.tokens_used:
            response_tokens = response.tokens_used
        elif response:
            # Estimate response tokens
            response_estimate = self.token_estimator.estimate_tokens(
                response.content or "", provider
            )
            response_tokens = response_estimate.estimated_tokens
        
        # Calculate total usage
        total_tokens = request_tokens + response_tokens
        
        # Update tracking
        if provider in self.context_usage:
            # Add to existing usage
            self.context_usage[provider].current_usage += total_tokens
        else:
            # Initialize new tracking
            self.context_usage[provider] = ContextUsage(
                provider=provider,
                model=model,
                context_limit=context_limit,
                current_usage=total_tokens,
                warning_level=ContextWarningLevel.SAFE,
                percentage_used=0.0,
                remaining_tokens=context_limit
            )
        
        # Update calculated fields
        usage = self.context_usage[provider]
        usage.percentage_used = (usage.current_usage / usage.context_limit) * 100
        usage.remaining_tokens = usage.context_limit - usage.current_usage
        usage.warning_level = self._get_warning_level(usage.percentage_used / 100)
        
        # Update total conversation tokens
        self.total_conversation_tokens += total_tokens
        
        return usage
    
    def check_context_before_request(self, provider: str, model: str, 
                                   request: LLMRequest) -> Tuple[bool, ContextUsage, Optional[str]]:
        """
        Check if request will exceed context limits.
        
        Args:
            provider: Provider name
            model: Model name  
            request: Request to check
            
        Returns:
            Tuple of (can_proceed, usage_info, warning_message)
        """
        context_limit = self.get_context_limit(model)
        request_tokens = self.estimate_request_tokens(request, provider, model)
        
        # Get current usage
        current_usage = self.context_usage.get(provider)
        if current_usage:
            total_estimated = current_usage.current_usage + request_tokens
        else:
            total_estimated = request_tokens
        
        # Reserve tokens for response (estimate)
        response_reserve = min(2000, context_limit * 0.1)  # 10% or 2000 tokens
        available_tokens = context_limit - response_reserve
        
        # Create usage info
        percentage = (total_estimated / context_limit) * 100
        warning_level = self._get_warning_level(percentage / 100)
        
        usage_info = ContextUsage(
            provider=provider,
            model=model,
            context_limit=context_limit,
            current_usage=total_estimated,
            warning_level=warning_level,
            percentage_used=percentage,
            remaining_tokens=context_limit - total_estimated
        )
        
        # Check if we can proceed
        can_proceed = total_estimated <= available_tokens
        warning_message = None
        
        if not can_proceed:
            warning_message = (
                f"Context limit exceeded for {model}: "
                f"{total_estimated:,} tokens (limit: {context_limit:,})"
            )
        elif warning_level == ContextWarningLevel.CRITICAL:
            warning_message = (
                f"Critical context usage for {model}: "
                f"{percentage:.1f}% ({total_estimated:,}/{context_limit:,} tokens)"
            )
        elif warning_level == ContextWarningLevel.WARNING:
            warning_message = (
                f"High context usage for {model}: "
                f"{percentage:.1f}% ({total_estimated:,}/{context_limit:,} tokens)"
            )
        elif warning_level == ContextWarningLevel.CAUTION:
            warning_message = (
                f"Moderate context usage for {model}: "
                f"{percentage:.1f}% ({total_estimated:,}/{context_limit:,} tokens)"
            )
        
        return can_proceed, usage_info, warning_message
    
    def get_usage_summary(self) -> Dict[str, any]:
        """Get a summary of context usage across all providers."""
        summary = {
            'total_conversation_tokens': self.total_conversation_tokens,
            'providers': {},
            'highest_usage_percentage': 0.0,
            'most_constrained_provider': None,
            'warnings': []
        }
        
        for provider, usage in self.context_usage.items():
            summary['providers'][provider] = {
                'model': usage.model,
                'current_usage': usage.current_usage,
                'context_limit': usage.context_limit,
                'percentage_used': usage.percentage_used,
                'remaining_tokens': usage.remaining_tokens,
                'warning_level': usage.warning_level.value
            }
            
            # Track highest usage
            if usage.percentage_used > summary['highest_usage_percentage']:
                summary['highest_usage_percentage'] = usage.percentage_used
                summary['most_constrained_provider'] = provider
            
            # Collect warnings
            if usage.warning_level != ContextWarningLevel.SAFE:
                summary['warnings'].append({
                    'provider': provider,
                    'model': usage.model,
                    'level': usage.warning_level.value,
                    'percentage': usage.percentage_used,
                    'message': f"{provider}: {usage.percentage_used:.1f}% context usage"
                })
        
        return summary
    
    def should_summarize(self, provider: str = None) -> bool:
        """
        Determine if context should be summarized before next request.
        
        Args:
            provider: Specific provider to check, or None for any provider
            
        Returns:
            True if summarization is recommended
        """
        if provider:
            usage = self.context_usage.get(provider)
            return usage and usage.warning_level == ContextWarningLevel.CRITICAL
        
        # Check if any provider is at critical level
        return any(
            usage.warning_level == ContextWarningLevel.CRITICAL
            for usage in self.context_usage.values()
        )
    
    def get_summarization_candidates(self) -> List[str]:
        """Get list of providers that should consider summarization."""
        candidates = []
        for provider, usage in self.context_usage.items():
            if usage.warning_level in [ContextWarningLevel.WARNING, ContextWarningLevel.CRITICAL]:
                candidates.append(provider)
        return candidates
    
    def reset_provider_context(self, provider: str):
        """Reset context tracking for a provider (after summarization)."""
        if provider in self.context_usage:
            del self.context_usage[provider]
    
    def reset_all_context(self):
        """Reset all context tracking (new discussion)."""
        self.context_usage.clear()
        self.total_conversation_tokens = 0