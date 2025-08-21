"""Error handling and recovery mechanisms."""

import time
import logging
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

from .llm_client import LLMError, LLMResponse, LLMRequest


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"           # Minor issues, retry recommended
    MEDIUM = "medium"     # Significant issues, fallback recommended  
    HIGH = "high"         # Critical issues, abort step
    FATAL = "fatal"       # System-wide failure, abort discussion


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"                    # Retry the same request
    FALLBACK_MODEL = "fallback_model"  # Use fallback model
    SKIP_PROVIDER = "skip_provider"    # Skip this provider
    PARTIAL_CONTINUE = "partial_continue"  # Continue with partial results
    ABORT = "abort"                    # Abort the discussion


@dataclass
class ErrorContext:
    """Context information for an error."""
    provider: str
    model: str
    step_number: int
    total_steps: int
    request: Optional[LLMRequest] = None
    original_error: Optional[Exception] = None
    retry_count: int = 0
    max_retries: int = 3
    elapsed_time: float = 0.0


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    strategy_used: RecoveryStrategy
    response: Optional[LLMResponse] = None
    error_message: Optional[str] = None
    should_continue: bool = True
    fallback_provider: Optional[str] = None


class ErrorClassifier:
    """Classifies errors and determines appropriate responses."""
    
    def __init__(self):
        """Initialize the error classifier."""
        self.error_patterns = {
            # API key/authentication errors
            'auth': {
                'patterns': ['unauthorized', 'invalid_api_key', 'authentication', 'forbidden'],
                'severity': ErrorSeverity.HIGH,
                'strategy': RecoveryStrategy.SKIP_PROVIDER
            },
            # Rate limiting errors
            'rate_limit': {
                'patterns': ['rate_limit', 'too_many_requests', 'quota_exceeded'],
                'severity': ErrorSeverity.MEDIUM,
                'strategy': RecoveryStrategy.RETRY
            },
            # Network/connectivity errors
            'network': {
                'patterns': ['connection', 'timeout', 'network', 'dns'],
                'severity': ErrorSeverity.LOW,
                'strategy': RecoveryStrategy.RETRY
            },
            # Content policy errors
            'content_policy': {
                'patterns': ['content_policy', 'safety', 'blocked', 'filtered'],
                'severity': ErrorSeverity.MEDIUM,
                'strategy': RecoveryStrategy.FALLBACK_MODEL
            },
            # Context length errors
            'context_length': {
                'patterns': ['context_length', 'too_long', 'token_limit'],
                'severity': ErrorSeverity.MEDIUM,
                'strategy': RecoveryStrategy.PARTIAL_CONTINUE
            },
            # Server errors
            'server': {
                'patterns': ['internal_error', 'server_error', '500', '502', '503'],
                'severity': ErrorSeverity.LOW,
                'strategy': RecoveryStrategy.RETRY
            }
        }
    
    def classify_error(self, error: Exception, context: ErrorContext) -> Tuple[ErrorSeverity, RecoveryStrategy]:
        """
        Classify an error and determine recovery strategy.
        
        Args:
            error: The exception that occurred
            context: Error context information
            
        Returns:
            Tuple of (severity, recovery_strategy)
        """
        error_message = str(error).lower()
        
        # Check for known error patterns
        for error_type, config in self.error_patterns.items():
            for pattern in config['patterns']:
                if pattern in error_message:
                    return config['severity'], config['strategy']
        
        # Check retry count to escalate severity
        if context.retry_count >= context.max_retries:
            if context.step_number == context.total_steps:
                # Last step failure is more critical
                return ErrorSeverity.HIGH, RecoveryStrategy.ABORT
            else:
                return ErrorSeverity.MEDIUM, RecoveryStrategy.SKIP_PROVIDER
        
        # Default: treat as temporary issue
        return ErrorSeverity.LOW, RecoveryStrategy.RETRY


class ErrorRecovery:
    """Handles error recovery using various strategies."""
    
    def __init__(self, config, model_manager):
        """Initialize error recovery system."""
        self.config = config
        self.model_manager = model_manager
        self.classifier = ErrorClassifier()
        
        # Retry settings
        self.default_retry_delays = [1, 2, 4, 8]  # Exponential backoff
        self.max_retry_time = 30  # Maximum time to spend retrying
    
    def handle_error(self, error: Exception, context: ErrorContext, 
                    client_factory: Callable, request: LLMRequest) -> RecoveryResult:
        """
        Handle an error with appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            context: Error context
            client_factory: Function to create LLM clients
            request: Original request
            
        Returns:
            RecoveryResult with outcome
        """
        severity, strategy = self.classifier.classify_error(error, context)
        
        logging.warning(
            f"Error in step {context.step_number}/{context.total_steps} "
            f"({context.provider}): {error} - Strategy: {strategy.value}"
        )
        
        if strategy == RecoveryStrategy.RETRY:
            return self._retry_request(error, context, client_factory, request)
        
        elif strategy == RecoveryStrategy.FALLBACK_MODEL:
            return self._fallback_model(error, context, client_factory, request)
        
        elif strategy == RecoveryStrategy.SKIP_PROVIDER:
            return self._skip_provider(error, context)
        
        elif strategy == RecoveryStrategy.PARTIAL_CONTINUE:
            return self._partial_continue(error, context)
        
        elif strategy == RecoveryStrategy.ABORT:
            return self._abort_discussion(error, context)
        
        # Fallback to retry if no specific strategy matched
        return self._retry_request(error, context, client_factory, request)
    
    def _retry_request(self, error: Exception, context: ErrorContext,
                      client_factory: Callable, request: LLMRequest) -> RecoveryResult:
        """Retry the original request with exponential backoff."""
        if context.retry_count >= context.max_retries:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RETRY,
                error_message=f"Max retries ({context.max_retries}) exceeded for {context.provider}",
                should_continue=context.step_number < context.total_steps
            )
        
        # Calculate delay with exponential backoff
        delay_index = min(context.retry_count, len(self.default_retry_delays) - 1)
        delay = self.default_retry_delays[delay_index]
        
        print(f"  Retrying in {delay}s... (attempt {context.retry_count + 1}/{context.max_retries})")
        time.sleep(delay)
        
        try:
            client = client_factory(context.provider)
            response = client.generate_response(request)
            
            print(f"  Retry successful after {context.retry_count + 1} attempts")
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.RETRY,
                response=response
            )
        
        except Exception as retry_error:
            context.retry_count += 1
            context.elapsed_time += delay
            
            if context.retry_count >= context.max_retries or context.elapsed_time >= self.max_retry_time:
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.RETRY,
                    error_message=f"Retry failed for {context.provider}: {retry_error}",
                    should_continue=context.step_number < context.total_steps
                )
            
            # Recursive retry
            return self._retry_request(retry_error, context, client_factory, request)
    
    def _fallback_model(self, error: Exception, context: ErrorContext,
                       client_factory: Callable, request: LLMRequest) -> RecoveryResult:
        """Try using a fallback model for the same provider."""
        try:
            # Get fallback model from model manager
            current_model = context.model
            quality_levels = [1, 2]  # Try different quality levels
            
            for quality_level in quality_levels:
                fallback_model = self.model_manager.get_model(context.provider, quality_level)
                
                if fallback_model and fallback_model != current_model:
                    print(f"  Trying fallback model: {fallback_model}")
                    
                    # Update request with fallback model
                    fallback_request = LLMRequest(
                        prompt=request.prompt,
                        system_prompt=request.system_prompt,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens
                    )
                    
                    client = client_factory(context.provider)
                    response = client.generate_response(fallback_request)
                    
                    print(f"  Fallback successful with {fallback_model}")
                    return RecoveryResult(
                        success=True,
                        strategy_used=RecoveryStrategy.FALLBACK_MODEL,
                        response=response
                    )
            
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.FALLBACK_MODEL,
                error_message=f"No fallback models available for {context.provider}",
                should_continue=context.step_number < context.total_steps
            )
        
        except Exception as fallback_error:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.FALLBACK_MODEL,
                error_message=f"Fallback failed: {fallback_error}",
                should_continue=context.step_number < context.total_steps
            )
    
    def _skip_provider(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Skip this provider and continue with the next one."""
        if context.step_number == context.total_steps:
            # This is the final step - we need some response
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.SKIP_PROVIDER,
                error_message=f"Cannot skip final provider {context.provider}",
                should_continue=False
            )
        
        print(f"  Skipping provider {context.provider} due to: {error}")
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.SKIP_PROVIDER,
            error_message=f"Skipped {context.provider}: {error}",
            should_continue=True
        )
    
    def _partial_continue(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Continue with partial results or a placeholder response."""
        placeholder_response = LLMResponse(
            content=f"[Response unavailable from {context.provider} due to: {str(error)[:100]}...]",
            provider=context.provider,
            model=context.model,
            tokens_used=0,
            response_time=0.0,
            is_final=context.step_number == context.total_steps,
            error=str(error)
        )
        
        print(f"  Continuing with placeholder response for {context.provider}")
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.PARTIAL_CONTINUE,
            response=placeholder_response,
            error_message=f"Using placeholder for {context.provider}"
        )
    
    def _abort_discussion(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Abort the entire discussion due to critical error."""
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.ABORT,
            error_message=f"Critical error in {context.provider}: {error}",
            should_continue=False
        )


@contextmanager
def error_handling_context(provider: str, model: str, step_number: int, 
                         total_steps: int, request: LLMRequest = None):
    """Context manager for consistent error handling."""
    start_time = time.time()
    context = ErrorContext(
        provider=provider,
        model=model,
        step_number=step_number,
        total_steps=total_steps,
        request=request
    )
    
    try:
        yield context
    except Exception as e:
        context.original_error = e
        context.elapsed_time = time.time() - start_time
        raise
    finally:
        if context.elapsed_time == 0:
            context.elapsed_time = time.time() - start_time


class ErrorReporter:
    """Handles error reporting and logging."""
    
    def __init__(self):
        """Initialize error reporter."""
        self.errors_encountered = []
        self.recovery_stats = {
            'total_errors': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'strategies_used': {}
        }
    
    def report_error(self, error: Exception, context: ErrorContext, 
                    recovery_result: RecoveryResult):
        """Report an error and its recovery result."""
        error_info = {
            'timestamp': time.time(),
            'provider': context.provider,
            'model': context.model,
            'step': f"{context.step_number}/{context.total_steps}",
            'error': str(error),
            'recovery_strategy': recovery_result.strategy_used.value,
            'recovery_success': recovery_result.success,
            'retry_count': context.retry_count
        }
        
        self.errors_encountered.append(error_info)
        self._update_stats(recovery_result)
    
    def _update_stats(self, recovery_result: RecoveryResult):
        """Update recovery statistics."""
        self.recovery_stats['total_errors'] += 1
        
        if recovery_result.success:
            self.recovery_stats['successful_recoveries'] += 1
        else:
            self.recovery_stats['failed_recoveries'] += 1
        
        strategy = recovery_result.strategy_used.value
        self.recovery_stats['strategies_used'][strategy] = \
            self.recovery_stats['strategies_used'].get(strategy, 0) + 1
    
    def generate_error_summary(self) -> str:
        """Generate a summary of errors encountered."""
        if not self.errors_encountered:
            return "No errors encountered during discussion."
        
        lines = [
            f"Error Summary ({len(self.errors_encountered)} errors encountered):",
            f"  Successful recoveries: {self.recovery_stats['successful_recoveries']}",
            f"  Failed recoveries: {self.recovery_stats['failed_recoveries']}",
            ""
        ]
        
        if self.recovery_stats['strategies_used']:
            lines.append("Recovery strategies used:")
            for strategy, count in self.recovery_stats['strategies_used'].items():
                lines.append(f"  {strategy}: {count}")
            lines.append("")
        
        if any(not error['recovery_success'] for error in self.errors_encountered):
            lines.append("Failed recoveries:")
            for error in self.errors_encountered:
                if not error['recovery_success']:
                    lines.append(f"  {error['provider']} (step {error['step']}): {error['error']}")
        
        return "\n".join(lines)