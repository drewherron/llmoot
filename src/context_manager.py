"""Context length management with automatic summarization."""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re

from .llm_client import LLMClient, LLMRequest, LLMResponse
from .context_monitor import ContextMonitor, ContextWarningLevel
from .prompt_builder import ResponseContext
from .config import Config


@dataclass
class SummarizationResult:
    """Result of a summarization operation."""
    original_content: str
    summarized_content: str
    original_tokens: int
    summarized_tokens: int
    compression_ratio: float
    model_used: str
    success: bool
    error_message: Optional[str] = None


class ContextSummarizer:
    """Handles context summarization using configurable models."""
    
    def __init__(self, config: Config):
        """Initialize the context summarizer."""
        self.config = config
        
        # Get summarization model from config
        self.summarization_model = config.get('summarization.model', 'gpt-3.5-turbo')
        self.summarization_provider = self._get_provider_from_model(self.summarization_model)
        
        # Summarization settings
        self.max_summary_tokens = config.get('summarization.max_tokens', 1000)
        self.preserve_recent_responses = config.get('summarization.preserve_recent', 2)
        
    def _get_provider_from_model(self, model: str) -> str:
        """Determine provider from model name."""
        if 'claude' in model.lower():
            return 'claude'
        elif 'gpt' in model.lower() or 'openai' in model.lower():
            return 'openai'
        elif 'gemini' in model.lower():
            return 'gemini'
        else:
            return 'openai'  # Default fallback
    
    def _create_summarization_prompt(self, responses: List[ResponseContext], 
                                   original_prompt: str, attribution: bool = False) -> str:
        """Create a prompt for summarizing conversation context."""
        
        # Get summarization prompt template from config
        custom_prompt = self.config.get('summarization.prompt')
        
        if custom_prompt:
            base_prompt = custom_prompt
        else:
            base_prompt = """Please summarize the following AI discussion responses, preserving the key insights and arguments while removing redundancy. Keep the essential points that would be important for the next AI to consider:

Original Question: {original_prompt}

Responses to summarize:
{responses}

Create a concise summary that:
1. Preserves the main arguments and insights from each response
2. Removes redundant information and repetitive points  
3. Maintains the logical flow of the discussion
4. Keeps key evidence, examples, and specific details that matter
5. Notes any disagreements or alternative perspectives
6. Stays focused on information relevant to the original question

Summary:"""
        
        # Format responses for summarization
        response_text = ""
        for i, response in enumerate(responses, 1):
            if attribution:
                provider_name = {
                    'claude': 'Claude',
                    'openai': 'ChatGPT', 
                    'gemini': 'Gemini'
                }.get(response.provider, response.provider.title())
                
                response_text += f"\n{provider_name} (Response {i}):\n{response.content}\n"
            else:
                response_text += f"\nResponse {i}:\n{response.content}\n"
        
        return base_prompt.format(
            original_prompt=original_prompt,
            responses=response_text.strip()
        )
    
    def summarize_context(self, responses: List[ResponseContext], original_prompt: str,
                         client_factory, attribution: bool = False) -> SummarizationResult:
        """
        Summarize a list of response contexts.
        
        Args:
            responses: List of ResponseContext objects to summarize
            original_prompt: Original user prompt for context
            client_factory: Function to create LLM clients
            attribution: Whether to include provider attribution
            
        Returns:
            SummarizationResult with summarization outcome
        """
        try:
            # Calculate original token count
            original_content = "\n".join([r.content for r in responses])
            original_tokens = sum([r.tokens_used for r in responses if hasattr(r, 'tokens_used')])
            
            # Create summarization prompt
            summary_prompt = self._create_summarization_prompt(responses, original_prompt, attribution)
            
            # Create request for summarization
            request = LLMRequest(
                prompt=summary_prompt,
                system_prompt="You are a helpful AI assistant skilled at creating concise, accurate summaries.",
                temperature=0.3,  # Lower temperature for more focused summaries
                max_tokens=self.max_summary_tokens
            )
            
            # Get summarization client
            client = client_factory(self.summarization_provider)
            
            # Generate summary
            response = client.generate_response(request)
            
            if response.error:
                return SummarizationResult(
                    original_content=original_content,
                    summarized_content="",
                    original_tokens=original_tokens,
                    summarized_tokens=0,
                    compression_ratio=0.0,
                    model_used=self.summarization_model,
                    success=False,
                    error_message=response.error
                )
            
            # Calculate compression metrics
            summarized_tokens = response.tokens_used or len(response.content) // 4
            compression_ratio = summarized_tokens / max(original_tokens, 1)
            
            return SummarizationResult(
                original_content=original_content,
                summarized_content=response.content,
                original_tokens=original_tokens,
                summarized_tokens=summarized_tokens,
                compression_ratio=compression_ratio,
                model_used=self.summarization_model,
                success=True
            )
            
        except Exception as e:
            return SummarizationResult(
                original_content=original_content if 'original_content' in locals() else "",
                summarized_content="",
                original_tokens=original_tokens if 'original_tokens' in locals() else 0,
                summarized_tokens=0,
                compression_ratio=0.0,
                model_used=self.summarization_model,
                success=False,
                error_message=str(e)
            )


class ContextManager:
    """Manages context length with automatic summarization."""
    
    def __init__(self, config: Config, context_monitor: ContextMonitor):
        """Initialize the context manager."""
        self.config = config
        self.context_monitor = context_monitor
        self.summarizer = ContextSummarizer(config)
        
        # Context management settings
        self.summarization_threshold = config.get('summarization.threshold', 0.75)  # 75%
        self.min_responses_to_summarize = config.get('summarization.min_responses', 3)
        self.preserve_recent_count = config.get('summarization.preserve_recent', 2)
        
        # Track summarizations performed
        self.summarizations_performed = []
    
    def should_summarize_for_provider(self, provider: str, model: str, 
                                    upcoming_request: LLMRequest) -> bool:
        """
        Check if context should be summarized for a specific provider before making a request.
        
        Args:
            provider: Provider name
            model: Model name
            upcoming_request: The request about to be sent
            
        Returns:
            True if summarization is recommended
        """
        # Check if provider is approaching context limits
        can_proceed, usage_info, _ = self.context_monitor.check_context_before_request(
            provider, model, upcoming_request
        )
        
        # If we can't proceed at all, definitely need summarization
        if not can_proceed:
            return True
        
        # If we're above the summarization threshold, recommend summarization
        threshold_percentage = self.summarization_threshold * 100
        if usage_info.percentage_used >= threshold_percentage:
            return True
        
        # If context monitor suggests summarization
        if self.context_monitor.should_summarize(provider):
            return True
        
        return False
    
    def summarize_conversation_context(self, responses: List[ResponseContext], 
                                     original_prompt: str, client_factory,
                                     attribution: bool = False) -> Tuple[List[ResponseContext], SummarizationResult]:
        """
        Summarize conversation context while preserving recent responses.
        
        Args:
            responses: List of all response contexts
            original_prompt: Original user prompt
            client_factory: Function to create LLM clients
            attribution: Whether to include provider attribution
            
        Returns:
            Tuple of (updated_responses, summarization_result)
        """
        if len(responses) < self.min_responses_to_summarize:
            # Not enough responses to warrant summarization
            return responses, SummarizationResult(
                original_content="",
                summarized_content="",
                original_tokens=0,
                summarized_tokens=0,
                compression_ratio=0.0,
                model_used="",
                success=False,
                error_message="Insufficient responses for summarization"
            )
        
        # Split responses into those to summarize and those to preserve
        preserve_count = min(self.preserve_recent_count, len(responses) - 1)
        responses_to_summarize = responses[:-preserve_count] if preserve_count > 0 else responses
        responses_to_preserve = responses[-preserve_count:] if preserve_count > 0 else []
        
        # Summarize the older responses
        summarization_result = self.summarizer.summarize_context(
            responses_to_summarize, original_prompt, client_factory, attribution
        )
        
        if not summarization_result.success:
            return responses, summarization_result
        
        # Create a summary response context
        summary_context = ResponseContext(
            provider="summarizer",
            model=summarization_result.model_used,
            content=summarization_result.summarized_content,
            step_number=0,  # Special marker for summary
            tokens_used=summarization_result.summarized_tokens
        )
        
        # Combine summary with preserved recent responses
        updated_responses = [summary_context] + responses_to_preserve
        
        # Track this summarization
        self.summarizations_performed.append({
            'original_count': len(responses_to_summarize),
            'preserved_count': len(responses_to_preserve),
            'compression_ratio': summarization_result.compression_ratio,
            'tokens_saved': summarization_result.original_tokens - summarization_result.summarized_tokens
        })
        
        return updated_responses, summarization_result
    
    def get_context_management_summary(self) -> Dict:
        """Get summary of context management operations performed."""
        total_summarizations = len(self.summarizations_performed)
        total_tokens_saved = sum(s['tokens_saved'] for s in self.summarizations_performed)
        
        return {
            'total_summarizations': total_summarizations,
            'total_tokens_saved': total_tokens_saved,
            'summarizations_performed': self.summarizations_performed,
            'summarization_model': self.summarizer.summarization_model,
            'threshold_percentage': self.summarization_threshold * 100
        }
    
    def reset_tracking(self):
        """Reset context management tracking (for new discussions)."""
        self.summarizations_performed.clear()
        self.context_monitor.reset_all_context()