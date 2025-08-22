from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from .llm_client import LLMRequest, LLMResponse, LLMError
from .provider_registry import ProviderRegistry
from .order_parser import order_parser, ExecutionStep
from .prompt_builder import PromptBuilder, ResponseContext
from .config import Config
from .error_handler import ErrorRecovery, ErrorReporter, ErrorContext, error_handling_context
from .model_manager import model_manager
from .logging_system import logging_manager, DiscussionLogger
from .context_monitor import ContextMonitor, ContextWarningLevel


@dataclass
class DiscussionResult:
    """Result of a complete discussion roundtable."""
    final_response: str
    responses: List[LLMResponse]
    execution_steps: List[ExecutionStep]
    success: bool
    error_message: Optional[str] = None
    error_summary: Optional[str] = None
    partial_success: bool = False
    context_usage_summary: Optional[Dict[str, Any]] = None


class DiscussionOrchestrator:
    """Orchestrates multi-LLM roundtable discussions."""
    
    def __init__(self, config: Config, enable_logging: bool = True, enable_console_logging: bool = False):
        self.config = config
        self.provider_registry = ProviderRegistry()
        self.prompt_builder = PromptBuilder(config)
        self.error_recovery = ErrorRecovery(config, model_manager)
        self.error_reporter = ErrorReporter()
        self.context_monitor = ContextMonitor()
        
        # Setup discussion logging
        self.enable_logging = enable_logging
        self.discussion_logger: Optional[DiscussionLogger] = None
        
        if enable_logging:
            logging_manager.configure(enable_console=enable_console_logging)
        
        # Setup basic Python logging
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
        
    def run_discussion(self, prompt: str, order: str, quality_level: int = 3, 
                      attribution: bool = False) -> DiscussionResult:
        """
        Run a complete discussion with the specified LLMs.
        
        Args:
            prompt: The initial discussion prompt
            order: Provider order string (e.g., "aog")  
            quality_level: Quality level 1-5 for model selection
            attribution: Whether to include model attribution
            
        Returns:
            DiscussionResult with final response and metadata
        """
        try:
            # Parse execution order
            execution_steps = order_parser.parse_order(order)
            print(f"Discussion order: {' -> '.join(step.provider_name for step in execution_steps)}")
            
            # Initialize logging for this discussion
            if self.enable_logging:
                self.discussion_logger = logging_manager.start_discussion_logging()
                self.discussion_logger.log_discussion_start(
                    prompt, order, quality_level, execution_steps, attribution
                )
            
            if attribution:
                print("Attribution mode: ON")
            
            # Track responses for context building
            responses = []
            response_contexts = []
            
            # Execute each step sequentially
            for i, step in enumerate(execution_steps, 1):
                print(f"Step {i}/{len(execution_steps)}: Querying {step.provider_name.title()}...")
                
                # Get model for this provider and quality level
                model = self.config.get_model(step.provider_name, quality_level)
                
                # Build the prompt with accumulated context
                step_info = {
                    'step_number': i,
                    'total_steps': len(execution_steps),
                    'is_final': i == len(execution_steps)
                }
                
                user_prompt = self.prompt_builder.build_user_prompt(
                    prompt, response_contexts, step_info, attribution
                )
                
                system_prompt = self.prompt_builder.build_system_prompt(
                    step.is_final, prompt, step_info, attribution
                )
                
                # Create request
                request = LLMRequest(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # Check context usage before request
                can_proceed, usage_info, warning_msg = self.context_monitor.check_context_before_request(
                    step.provider_name, model, request
                )
                
                # Display context warnings
                if warning_msg:
                    if usage_info.warning_level == ContextWarningLevel.CRITICAL:
                        print(f"  ⚠️ {warning_msg}")
                    elif usage_info.warning_level == ContextWarningLevel.WARNING:
                        print(f"  ⚠️ {warning_msg}")
                
                # Log the request
                if self.discussion_logger:
                    self.discussion_logger.log_request(i, step.provider_name, model, request, step)
                
                # Attempt to get response with error handling
                response = self._execute_step_with_recovery(
                    step, request, i, len(execution_steps), quality_level
                )
                
                if response is None:
                    # Critical error - cannot continue
                    error_summary = self.error_reporter.generate_error_summary()
                    return DiscussionResult(
                        final_response="",
                        responses=responses,
                        execution_steps=execution_steps,
                        success=False,
                        error_message="Discussion aborted due to critical errors",
                        error_summary=error_summary
                    )
                
                responses.append(response)
                
                # Update context monitoring
                updated_usage = self.context_monitor.update_context_usage(
                    step.provider_name, model, request, response
                )
                
                # Log the response
                if self.discussion_logger:
                    self.discussion_logger.log_response(i, step.provider_name, response)
                
                # Add to context for next iterations (even if it's a placeholder)
                response_contexts.append(ResponseContext(
                    provider=step.provider_name,
                    model=model,
                    content=response.content,
                    step_number=i,
                    tokens_used=response.tokens_used
                ))
                
                if response.error:
                    print(f"  Response received with recovery ({len(response.content)} chars)")
                else:
                    print(f"  Response received ({len(response.content)} chars)")
                
                # Show context usage info
                if updated_usage.warning_level != ContextWarningLevel.SAFE:
                    print(f"  Context usage: {updated_usage.percentage_used:.1f}% ({updated_usage.current_usage:,}/{updated_usage.context_limit:,} tokens)")
            
            # Return successful result
            final_response = responses[-1].content if responses else ""
            error_summary = self.error_reporter.generate_error_summary()
            has_errors = len(self.error_reporter.errors_encountered) > 0
            
            if has_errors:
                print(f"\nDiscussion completed with some errors!")
                print(f"Final response from {execution_steps[-1].provider_name.title()}: {len(final_response)} characters")
            else:
                print(f"\nDiscussion completed successfully!")
                print(f"Final response from {execution_steps[-1].provider_name.title()}: {len(final_response)} characters")
            
            # Log discussion completion
            if self.discussion_logger:
                self.discussion_logger.log_discussion_end(
                    success=True,
                    final_response=final_response,
                    total_responses=len(responses),
                    partial_success=has_errors
                )
            
            # Show context usage summary if any warnings occurred
            context_summary = self.context_monitor.get_usage_summary()
            if context_summary['warnings']:
                print(f"\nContext Usage Summary:")
                print(f"  Total conversation tokens: {context_summary['total_conversation_tokens']:,}")
                for warning in context_summary['warnings']:
                    level_symbol = "🔴" if warning['level'] == 'critical' else "🟡"
                    print(f"  {level_symbol} {warning['message']}")
            
            return DiscussionResult(
                final_response=final_response,
                responses=responses,
                execution_steps=execution_steps,
                success=True,
                error_summary=error_summary if has_errors else None,
                partial_success=has_errors,
                context_usage_summary=context_summary
            )
            
        except Exception as e:
            error_msg = f"Failed to run discussion: {e}"
            print(error_msg)
            error_summary = self.error_reporter.generate_error_summary()
            
            # Log discussion failure
            if self.discussion_logger:
                self.discussion_logger.log_discussion_end(
                    success=False,
                    final_response="",
                    total_responses=0,
                    error_message=error_msg
                )
            
            return DiscussionResult(
                final_response="",
                responses=[],
                execution_steps=[],
                success=False,
                error_message=error_msg,
                error_summary=error_summary
            )
    
    def _execute_step_with_recovery(self, step: ExecutionStep, request: LLMRequest, 
                                   step_number: int, total_steps: int, quality_level: int = 1) -> Optional[LLMResponse]:
        """
        Execute a single step with error recovery.
        
        Args:
            step: Execution step to run
            request: LLM request
            step_number: Current step number
            total_steps: Total number of steps
            
        Returns:
            LLMResponse or None if recovery failed
        """
        model = self.config.get_model(step.provider_name, 1)
        
        with error_handling_context(
            step.provider_name, model, step_number, total_steps, request
        ) as context:
            
            try:
                # Get the appropriate client
                client = self.provider_registry.create_client(step.provider_name, self.config, quality_level)
                
                # Attempt to send request
                response = client.generate_response(request)
                return response
                
            except Exception as e:
                # Log the error
                if self.discussion_logger:
                    self.discussion_logger.log_error(
                        step_number, step.provider_name, e,
                        recovery_attempted=True
                    )
                
                # Use error recovery system
                def client_factory(provider_name):
                    return self.provider_registry.create_client(provider_name, self.config, quality_level)
                
                recovery_result = self.error_recovery.handle_error(
                    e, context, client_factory, request
                )
                
                # Report the error
                self.error_reporter.report_error(e, context, recovery_result)
                
                if recovery_result.success and recovery_result.response:
                    return recovery_result.response
                elif recovery_result.should_continue:
                    # Return placeholder response to continue discussion
                    return LLMResponse(
                        content=f"[Step {step_number} skipped: {recovery_result.error_message}]",
                        provider=step.provider_name,
                        model=model,
                        tokens_used=0,
                        response_time=0.0,
                        is_final=step.is_final,
                        error=recovery_result.error_message
                    )
                else:
                    # Critical error - abort discussion
                    return None