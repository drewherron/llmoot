from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .llm_client import LLMRequest, LLMResponse, LLMError
from .provider_registry import ProviderRegistry
from .order_parser import order_parser, ExecutionStep
from .prompt_builder import PromptBuilder, ResponseContext
from .config import Config


@dataclass
class DiscussionResult:
    """Result of a complete discussion roundtable."""
    final_response: str
    responses: List[LLMResponse]
    execution_steps: List[ExecutionStep]
    success: bool
    error_message: Optional[str] = None


class DiscussionOrchestrator:
    """Orchestrates multi-LLM roundtable discussions."""
    
    def __init__(self, config: Config):
        self.config = config
        self.provider_registry = ProviderRegistry(config)
        self.prompt_builder = PromptBuilder()
        
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
            
            if attribution:
                print("Attribution mode: ON")
            
            # Track responses for context building
            responses = []
            response_contexts = []
            
            # Execute each step sequentially
            for i, step in enumerate(execution_steps, 1):
                print(f"Step {i}/{len(execution_steps)}: Querying {step.provider_name.title()}...")
                
                try:
                    # Get the appropriate client
                    client = self.provider_registry.get_client(step.provider_name)
                    
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
                        step_info, attribution
                    )
                    
                    # Get model for this provider and quality level
                    model = self.config.get_model(step.provider_name, quality_level)
                    
                    # Create and send request
                    request = LLMRequest(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        model=model,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    
                    response = client.send_request(request)
                    responses.append(response)
                    
                    # Add to context for next iterations
                    response_contexts.append(ResponseContext(
                        provider=step.provider_name,
                        content=response.content,
                        step_number=i
                    ))
                    
                    print(f"  Response received ({len(response.content)} chars)")
                    
                except LLMError as e:
                    error_msg = f"Error in step {i} ({step.provider_name}): {e}"
                    print(f"  {error_msg}")
                    return DiscussionResult(
                        final_response="",
                        responses=responses,
                        execution_steps=execution_steps,
                        success=False,
                        error_message=error_msg
                    )
                except Exception as e:
                    error_msg = f"Unexpected error in step {i} ({step.provider_name}): {e}"
                    print(f"  {error_msg}")
                    return DiscussionResult(
                        final_response="",
                        responses=responses,
                        execution_steps=execution_steps,
                        success=False,
                        error_message=error_msg
                    )
            
            # Return successful result
            final_response = responses[-1].content if responses else ""
            
            print(f"\nDiscussion completed successfully!")
            print(f"Final response from {execution_steps[-1].provider_name.title()}: {len(final_response)} characters")
            
            return DiscussionResult(
                final_response=final_response,
                responses=responses,
                execution_steps=execution_steps,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Failed to run discussion: {e}"
            print(error_msg)
            return DiscussionResult(
                final_response="",
                responses=[],
                execution_steps=[],
                success=False,
                error_message=error_msg
            )