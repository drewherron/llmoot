#!/usr/bin/env python3
"""
llmoot - Multi-LLM Roundtable Discussion Tool
"""

import argparse
import os
import sys
from src.config import Config
from src.mock_responses import DEV_MODE, get_mock_client
from src.provider_registry import registry
from src.order_parser import order_parser
from src.prompt_builder import PromptBuilder
from src.discussion_orchestrator import DiscussionOrchestrator, DiscussionResult
from src.response_formatter import ResponseFormatter, OutputFormat
from src.llm_client import LLMResponse


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-LLM Roundtable Discussion Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llmoot.py -o aog "What is the meaning of life?"
  llmoot.py -o agoa --quality 2 "Explain quantum computing"
  llmoot.py -o aog prompt.txt

Order codes:
  a = Anthropic (Claude)
  o = OpenAI (ChatGPT)
  g = Google (Gemini)
        """
    )

    parser.add_argument(
        "prompt",
        help="The question/prompt for discussion, or path to file containing the prompt"
    )

    parser.add_argument(
        "-o", "--order",
        required=True,
        help="Order of LLM execution (e.g., 'aog' for Anthropic->OpenAI->Google)"
    )

    parser.add_argument(
        "--quality",
        type=int,
        choices=[1, 2],
        default=1,
        help="Quality level: 1=highest models, 2=mid-tier models (default: 1)"
    )

    parser.add_argument(
        "--attribution",
        action="store_true",
        help="Include model names in context and enable attribution in final response"
    )

    parser.add_argument(
        "--format",
        choices=["plain", "markdown", "json", "structured"],
        default="plain",
        help="Output format (default: plain)"
    )

    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Show all intermediate responses, not just final result"
    )

    parser.add_argument(
        "--export",
        help="Export results to specified file (format inferred from extension)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="llmoot 0.1.0"
    )

    return parser.parse_args()


def validate_order(order):
    """Validate the order string and return normalized version."""
    try:
        # Use the enhanced order parser for validation
        order_parser.validate_order(order)
        return order.lower().strip()
    except ValueError:
        # Re-raise with our error for consistency
        raise


def load_prompt(prompt_arg):
    """Load prompt from argument - either direct text or file path.
    
    Args:
        prompt_arg: Command line argument (text or file path)
        
    Returns:
        The prompt text
        
    Raises:
        ValueError: If file doesn't exist or can't be read
    """
    # Check if it's a file path
    if os.path.isfile(prompt_arg):
        try:
            with open(prompt_arg, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if not content:
                raise ValueError(f"Prompt file '{prompt_arg}' is empty")
            return content
        except IOError as e:
            raise ValueError(f"Cannot read prompt file '{prompt_arg}': {e}")
    
    # Otherwise treat as direct prompt text
    if not prompt_arg.strip():
        raise ValueError("Prompt cannot be empty")
    
    return prompt_arg.strip()



def run_mock_discussion_with_result(order, prompt, config, quality_level, attribution=False):
    """Run a mock roundtable discussion and return DiscussionResult object."""
    # Parse order into execution steps
    steps = order_parser.parse_order(order)
    responses = []
    llm_responses = []
    prompt_builder = PromptBuilder(config)
    
    for step in steps:
        provider = step.provider_name
        model = config.get_model(provider, quality_level)
        
        # Create mock client
        client = get_mock_client(provider, model)
        
        # Show progress
        status = "Final response" if step.is_final else f"Response {step.step_number}/{step.total_steps}"
        print(f"{provider.title()} ({model}) - {status}...")
        
        # Build enhanced context using prompt builder
        step_info = {
            'step_number': step.step_number,
            'total_steps': step.total_steps,
            'is_final': step.is_final
        }
        
        # Convert previous responses to context format
        response_contexts = []
        for resp in responses:
            response_contexts.append(type('ResponseContext', (), {
                'provider': resp['provider'],
                'model': resp.get('model', 'unknown'),
                'content': resp['content'],
                'step_number': len(response_contexts) + 1,
                'tokens_used': resp.get('tokens_used', 0)
            })())
        
        # Build system and user prompts
        system_prompt = prompt_builder.build_system_prompt(step.is_final, prompt, step_info, attribution)
        user_prompt = prompt_builder.build_user_prompt(prompt, response_contexts, step_info, attribution)
        
        # Generate response (mock client still uses simple context)
        context = f"System: {system_prompt}\n\nUser: {user_prompt}"
        response = client.generate_response(context, step.is_final)
        responses.append(response)
        
        # Create LLMResponse object for compatibility with formatter
        llm_response = LLMResponse(
            content=response['content'],
            model=response['model'],
            provider=response['provider'],
            tokens_used=response.get('tokens_used', 0),
            response_time=response.get('response_time', 0.0),
            is_final=response.get('is_final', False),
            raw_response={'mock': True}
        )
        llm_responses.append(llm_response)
        
        # Show the response
        print(f"  Response received ({len(response['content'])} chars)")
    
    print()
    print("Mock discussion processing complete!")
    
    # Return DiscussionResult object
    return DiscussionResult(
        final_response=responses[-1]['content'] if responses else "",
        responses=llm_responses,
        execution_steps=steps,
        success=True
    )


def main():
    try:
        args = parse_arguments()

        # Validate order
        order = validate_order(args.order)
        
        # Load prompt (from file or direct text)
        prompt = load_prompt(args.prompt)
        
        # Load configuration
        try:
            config = Config()
        except ValueError as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            return 1

        print(f"llmoot - Multi-LLM Roundtable Discussion Tool")
        if DEV_MODE:
            print("WARNING: Running in DEVELOPMENT MODE with mock responses")
        
        # Show prompt source
        if os.path.isfile(args.prompt):
            print(f"Prompt file: {args.prompt}")
        else:
            print(f"Prompt: {prompt}")
        
        # Show detailed order information
        order_analysis = order_parser.analyze_order(order)
        print(f"Order: {order} (quality level {args.quality})")
        print(f"Execution: {order_analysis['execution_summary']}")
        if order_analysis['has_duplicates']:
            counts = order_analysis['provider_counts']
            duplicates = [f"{code}×{count}" for code, count in counts.items() if count > 1]
            print(f"Note: Duplicate providers - {', '.join(duplicates)}")
        
        # Check API key availability (skip in dev mode)
        if not DEV_MODE:
            missing_providers = registry.get_missing_providers(order, config)
            
            if missing_providers:
                print(f"Error: Missing API keys for: {', '.join(missing_providers)}", file=sys.stderr)
                print("Set environment variables or create config.yaml from config.yaml.template", file=sys.stderr)
                return 1
        
        # Show selected models
        provider_map = {'a': 'claude', 'o': 'openai', 'g': 'gemini'}
        models = []
        for code in order:
            provider = provider_map[code]
            model = config.get_model(provider, args.quality)
            models.append(f"{code}={model}")
        print(f"Models: {' -> '.join(models)}")
        print()
        
        # Run discussion (mock or real based on DEV_MODE)
        if DEV_MODE:
            print("Starting mock roundtable discussion...")
            if args.attribution:
                print("Attribution mode: ON - model names will be included in context")
            
            # Create a mock result that can be formatted
            mock_result = run_mock_discussion_with_result(order, prompt, config, args.quality, args.attribution)
            
            # Format and display results using the same logic as production
            formatter = ResponseFormatter()
            output_format = OutputFormat(args.format)
            
            if mock_result.success:
                formatted_result = formatter.format_discussion_result(
                    mock_result, 
                    format_type=output_format,
                    include_metadata=True,
                    include_intermediate=args.show_steps
                )
                
                print()
                print(formatted_result.content)
                
                # Export if requested
                if args.export:
                    success = formatter.export_to_file(formatted_result, args.export)
                    if success:
                        print(f"\nResults exported to: {args.export}")
                    else:
                        print(f"\nFailed to export to: {args.export}")
                        return 1
                
                # Show summary unless in JSON format
                if output_format != OutputFormat.JSON:
                    print()
                    print("-" * 60)
                    print(formatter.generate_summary(mock_result))
        else:
            print("Starting roundtable discussion...")
            orchestrator = DiscussionOrchestrator(config)
            result = orchestrator.run_discussion(prompt, order, args.quality, args.attribution)
            
            # Format and display results
            formatter = ResponseFormatter()
            output_format = OutputFormat(args.format)
            
            if result.success:
                # Format the result
                formatted_result = formatter.format_discussion_result(
                    result, 
                    format_type=output_format,
                    include_metadata=True,
                    include_intermediate=args.show_steps
                )
                
                print()
                print(formatted_result.content)
                
                # Export if requested
                if args.export:
                    success = formatter.export_to_file(formatted_result, args.export)
                    if success:
                        print(f"\nResults exported to: {args.export}")
                    else:
                        print(f"\nFailed to export to: {args.export}")
                        return 1
                
                # Show summary unless in JSON format
                if output_format != OutputFormat.JSON:
                    print()
                    print("-" * 60)
                    print(formatter.generate_summary(result))
            else:
                print(f"Discussion failed: {result.error_message}")
                return 1

        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
