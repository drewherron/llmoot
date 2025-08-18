#!/usr/bin/env python3
"""
llmoot - Multi-LLM Roundtable Discussion Tool
"""

import argparse
import os
import sys
from src.config import Config
from src.mock_responses import DEV_MODE, get_mock_client


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-LLM Roundtable Discussion Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llmoot.py -o cgo "What is the meaning of life?"
  llmoot.py -o cgoc --quality 2 "Explain quantum computing"
  llmoot.py -o cgo prompt.txt

Order codes:
  c = Claude
  g = Gemini
  o = OpenAI/ChatGPT
        """
    )

    parser.add_argument(
        "prompt",
        help="The question/prompt for discussion, or path to file containing the prompt"
    )

    parser.add_argument(
        "-o", "--order",
        required=True,
        help="Order of LLM execution (e.g., 'cgo' for Claude->Gemini->OpenAI)"
    )

    parser.add_argument(
        "--quality",
        type=int,
        choices=[1, 2],
        default=1,
        help="Quality level: 1=highest models, 2=mid-tier models (default: 1)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="llmoot 0.1.0"
    )

    return parser.parse_args()


def validate_order(order):
    """Validate the order string contains only valid provider codes."""
    valid_codes = {'c', 'g', 'o'}
    if not order:
        raise ValueError("Order cannot be empty")

    for code in order.lower():
        if code not in valid_codes:
            raise ValueError(f"Invalid provider code '{code}'. Valid codes: c, g, o")

    return order.lower()


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


def run_mock_discussion(order, prompt, config, quality_level):
    """Run a mock roundtable discussion for development/testing."""
    provider_map = {'c': 'claude', 'g': 'gemini', 'o': 'openai'}
    responses = []
    
    for i, code in enumerate(order):
        provider = provider_map[code]
        model = config.get_model(provider, quality_level)
        is_final = (i == len(order) - 1)
        
        # Create mock client
        client = get_mock_client(provider, model)
        
        # Show progress
        status = "Final response" if is_final else f"Response {i+1}/{len(order)}"
        print(f"🤖 {provider.title()} ({model}) - {status}...")
        
        # Build context from previous responses
        context = f"User prompt: {prompt}\n\n"
        if responses:
            context += "Previous responses:\n"
            for j, resp in enumerate(responses):
                context += f"\n{resp['provider'].title()} response:\n{resp['content']}\n"
        
        # Generate response
        response = client.generate_response(context, is_final)
        responses.append(response)
        
        # Show the response
        print(f"\n--- {provider.title()} Response ---")
        print(response['content'])
        print(f"({response['tokens_used']} tokens, {response['response_time']:.1f}s)")
        print()
    
    print("=" * 60)
    print("🎉 Mock roundtable discussion complete!")
    print(f"Total responses: {len(responses)}")
    total_tokens = sum(r['tokens_used'] for r in responses)
    total_time = sum(r['response_time'] for r in responses)
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time:.1f}s")


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
            print("⚠️  Running in DEVELOPMENT MODE with mock responses")
        
        # Show prompt source
        if os.path.isfile(args.prompt):
            print(f"Prompt file: {args.prompt}")
        else:
            print(f"Prompt: {prompt}")
        print(f"Order: {order} (quality level {args.quality})")
        
        # Check API key availability (skip in dev mode)
        if not DEV_MODE:
            provider_status = config.validate_providers(order)
            missing_keys = [code for code, available in provider_status.items() if not available]
            
            if missing_keys:
                provider_names = {'c': 'Claude (Anthropic)', 'g': 'Gemini (Google)', 'o': 'OpenAI'}
                missing_names = [provider_names[code] for code in missing_keys]
                print(f"Error: Missing API keys for: {', '.join(missing_names)}", file=sys.stderr)
                print("Set environment variables or create config.yaml from config.yaml.template", file=sys.stderr)
                return 1
        
        # Show selected models
        provider_map = {'c': 'claude', 'g': 'gemini', 'o': 'openai'}
        models = []
        for code in order:
            provider = provider_map[code]
            model = config.get_model(provider, args.quality)
            models.append(f"{code}={model}")
        print(f"Models: {' -> '.join(models)}")
        print()
        
        # Run mock processing in dev mode
        if DEV_MODE:
            print("Starting mock roundtable discussion...")
            run_mock_discussion(order, prompt, config, args.quality)
        else:
            # TODO: actual LLM processing
            print("Real LLM processing not yet implemented...")

        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
