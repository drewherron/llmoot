#!/usr/bin/env python3
"""
llmoot - Multi-LLM Roundtable Discussion Tool
"""

import argparse
import sys
from src.config import Config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-LLM Roundtable Discussion Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llmoot.py -o cgo "What is the meaning of life?"
  llmoot.py -o cgoc --quality 2 "Explain quantum computing"

Order codes:
  c = Claude
  g = Gemini
  o = OpenAI/ChatGPT
        """
    )

    parser.add_argument(
        "prompt",
        help="The question or prompt for the LLM roundtable discussion"
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


def main():
    try:
        args = parse_arguments()

        # Validate order
        order = validate_order(args.order)
        
        # Load configuration
        try:
            config = Config()
        except ValueError as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            return 1

        print(f"llmoot - Multi-LLM Roundtable Discussion Tool")
        print(f"Prompt: {args.prompt}")
        print(f"Order: {order} (quality level {args.quality})")
        
        # Check API key availability
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

        # TODO: actual LLM processing
        print("Processing not yet implemented...")

        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
