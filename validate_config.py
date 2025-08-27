#!/usr/bin/env python3
"""CLI tool for validating llmoot configuration files."""

import sys
import argparse
from src.config_validator import validate_config_cli


def main():
    """Main function for config validation CLI."""
    parser = argparse.ArgumentParser(
        description="Validate llmoot configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 validate_config.py config.yaml
  python3 validate_config.py config.yaml --verbose
        """
    )
    
    parser.add_argument(
        "config_file",
        help="Path to configuration file to validate"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed suggestions and fixes"
    )
    
    args = parser.parse_args()
    
    # Validate the config file
    is_valid = validate_config_cli(args.config_file, args.verbose)
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()