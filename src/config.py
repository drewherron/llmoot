"""Configuration management for llmoot."""

import os
import re
import yaml
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for llmoot."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration.

        Args:
            config_path: Path to config file (defaults to config.yaml)
        """
        self.config_path = config_path
        self._config = {}
        self.load()

    def load(self):
        """Load configuration from file with environment variable fallback."""
        # Try to load config file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    content = f.read()
                    # Replace environment variable placeholders
                    content = self._substitute_env_vars(content)
                    self._config = yaml.safe_load(content) or {}
            except (yaml.YAMLError, IOError) as e:
                raise ValueError(f"Error loading config file '{self.config_path}': {e}")
        else:
            # No config file, use empty config (will fall back to env vars)
            self._config = {}

        # Ensure required structure exists
        self._ensure_structure()

    def _substitute_env_vars(self, content: str) -> str:
        """Replace ${ENV_VAR} placeholders with environment variable values."""
        def replace_env_var(match):
            env_var = match.group(1)
            return os.environ.get(env_var, match.group(0))  # Keep placeholder if env var not found

        return re.sub(r'\$\{([^}]+)\}', replace_env_var, content)

    def _ensure_structure(self):
        """Ensure config has required structure with defaults."""
        if 'api_keys' not in self._config:
            self._config['api_keys'] = {}

        if 'model_catalog' not in self._config:
            self._config['model_catalog'] = {}

        if 'models' not in self._config:
            self._config['models'] = {
                1: {
                    'claude': 'claude-3-opus-20240229',
                    'openai': 'gpt-4o',
                    'gemini': 'gemini-2.5-pro'
                },
                2: {
                    'claude': 'claude-3-sonnet-20240229',
                    'openai': 'gpt-3.5-turbo-16k',
                    'gemini': 'gemini-1.5-pro'
                }
            }

        if 'defaults' not in self._config:
            self._config['defaults'] = {
                'quality_level': 1
            }

        if 'logging' not in self._config:
            self._config['logging'] = {
                'enabled': True,
                'directory': 'logs'
            }

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for provider, checking config file then environment.

        Args:
            provider: Provider name (anthropic, openai, google)

        Returns:
            API key or None if not found
        """
        # Check config file first
        key = self._config.get('api_keys', {}).get(provider)
        if key and not key.startswith('${'):
            return key

        # Fall back to environment variables
        env_vars = {
            'anthropic': 'ANTHROPIC_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'google': 'GOOGLE_API_KEY'
        }

        env_var = env_vars.get(provider)
        if env_var:
            return os.environ.get(env_var)

        return None

    def get_model(self, provider: str, quality_level: int) -> str:
        """Get model name for a provider and quality level from the config.

        Args:
            provider: Provider name (claude, openai, gemini)
            quality_level: Quality level as an integer (e.g., 1, 2, 3...)

        Returns:
            Model name as a string.

        Raises:
            ValueError: If the quality level or provider is not found in the config.
        """
        models_config = self._config.get('models', {})

        if quality_level not in models_config:
            raise ValueError(
                f"Quality level '{quality_level}' is not defined in your config file. "
                f"Available levels: {list(models_config.keys())}"
            )

        if provider not in models_config[quality_level]:
            raise ValueError(
                f"Provider '{provider}' is not defined for quality level '{quality_level}' "
                f"in your config file. Available providers for this level: "
                f"{list(models_config[quality_level].keys())}"
            )

        return models_config[quality_level][provider]

    def get_system_prompt(self, is_final: bool) -> str:
        """Get system prompt for LLM.

        Args:
            is_final: Whether this is the final LLM in sequence

        Returns:
            System prompt text
        """
        prompts = self._config.get('system_prompts', {})

        if is_final:
            return prompts.get('final', self._get_default_final_prompt())
        else:
            return prompts.get('intermediate', self._get_default_intermediate_prompt())

    def _get_default_intermediate_prompt(self) -> str:
        """Default intermediate system prompt."""
        return """You are participating in a collaborative roundtable discussion with other AI assistants.
Your role is to provide thoughtful insights that complement previous responses, add new perspectives,
and build upon good points made by others. Focus on content and ideas, not final formatting."""

    def _get_default_final_prompt(self) -> str:
        """Default final system prompt."""
        return """You are the final participant in a collaborative roundtable discussion.
Your role is to synthesize the best insights from all previous responses into a comprehensive
final answer. Apply any formatting requested by the user."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation like 'defaults.quality_level')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def is_logging_enabled(self) -> bool:
        """Check if logging is enabled."""
        return self.get('logging.enabled', True)

    def get_log_directory(self) -> str:
        """Get the log directory path."""
        return self.get('logging.directory', 'logs')
