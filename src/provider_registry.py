"""Provider registry for managing LLM clients."""

from typing import Dict, Type, Optional
from src.llm_client import LLMClient
from src.config import Config


class ProviderRegistry:
    """Registry for LLM provider clients."""
    
    def __init__(self):
        """Initialize the registry."""
        self._providers: Dict[str, Type[LLMClient]] = {}
        self._provider_info = {
            'claude': {
                'name': 'Claude (Anthropic)',
                'config_key': 'anthropic',
                'models': {
                    1: 'claude-3-opus-20240229',
                    2: 'claude-3-sonnet-20240229'
                }
            },
            'openai': {
                'name': 'OpenAI',
                'config_key': 'openai', 
                'models': {
                    1: 'gpt-4',
                    2: 'gpt-3.5-turbo'
                }
            },
            'gemini': {
                'name': 'Gemini (Google)',
                'config_key': 'google',
                'models': {
                    1: 'gemini-pro',
                    2: 'gemini-pro'
                }
            }
        }
        self._register_providers()

    def _register_providers(self):
        """Register all available provider clients."""
        try:
            from src.claude_client import ClaudeClient
            self.register_provider('claude', ClaudeClient)
        except ImportError:
            pass

        try:
            from src.openai_client import OpenAIClient
            self.register_provider('openai', OpenAIClient)
        except ImportError:
            pass

        from src.gemini_client import GeminiClient
        self.register_provider('gemini', GeminiClient)
    
    def register_provider(self, provider_name: str, client_class: Type[LLMClient]):
        """Register a new provider client.
        
        Args:
            provider_name: The name of the provider (e.g., 'claude', 'openai')
            client_class: LLMClient subclass
        """
        self._providers[provider_name] = client_class
    
    def get_provider_names(self) -> Dict[str, str]:
        """Get mapping of provider codes to human-readable names."""
        return {code: info['name'] for code, info in self._provider_info.items()}
    
    def create_client(self, provider_code: str, config: Config, quality_level: int) -> LLMClient:
        """Create a client instance for the given provider.
        
        Args:
            provider_code: Provider code (a, o, g)
            config: Configuration object
            quality_level: Quality level (1 or 2)
            
        Returns:
            Configured LLMClient instance
            
        Raises:
            ValueError: If provider not registered or missing API key
        """
        # Map provider codes to names
        provider_map = {'a': 'claude', 'o': 'openai', 'g': 'gemini'}
        provider_name = provider_map.get(provider_code)
        
        if not provider_name or provider_name not in self._provider_info:
            raise ValueError(f"Unknown provider code: {provider_code}")
        
        info = self._provider_info[provider_name]
        
        # Get API key
        api_key = config.get_api_key(info['config_key'])
        if not api_key:
            raise ValueError(f"Missing API key for {info['name']}")
        
        # Get model for quality level
        model = config.get_model(provider_name, quality_level)
        
        # Get client class
        if provider_name not in self._providers:
            raise ValueError(f"Provider {provider_name} not registered. Make sure to register the client class.")
        
        client_class = self._providers[provider_name]
        
        # Create and return client
        return client_class(api_key=api_key, model=model)
    
    def validate_providers(self, order: str, config: Config) -> Dict[str, bool]:
        """Validate that all providers in order have available API keys.
        
        Args:
            order: Order string like 'aog'
            config: Configuration object
            
        Returns:
            Dict mapping provider codes to availability status
        """
        results = {}
        provider_map = {'a': 'claude', 'o': 'openai', 'g': 'gemini'}
        
        for code in order:
            provider_name = provider_map.get(code)
            if provider_name and provider_name in self._provider_info:
                info = self._provider_info[provider_name]
                api_key = config.get_api_key(info['config_key'])
                results[code] = api_key is not None
            else:
                results[code] = False
        
        return results
    
    def get_missing_providers(self, order: str, config: Config) -> list:
        """Get list of missing provider names for error messages.
        
        Args:
            order: Order string like 'aog'
            config: Configuration object
            
        Returns:
            List of human-readable provider names that are missing API keys
        """
        provider_status = self.validate_providers(order, config)
        provider_names = self.get_provider_names()
        provider_map = {'a': 'claude', 'o': 'openai', 'g': 'gemini'}
        
        missing = []
        for code, available in provider_status.items():
            if not available:
                provider_name = provider_map.get(code)
                if provider_name in provider_names:
                    missing.append(provider_names[provider_name])
        
        return missing


# Global registry instance
registry = ProviderRegistry()

