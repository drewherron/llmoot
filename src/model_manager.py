"""Model selection and management utilities."""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a specific model."""
    name: str
    provider: str
    quality_level: int
    context_limit: int
    description: str
    is_available: bool = True


class ModelManager:
    """Manages model selection and fallback logic."""
    
    def __init__(self, config=None):
        """Initialize the model manager.
        
        Args:
            config: Optional Config object for reading model catalog
        """
        self.config = config
        
        # Fallback model catalog if config is not available
        self.fallback_catalog = {
            'claude': {
                1: [  # Quality level 1 (highest)
                    ModelInfo('claude-3-opus-20240229', 'claude', 1, 200000, 'Claude 3 Opus - Most capable model'),
                    ModelInfo('claude-3-sonnet-20240229', 'claude', 1, 200000, 'Claude 3 Sonnet - Fast and capable'),
                ],
                2: [  # Quality level 2 (mid-tier)
                    ModelInfo('claude-3-sonnet-20240229', 'claude', 2, 200000, 'Claude 3 Sonnet - Balanced performance'),
                    ModelInfo('claude-3-haiku-20240307', 'claude', 2, 200000, 'Claude 3 Haiku - Fast and efficient'),
                ]
            },
            'openai': {
                1: [  # Quality level 1 (highest)
                    ModelInfo('gpt-4', 'openai', 1, 128000, 'GPT-4 - Most capable OpenAI model with large context'),
                    ModelInfo('gpt-4-turbo-preview', 'openai', 1, 128000, 'GPT-4 Turbo - Large context window'),
                    ModelInfo('gpt-4o', 'openai', 1, 128000, 'GPT-4o - Latest OpenAI model'),
                ],
                2: [  # Quality level 2 (mid-tier)
                    ModelInfo('gpt-3.5-turbo', 'openai', 2, 4096, 'GPT-3.5 Turbo - Fast and cost-effective'),
                    ModelInfo('gpt-3.5-turbo-16k', 'openai', 2, 16384, 'GPT-3.5 Turbo - Extended context'),
                ]
            },
            'gemini': {
                1: [  # Quality level 1 (highest)
                    ModelInfo('gemini-2.5-pro', 'gemini', 1, 1000000, 'Gemini 2.5 Pro - Google\'s latest and most capable model'),
                    ModelInfo('gemini-1.5-pro', 'gemini', 1, 1000000, 'Gemini 1.5 Pro - Large context window'),
                    ModelInfo('gemini-pro', 'gemini', 1, 30720, 'Gemini Pro - Standard model'),
                ],
                2: [  # Quality level 2 (mid-tier)
                    ModelInfo('gemini-1.5-pro', 'gemini', 2, 1000000, 'Gemini 1.5 Pro - Balanced performance with large context'),
                    ModelInfo('gemini-pro', 'gemini', 2, 30720, 'Gemini Pro - Fast and reliable'),
                ]
            }
        }
    
    def _get_model_catalog(self):
        """Get model catalog from config or fallback."""
        if self.config:
            catalog_data = self.config.get('model_catalog')
            if catalog_data:
                return self._convert_config_catalog(catalog_data)
        
        # Fall back to hardcoded catalog
        return self.fallback_catalog
    
    def _convert_config_catalog(self, config_catalog):
        """Convert config model catalog to ModelInfo format.
        
        Args:
            config_catalog: Raw catalog from config
            
        Returns:
            Converted catalog with ModelInfo objects
        """
        converted = {}
        
        for provider, provider_data in config_catalog.items():
            converted[provider] = {}
            
            for quality_level, models in provider_data.items():
                quality_int = int(quality_level.replace('quality_', ''))
                converted[provider][quality_int] = []
                
                for model_data in models:
                    model_info = ModelInfo(
                        name=model_data.get('model', ''),
                        provider=provider,
                        quality_level=quality_int,
                        context_limit=model_data.get('context_limit', 8192),
                        description=model_data.get('description', ''),
                        is_available=model_data.get('is_available', True)
                    )
                    converted[provider][quality_int].append(model_info)
        
        return converted
    
    def get_model(self, provider: str, quality_level: int, preferred_model: Optional[str] = None) -> str:
        """Get the best available model for provider and quality level.
        
        Args:
            provider: Provider name (claude, openai, gemini)
            quality_level: Quality level (1 or 2)
            preferred_model: Specific model name to prefer if available
            
        Returns:
            Model name to use
            
        Raises:
            ValueError: If no models available for provider/quality
        """
        model_catalog = self._get_model_catalog()
        
        if provider not in model_catalog:
            raise ValueError(f"Unknown provider: {provider}")
        
        if quality_level not in model_catalog[provider]:
            raise ValueError(f"Quality level {quality_level} not available for {provider}")
        
        models = model_catalog[provider][quality_level]
        available_models = [m for m in models if m.is_available]
        
        if not available_models:
            raise ValueError(f"No available models for {provider} quality level {quality_level}")
        
        # If specific model preferred and available, use it
        if preferred_model:
            for model in available_models:
                if model.name == preferred_model:
                    return model.name
        
        # Otherwise return the first (primary) available model
        return available_models[0].name
    


# Global model manager instance (for backwards compatibility)
model_manager = ModelManager()