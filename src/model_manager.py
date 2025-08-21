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
    
    def __init__(self):
        """Initialize the model manager."""
        self.model_catalog = {
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
                    ModelInfo('gpt-4', 'openai', 1, 8192, 'GPT-4 - Most capable OpenAI model'),
                    ModelInfo('gpt-4-turbo-preview', 'openai', 1, 128000, 'GPT-4 Turbo - Large context window'),
                ],
                2: [  # Quality level 2 (mid-tier)
                    ModelInfo('gpt-3.5-turbo', 'openai', 2, 4096, 'GPT-3.5 Turbo - Fast and cost-effective'),
                    ModelInfo('gpt-3.5-turbo-16k', 'openai', 2, 16384, 'GPT-3.5 Turbo - Extended context'),
                ]
            },
            'gemini': {
                1: [  # Quality level 1 (highest)
                    ModelInfo('gemini-pro', 'gemini', 1, 30720, 'Gemini Pro - Google\'s most capable model'),
                ],
                2: [  # Quality level 2 (mid-tier)
                    ModelInfo('gemini-pro', 'gemini', 2, 30720, 'Gemini Pro - Same model, different usage tier'),
                ]
            }
        }
    
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
        if provider not in self.model_catalog:
            raise ValueError(f"Unknown provider: {provider}")
        
        if quality_level not in self.model_catalog[provider]:
            raise ValueError(f"Quality level {quality_level} not available for {provider}")
        
        models = self.model_catalog[provider][quality_level]
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
    


# Global model manager instance
model_manager = ModelManager()