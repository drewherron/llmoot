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
    
    def get_model_info(self, provider: str, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model.
        
        Args:
            provider: Provider name
            model_name: Model name
            
        Returns:
            ModelInfo object or None if not found
        """
        if provider not in self.model_catalog:
            return None
        
        for quality_level, models in self.model_catalog[provider].items():
            for model in models:
                if model.name == model_name:
                    return model
        
        return None
    
    def list_available_models(self, provider: str, quality_level: int) -> List[ModelInfo]:
        """List all available models for provider and quality level.
        
        Args:
            provider: Provider name
            quality_level: Quality level
            
        Returns:
            List of available ModelInfo objects
        """
        if provider not in self.model_catalog:
            return []
        
        if quality_level not in self.model_catalog[provider]:
            return []
        
        return [m for m in self.model_catalog[provider][quality_level] if m.is_available]
    
    def get_all_providers(self) -> List[str]:
        """Get list of all supported providers."""
        return list(self.model_catalog.keys())
    
    def get_quality_levels(self, provider: str) -> List[int]:
        """Get available quality levels for a provider."""
        if provider not in self.model_catalog:
            return []
        return list(self.model_catalog[provider].keys())
    
    def mark_model_unavailable(self, provider: str, model_name: str):
        """Mark a model as unavailable (for fallback logic).
        
        Args:
            provider: Provider name
            model_name: Model name to mark unavailable
        """
        model_info = self.get_model_info(provider, model_name)
        if model_info:
            model_info.is_available = False
    
    def get_fallback_model(self, provider: str, quality_level: int, failed_model: str) -> Optional[str]:
        """Get a fallback model when the primary model fails.
        
        Args:
            provider: Provider name
            quality_level: Quality level
            failed_model: Model that failed
            
        Returns:
            Fallback model name or None if no fallback available
        """
        models = self.list_available_models(provider, quality_level)
        
        # Remove the failed model from consideration
        fallback_models = [m for m in models if m.name != failed_model]
        
        if fallback_models:
            return fallback_models[0].name
        
        return None
    
    def generate_config_documentation(self) -> str:
        """Generate documentation for model configuration."""
        doc = "# Available Models by Provider and Quality Level\n\n"
        
        for provider in self.get_all_providers():
            doc += f"## {provider.title()}\n\n"
            
            for quality_level in self.get_quality_levels(provider):
                models = self.list_available_models(provider, quality_level)
                tier_name = "Highest Quality" if quality_level == 1 else "Mid-Tier"
                doc += f"### Quality Level {quality_level} ({tier_name})\n"
                
                for model in models:
                    doc += f"- **{model.name}**: {model.description}\n"
                    doc += f"  - Context: {model.context_limit:,} tokens\n"
                
                doc += "\n"
            
            doc += "\n"
        
        return doc


# Global model manager instance
model_manager = ModelManager()