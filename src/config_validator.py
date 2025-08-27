"""Configuration validation and error handling for llmoot."""

import os
import yaml
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Will prevent program from running
    WARNING = "warning"  # May cause issues but program can continue
    INFO = "info"        # Helpful suggestions


@dataclass
class ValidationIssue:
    """Represents a configuration validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    suggestion: Optional[str] = None
    path: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int
    errors_count: int


class ConfigValidator:
    """Validates llmoot configuration files."""
    
    def __init__(self):
        """Initialize the config validator."""
        self.known_providers = {'claude', 'openai', 'gemini', 'google'}  # Support both google and gemini
        self.known_quality_levels = {1, 2, 3}  # Support common quality levels
        
        # Known model patterns for basic validation
        self.known_model_patterns = {
            'claude': [
                r'claude-3-(opus|sonnet|haiku)-\d{8}',
                r'claude-2\.\d+',
            ],
            'openai': [
                r'gpt-4o?(-\w+)*',
                r'gpt-3\.5-turbo(-\w+)*',
            ],
            'gemini': [
                r'gemini-\d+\.?\d*-pro',
                r'gemini-pro(-vision)?',
            ]
        }
    
    def validate_config_file(self, config_path: str) -> ValidationResult:
        """Validate a complete configuration file.
        
        Args:
            config_path: Path to the config file
            
        Returns:
            ValidationResult with all issues found
        """
        issues = []
        
        # Check if file exists
        if not os.path.exists(config_path):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="file",
                message=f"Configuration file not found: {config_path}",
                suggestion="Create config file from config.yaml.template"
            ))
            return ValidationResult(False, issues, 0, 1)
        
        # Try to load YAML
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                config_data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="yaml",
                message=f"Invalid YAML format: {e}",
                suggestion="Check YAML syntax, ensure proper indentation and no tabs"
            ))
            return ValidationResult(False, issues, 0, 1)
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="file",
                message=f"Cannot read config file: {e}",
                suggestion="Check file permissions and format"
            ))
            return ValidationResult(False, issues, 0, 1)
        
        if config_data is None:
            config_data = {}
        
        # Validate different sections
        issues.extend(self._validate_structure(config_data))
        issues.extend(self._validate_api_keys(config_data))
        issues.extend(self._validate_model_catalog(config_data))
        issues.extend(self._validate_legacy_models(config_data))
        issues.extend(self._validate_defaults(config_data))
        issues.extend(self._validate_summarization(config_data))
        
        # Count issues by severity
        errors = sum(1 for issue in issues if issue.severity == ValidationSeverity.ERROR)
        warnings = sum(1 for issue in issues if issue.severity == ValidationSeverity.WARNING)
        
        return ValidationResult(
            is_valid=(errors == 0),
            issues=issues,
            warnings_count=warnings,
            errors_count=errors
        )
    
    def _validate_structure(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate basic configuration structure."""
        issues = []
        
        # Check for required top-level sections
        recommended_sections = {
            'api_keys': 'API key configuration',
            'model_catalog': 'Available models and context limits', 
            'defaults': 'Default settings',
            'logging': 'Logging configuration'
        }
        
        for section, description in recommended_sections.items():
            if section not in config:
                severity = ValidationSeverity.ERROR if section == 'api_keys' else ValidationSeverity.WARNING
                issues.append(ValidationIssue(
                    severity=severity,
                    category="structure",
                    message=f"Missing '{section}' section",
                    suggestion=f"Add '{section}' section for {description}",
                    path=section
                ))
        
        return issues
    
    def _validate_api_keys(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate API key configuration."""
        issues = []
        
        api_keys = config.get('api_keys', {})
        if not isinstance(api_keys, dict):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="api_keys",
                message="'api_keys' must be a dictionary",
                path="api_keys"
            ))
            return issues
        
        # Check for provider API keys
        for provider in self.known_providers:
            key_value = api_keys.get(provider)
            env_var_pattern = r'\$\{[^}]+\}'
            
            if not key_value:
                # Check if environment variable is available
                env_vars = {
                    'claude': 'ANTHROPIC_API_KEY',
                    'openai': 'OPENAI_API_KEY',
                    'google': 'GOOGLE_API_KEY',
                    'gemini': 'GOOGLE_API_KEY'  # gemini also uses GOOGLE_API_KEY
                }
                env_var = env_vars.get(provider)
                if env_var and os.environ.get(env_var):
                    continue  # Environment variable is available
                
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="api_keys",
                    message=f"No API key configured for {provider}",
                    suggestion=f"Add {provider} API key or set {env_var} environment variable",
                    path=f"api_keys.{provider}"
                ))
            elif isinstance(key_value, str):
                # Check if it's an environment variable reference
                if re.match(env_var_pattern, key_value):
                    env_var_name = key_value.strip('${}')
                    if not os.environ.get(env_var_name):
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category="api_keys",
                            message=f"Environment variable {env_var_name} not set",
                            suggestion=f"Set {env_var_name} environment variable",
                            path=f"api_keys.{provider}"
                        ))
                # Check if it looks like a real API key
                elif len(key_value) < 10:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="api_keys",
                        message=f"API key for {provider} seems too short",
                        suggestion="Verify API key is complete and correct",
                        path=f"api_keys.{provider}"
                    ))
        
        return issues
    
    def _validate_model_catalog(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate model catalog structure and content (flat format)."""
        issues = []
        
        model_catalog = config.get('model_catalog')
        if not model_catalog:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="model_catalog",
                message="No model_catalog defined",
                suggestion="Add model_catalog section with models and their context limits"
            ))
            return issues
        
        if not isinstance(model_catalog, dict):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="model_catalog",
                message="model_catalog must be a dictionary",
                path="model_catalog"
            ))
            return issues
        
        # Validate each model in the flat catalog
        for model_name, model_data in model_catalog.items():
            if not isinstance(model_data, dict):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="model_catalog",
                    message=f"Model data for '{model_name}' must be a dictionary",
                    path=f"model_catalog.{model_name}"
                ))
                continue
            
            # Validate context_limit
            context_limit = model_data.get('context_limit')
            if context_limit is None:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="model_catalog",
                    message=f"Model '{model_name}' missing context_limit",
                    suggestion="Add context_limit field with token count",
                    path=f"model_catalog.{model_name}"
                ))
            elif not isinstance(context_limit, int) or context_limit <= 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="model_catalog",
                    message=f"context_limit for '{model_name}' must be a positive integer",
                    path=f"model_catalog.{model_name}.context_limit"
                ))
            elif context_limit < 1000:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="model_catalog",
                    message=f"Very small context limit for '{model_name}': {context_limit}",
                    suggestion="Verify context limit is correct (typically 4K-1M+ tokens)",
                    path=f"model_catalog.{model_name}.context_limit"
                ))
            
            # Check if model name matches known patterns
            model_provider = self._guess_provider_from_model_name(model_name)
            if model_provider:
                patterns = self.known_model_patterns.get(model_provider, [])
                if patterns and not any(re.match(pattern, model_name) for pattern in patterns):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="model_catalog",
                        message=f"Unusual model name for {model_provider}: {model_name}",
                        suggestion=f"Verify model name is correct for {model_provider}",
                        path=f"model_catalog.{model_name}"
                    ))
        
        return issues
    
    def _guess_provider_from_model_name(self, model_name: str) -> Optional[str]:
        """Guess provider from model name."""
        if model_name.startswith('gpt-'):
            return 'openai'
        elif model_name.startswith('claude-'):
            return 'claude'
        elif model_name.startswith('gemini-'):
            return 'gemini'
        return None
    
    def _validate_model_info(self, model_info: Any, provider: str, 
                           quality_level: str, index: int) -> List[ValidationIssue]:
        """Validate individual model information."""
        issues = []
        path_base = f"model_catalog.{provider}.{quality_level}[{index}]"
        
        if not isinstance(model_info, dict):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="model_catalog",
                message=f"Model info must be a dictionary",
                path=path_base
            ))
            return issues
        
        # Required fields
        required_fields = {'model', 'context_limit', 'description'}
        for field in required_fields:
            if field not in model_info:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="model_catalog",
                    message=f"Missing required field: {field}",
                    path=f"{path_base}.{field}"
                ))
        
        # Validate model name
        model_name = model_info.get('model')
        if model_name:
            if not isinstance(model_name, str) or not model_name.strip():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="model_catalog",
                    message="Model name must be a non-empty string",
                    path=f"{path_base}.model"
                ))
            else:
                # Check if model name matches known patterns
                patterns = self.known_model_patterns.get(provider, [])
                if patterns and not any(re.match(pattern, model_name) for pattern in patterns):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="model_catalog",
                        message=f"Unusual model name for {provider}: {model_name}",
                        suggestion=f"Verify model name is correct for {provider}",
                        path=f"{path_base}.model"
                    ))
        
        # Validate context limit
        context_limit = model_info.get('context_limit')
        if context_limit is not None:
            if not isinstance(context_limit, int) or context_limit <= 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="model_catalog",
                    message="Context limit must be a positive integer",
                    path=f"{path_base}.context_limit"
                ))
            elif context_limit < 1000:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="model_catalog",
                    message=f"Very small context limit: {context_limit}",
                    suggestion="Verify context limit is correct (typically 4K-1M+ tokens)",
                    path=f"{path_base}.context_limit"
                ))
        
        # Validate description
        description = model_info.get('description')
        if description is not None and not isinstance(description, str):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="model_catalog",
                message="Description should be a string",
                path=f"{path_base}.description"
            ))
        
        return issues
    
    def _validate_legacy_models(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate legacy models configuration."""
        issues = []
        
        models = config.get('models')
        if not models:
            return issues  # Legacy models are optional
        
        if not isinstance(models, dict):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="models",
                message="'models' section must be a dictionary",
                path="models"
            ))
            return issues
        
        # Validate quality levels (support both integer and "quality_N" format)
        for quality_key, provider_models in models.items():
            quality_key_str = str(quality_key)
            
            # Check if it's an integer (current format) or quality_N (legacy format)
            if isinstance(quality_key, int):
                if quality_key not in self.known_quality_levels:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="models",
                        message=f"Unusual quality level: {quality_key}",
                        suggestion="Typically use quality levels 1, 2, or 3",
                        path=f"models.{quality_key}"
                    ))
            elif quality_key_str.startswith('quality_'):
                try:
                    level_num = int(quality_key_str.replace('quality_', ''))
                    if level_num not in self.known_quality_levels:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category="models",
                            message=f"Unusual quality level: {level_num}",
                            suggestion="Typically use quality levels 1, 2, or 3",
                            path=f"models.{quality_key}"
                        ))
                except ValueError:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="models",
                        message=f"Invalid quality level format: {quality_key}",
                        path=f"models.{quality_key}"
                    ))
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="models",
                    message=f"Quality key must be integer or 'quality_N': {quality_key_str}",
                    suggestion="Use 1, 2, 3 or 'quality_1', 'quality_2', etc.",
                    path=f"models.{quality_key}"
                ))
            
            if not isinstance(provider_models, dict):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="models",
                    message=f"Quality level {quality_key} must be a dictionary",
                    path=f"models.{quality_key}"
                ))
                continue
            
            # Check each provider
            for provider, model_name in provider_models.items():
                if provider not in self.known_providers:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="models",
                        message=f"Unknown provider: {provider}",
                        suggestion=f"Known providers: {', '.join(self.known_providers)}",
                        path=f"models.{quality_key}.{provider}"
                    ))
                
                if not isinstance(model_name, str) or not model_name.strip():
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="models",
                        message=f"Model name for {provider} must be a non-empty string",
                        path=f"models.{quality_key}.{provider}"
                    ))
        
        return issues
    
    def _validate_defaults(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate defaults configuration."""
        issues = []
        
        defaults = config.get('defaults')
        if not defaults:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="defaults",
                message="No defaults section defined",
                suggestion="Consider adding defaults section for quality_level"
            ))
            return issues
        
        if not isinstance(defaults, dict):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="defaults",
                message="'defaults' section must be a dictionary",
                path="defaults"
            ))
            return issues
        
        # Validate quality_level
        quality_level = defaults.get('quality_level')
        if quality_level is not None:
            if not isinstance(quality_level, int):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="defaults",
                    message="quality_level must be an integer",
                    path="defaults.quality_level"
                ))
            elif quality_level not in self.known_quality_levels:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="defaults",
                    message=f"Unusual default quality level: {quality_level}",
                    suggestion="Typically use 1 or 2",
                    path="defaults.quality_level"
                ))
        
        return issues
    
    def _validate_summarization(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate summarization configuration."""
        issues = []
        
        summarization = config.get('summarization')
        if not summarization:
            return issues  # Summarization config is optional
        
        if not isinstance(summarization, dict):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="summarization",
                message="'summarization' section must be a dictionary",
                path="summarization"
            ))
            return issues
        
        # Validate threshold
        threshold = summarization.get('threshold')
        if threshold is not None:
            if not isinstance(threshold, (int, float)):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="summarization",
                    message="threshold must be a number",
                    path="summarization.threshold"
                ))
            elif not 0.1 <= threshold <= 1.0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="summarization",
                    message=f"Unusual threshold value: {threshold}",
                    suggestion="Threshold should be between 0.1 and 1.0 (e.g., 0.75 for 75%)",
                    path="summarization.threshold"
                ))
        
        # Validate max_tokens
        max_tokens = summarization.get('max_tokens')
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="summarization",
                    message="max_tokens must be a positive integer",
                    path="summarization.max_tokens"
                ))
        
        return issues
    
    def format_validation_report(self, result: ValidationResult, verbose: bool = False) -> str:
        """Format validation result as a readable report.
        
        Args:
            result: Validation result to format
            verbose: Include detailed suggestions
            
        Returns:
            Formatted report string
        """
        if not result.issues:
            return "✅ Configuration validation passed with no issues!"
        
        report = []
        report.append(f"Configuration Validation Report:")
        report.append(f"  Errors: {result.errors_count}")
        report.append(f"  Warnings: {result.warnings_count}")
        report.append("")
        
        if result.errors_count > 0:
            report.append("❌ ERRORS (must fix to continue):")
            for issue in result.issues:
                if issue.severity == ValidationSeverity.ERROR:
                    report.append(f"  • {issue.message}")
                    if issue.path:
                        report.append(f"    Path: {issue.path}")
                    if verbose and issue.suggestion:
                        report.append(f"    Fix: {issue.suggestion}")
            report.append("")
        
        if result.warnings_count > 0:
            report.append("⚠️ WARNINGS (recommended to fix):")
            for issue in result.issues:
                if issue.severity == ValidationSeverity.WARNING:
                    report.append(f"  • {issue.message}")
                    if issue.path:
                        report.append(f"    Path: {issue.path}")
                    if verbose and issue.suggestion:
                        report.append(f"    Fix: {issue.suggestion}")
            report.append("")
        
        info_count = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.INFO)
        if info_count > 0:
            report.append("ℹ️ SUGGESTIONS:")
            for issue in result.issues:
                if issue.severity == ValidationSeverity.INFO:
                    report.append(f"  • {issue.message}")
                    if verbose and issue.suggestion:
                        report.append(f"    Suggestion: {issue.suggestion}")
        
        if not result.is_valid:
            report.append("")
            report.append("❌ Configuration is invalid and cannot be used.")
            report.append("Please fix the errors above before running llmoot.")
        
        return "\n".join(report)


def validate_config_cli(config_path: str, verbose: bool = False) -> bool:
    """CLI helper for config validation.
    
    Args:
        config_path: Path to config file
        verbose: Show detailed suggestions
        
    Returns:
        True if config is valid, False otherwise
    """
    validator = ConfigValidator()
    result = validator.validate_config_file(config_path)
    
    print(validator.format_validation_report(result, verbose))
    
    return result.is_valid