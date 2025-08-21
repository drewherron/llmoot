"""Prompt engineering and context building utilities."""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from src.config import Config


@dataclass
class ResponseContext:
    """Context from a previous LLM response."""
    provider: str
    model: str
    content: str
    step_number: int
    tokens_used: int


class PromptBuilder:
    """Builds prompts for roundtable discussions."""
    
    def __init__(self, config: Config):
        """Initialize prompt builder with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def build_system_prompt(self, is_final: bool, user_prompt: str, step_info: Dict, attribution: bool = False) -> str:
        """Build system prompt for an LLM.
        
        Args:
            is_final: Whether this is the final LLM in sequence
            user_prompt: Original user prompt
            step_info: Information about current step (step_number, total_steps, etc.)
            attribution: Whether attribution mode is enabled
            
        Returns:
            System prompt text
        """
        # Get base system prompt from config
        base_prompt = self.config.get_system_prompt(is_final)
        
        # Extract format requirements from user prompt
        format_instructions = self._extract_format_requirements(user_prompt)
        
        # Build enhanced system prompt
        if is_final:
            enhanced_prompt = self._build_final_system_prompt(
                base_prompt, format_instructions, step_info, attribution
            )
        else:
            enhanced_prompt = self._build_intermediate_system_prompt(
                base_prompt, step_info
            )
        
        return enhanced_prompt
    
    def build_user_prompt(self, 
                         original_prompt: str, 
                         previous_responses: List[ResponseContext],
                         step_info: Dict,
                         attribution: bool = False) -> str:
        """Build user prompt with context from previous responses.
        
        Args:
            original_prompt: Original user prompt
            previous_responses: List of previous LLM responses
            step_info: Current step information
            attribution: Whether to include model names in context
            
        Returns:
            Enhanced user prompt with context
        """
        # Start with original prompt
        enhanced_prompt = f"Original user request:\n{original_prompt}\n"
        
        # Add context from previous responses
        if previous_responses:
            enhanced_prompt += "\nPrevious discussion:\n"
            
            for i, response in enumerate(previous_responses, 1):
                if attribution:
                    # Use "Model says:" format for attribution
                    provider_name = self._get_display_name(response.provider)
                    enhanced_prompt += f"\n{provider_name} says:\n{response.content}\n"
                else:
                    # Use numbered format without attribution
                    enhanced_prompt += f"\n{i}. {response.provider.title()} ({response.model}):\n"
                    enhanced_prompt += f"{response.content}\n"
        
        # Add step context
        if step_info.get('total_steps', 0) > 1:
            enhanced_prompt += f"\nYou are step {step_info['step_number']} of {step_info['total_steps']} in this roundtable discussion."
        
        return enhanced_prompt
    
    def _get_display_name(self, provider: str) -> str:
        """Get display name for provider attribution.
        
        Args:
            provider: Provider name (claude, openai, gemini)
            
        Returns:
            Display name for attribution
        """
        display_names = {
            'claude': 'Claude',
            'openai': 'ChatGPT', 
            'gemini': 'Gemini'
        }
        return display_names.get(provider, provider.title())
    
    def _extract_format_requirements(self, user_prompt: str) -> Dict[str, str]:
        """Extract format requirements from user prompt.
        
        Args:
            user_prompt: User's prompt text
            
        Returns:
            Dict with format requirements
        """
        format_requirements = {}
        
        # Check for explicit format requests
        format_patterns = {
            'markdown': r'(?i)\b(markdown|\.md)\b',
            'json': r'(?i)\b(json|\.json)\b',
            'yaml': r'(?i)\b(yaml|\.yaml)\b',
            'org-mode': r'(?i)\b(org[-\s]?mode|\.org)\b',
            'code': r'(?i)\b(code|programming|script)\b',
            'bullet_points': r'(?i)\b(bullet points|bullets|list)\b',
            'numbered': r'(?i)\b(numbered|numbered list)\b'
        }
        
        for format_type, pattern in format_patterns.items():
            if re.search(pattern, user_prompt):
                format_requirements[format_type] = True
        
        # Check for structure requests
        structure_patterns = {
            'summary': r'(?i)\b(summary|summarize)\b',
            'detailed': r'(?i)\b(detailed|comprehensive|thorough)\b',
            'concise': r'(?i)\b(concise|brief|short)\b',
            'step_by_step': r'(?i)\b(step[-\s]?by[-\s]?step|steps)\b'
        }
        
        for structure_type, pattern in structure_patterns.items():
            if re.search(pattern, user_prompt):
                format_requirements[structure_type] = True
        
        return format_requirements
    
    def _build_final_system_prompt(self, 
                                 base_prompt: str, 
                                 format_requirements: Dict[str, str],
                                 step_info: Dict,
                                 attribution: bool = False) -> str:
        """Build system prompt for the final LLM."""
        enhanced_prompt = base_prompt + "\n\n"
        
        # Add final synthesis instructions
        enhanced_prompt += """IMPORTANT: You are providing the FINAL response that will be delivered to the user. 
Your response should be complete, well-structured, and incorporate the best insights from all previous discussion participants.
"""
        
        # Add format instructions if detected
        if format_requirements:
            enhanced_prompt += "\nFormatting requirements detected in user request:\n"
            
            if format_requirements.get('markdown'):
                enhanced_prompt += "- Format your response in Markdown with proper headers and structure\n"
            if format_requirements.get('json'):
                enhanced_prompt += "- Structure your response as valid JSON\n"
            if format_requirements.get('code'):
                enhanced_prompt += "- Include code examples with proper syntax highlighting\n"
            if format_requirements.get('bullet_points'):
                enhanced_prompt += "- Use bullet points for key information\n"
            if format_requirements.get('numbered'):
                enhanced_prompt += "- Use numbered lists for sequential information\n"
            if format_requirements.get('step_by_step'):
                enhanced_prompt += "- Provide step-by-step instructions or analysis\n"
        
        # Add synthesis guidance
        enhanced_prompt += "\nSynthesis guidelines:\n"
        enhanced_prompt += "- Identify and highlight consensus points from the discussion\n"
        enhanced_prompt += "- Address any conflicting viewpoints constructively\n"
        enhanced_prompt += "- Add your own insights while building on others' contributions\n"
        enhanced_prompt += "- Ensure the response fully addresses the user's original question\n"
        
        # Add attribution instructions if enabled
        if attribution:
            enhanced_prompt += "\nAttribution instructions:\n"
            enhanced_prompt += "When synthesizing, reference specific models by name when discussing their contributions, "
            enhanced_prompt += "agreements, or disagreements (e.g., 'As Claude emphasized...' or 'While ChatGPT focused on X, Gemini highlighted Y'). "
            enhanced_prompt += "This helps readers understand how different AI models approached the question.\n"
        
        return enhanced_prompt
    
    def _build_intermediate_system_prompt(self, 
                                        base_prompt: str,
                                        step_info: Dict) -> str:
        """Build system prompt for intermediate LLMs."""
        enhanced_prompt = base_prompt + "\n\n"
        
        # Add intermediate-specific guidance
        enhanced_prompt += f"""You are participant {step_info['step_number']} of {step_info['total_steps']} in this discussion.
Your goal is to contribute valuable insights that will help create a comprehensive final response.

Focus on:
- Adding unique perspectives not yet covered
- Building upon strong points made by others
- Identifying potential gaps or areas needing more exploration
- Providing supporting evidence or examples
- Raising thoughtful questions or considerations

Remember: Another AI will provide the final synthesized response to the user, so focus on contributing substantive content rather than final formatting.
"""
        
        return enhanced_prompt
    
    def estimate_prompt_tokens(self, prompt: str) -> int:
        """Estimate token count for a prompt.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token
        return len(prompt) // 4
    
    def truncate_context_if_needed(self, 
                                 prompt: str, 
                                 context_limit: int, 
                                 max_response_tokens: int = 1000) -> str:
        """Truncate prompt if it exceeds context limits.
        
        Args:
            prompt: Full prompt text
            context_limit: Model's context limit in tokens
            max_response_tokens: Tokens to reserve for response
            
        Returns:
            Truncated prompt if necessary
        """
        estimated_tokens = self.estimate_prompt_tokens(prompt)
        available_tokens = context_limit - max_response_tokens
        
        if estimated_tokens <= available_tokens:
            return prompt
        
        # Need to truncate - aim for 80% of available tokens
        target_tokens = int(available_tokens * 0.8)
        target_chars = target_tokens * 4
        
        if target_chars < len(prompt):
            # Truncate and add notice
            truncated = prompt[:target_chars]
            # Try to truncate at a sensible boundary
            if '\n' in truncated[-100:]:
                last_newline = truncated.rfind('\n')
                if last_newline > target_chars - 200:
                    truncated = truncated[:last_newline]
            
            truncated += f"\n\n[Note: Context was truncated due to length limits - {estimated_tokens} tokens reduced to ~{target_tokens} tokens]"
            
            # Notify user about truncation
            print("WARNING: Context exceeded token limit - truncated previous responses to fit within model constraints")
            
            return truncated
        
        return prompt