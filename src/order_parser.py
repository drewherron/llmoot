"""Order parsing and validation utilities."""

from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ExecutionStep:
    """Represents a single step in the execution order."""
    provider_code: str
    provider_name: str
    step_number: int
    total_steps: int
    is_final: bool


class OrderParser:
    """Parser and validator for execution orders."""
    
    def __init__(self):
        """Initialize the order parser."""
        self.provider_map = {
            'a': 'claude',
            'o': 'openai', 
            'g': 'gemini'
        }
        self.provider_display_names = {
            'a': 'Anthropic (Claude)',
            'o': 'OpenAI (ChatGPT)',
            'g': 'Google (Gemini)'
        }
    
    def parse_order(self, order_string: str) -> List[ExecutionStep]:
        """Parse order string into execution steps.
        
        Args:
            order_string: Order string like 'aog' or 'agoa'
            
        Returns:
            List of ExecutionStep objects
            
        Raises:
            ValueError: If order is invalid
        """
        if not order_string or not order_string.strip():
            raise ValueError("Order string cannot be empty")
        
        order = order_string.lower().strip()
        
        # Validate all characters are valid provider codes
        valid_codes = set(self.provider_map.keys())
        invalid_codes = set(order) - valid_codes
        if invalid_codes:
            raise ValueError(
                f"Invalid provider codes: {', '.join(sorted(invalid_codes))}. "
                f"Valid codes: {', '.join(sorted(valid_codes))}"
            )
        
        # Create execution steps
        steps = []
        total_steps = len(order)
        
        for i, code in enumerate(order):
            step = ExecutionStep(
                provider_code=code,
                provider_name=self.provider_map[code],
                step_number=i + 1,
                total_steps=total_steps,
                is_final=(i == total_steps - 1)
            )
            steps.append(step)
        
        return steps
    
    def validate_order(self, order_string: str) -> None:
        """Validate order string without parsing.
        
        Args:
            order_string: Order string to validate
            
        Raises:
            ValueError: If order is invalid
        """
        # This will raise ValueError if invalid
        self.parse_order(order_string)
    
    def get_order_summary(self, order_string: str) -> str:
        """Get human-readable summary of execution order.
        
        Args:
            order_string: Order string like 'aog'
            
        Returns:
            Human-readable summary like "Anthropic -> OpenAI -> Google"
        """
        steps = self.parse_order(order_string)
        names = [self.provider_display_names[step.provider_code] for step in steps]
        return " -> ".join(names)
    
    def get_provider_counts(self, order_string: str) -> Dict[str, int]:
        """Get count of how many times each provider appears.
        
        Args:
            order_string: Order string like 'agoa'
            
        Returns:
            Dict mapping provider codes to counts
        """
        steps = self.parse_order(order_string)
        counts = {}
        for step in steps:
            counts[step.provider_code] = counts.get(step.provider_code, 0) + 1
        return counts
    
    def analyze_order(self, order_string: str) -> Dict[str, any]:
        """Analyze order string and return detailed information.
        
        Args:
            order_string: Order string to analyze
            
        Returns:
            Dict with analysis information
        """
        steps = self.parse_order(order_string)
        provider_counts = self.get_provider_counts(order_string)
        
        return {
            'total_steps': len(steps),
            'unique_providers': len(set(step.provider_code for step in steps)),
            'provider_counts': provider_counts,
            'has_duplicates': any(count > 1 for count in provider_counts.values()),
            'execution_summary': self.get_order_summary(order_string),
            'final_provider': self.provider_display_names[steps[-1].provider_code] if steps else None
        }


# Global parser instance
order_parser = OrderParser()