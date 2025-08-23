"""Mock responses for development and testing."""

import time
import random
from typing import Dict, Any

# Development mode toggle - set to False for production
DEV_MODE = False

# Mock response delays (in seconds) to simulate API calls
MOCK_DELAYS = {
    'claude': (1.0, 3.0),    # Random delay between 1-3 seconds
    'openai': (0.8, 2.5),    # Random delay between 0.8-2.5 seconds  
    'gemini': (1.2, 2.8),    # Random delay between 1.2-2.8 seconds
}

# Sample mock responses for different providers and scenarios
MOCK_RESPONSES = {
    'claude': {
        'intermediate': [
            "Looking at this question, I think we need to consider multiple dimensions. From a philosophical perspective, this touches on fundamental questions about existence and purpose. The classical approaches from ancient philosophy provide some foundational insights, while modern perspectives offer new frameworks for understanding.",
            
            "This is a fascinating question that deserves a multifaceted analysis. I'd like to approach this from both theoretical and practical angles. The research in this area suggests several key factors that we should examine carefully.",
            
            "Building on what we know from the literature, there are several important considerations here. The interdisciplinary nature of this question means we need to draw from multiple fields to get a complete picture."
        ],
        'final': [
            """Based on our collaborative discussion, I can provide a comprehensive synthesis:

The key insights from our analysis reveal several important points:

1. **Primary Considerations**: The foundational elements we've identified show that this question requires both theoretical understanding and practical application.

2. **Multiple Perspectives**: Each approach we've discussed contributes valuable insights, and the convergence of these viewpoints strengthens our overall understanding.

3. **Practical Implications**: The real-world applications of these concepts suggest actionable next steps and areas for further exploration.

In conclusion, the multifaceted nature of this question demonstrates the value of collaborative analysis, and the synthesis of different viewpoints provides a more robust and nuanced understanding than any single perspective could offer."""
        ]
    },
    
    'openai': {
        'intermediate': [
            "I'd like to add some additional context to this discussion. There are several empirical studies and data points that support different aspects of this question. The quantitative analysis shows some interesting patterns that complement the theoretical framework.",
            
            "From a systems thinking perspective, this question involves complex interactions between multiple variables. I think it's worth examining the feedback loops and emergent properties that arise from these interactions.",
            
            "The historical context here is quite relevant. Similar questions have been explored across different time periods and cultures, and those precedents offer valuable lessons for our current analysis."
        ],
        'final': [
            """Drawing together all the perspectives shared in this discussion, here's a comprehensive response:

**Executive Summary:**
Our collaborative analysis has revealed multiple layers of complexity in this question, with each contributing perspective adding crucial insights.

**Key Findings:**
• The theoretical frameworks provide essential grounding
• Empirical evidence supports multiple interpretations  
• Historical precedents offer valuable guidance
• Practical applications suggest clear next steps

**Recommendations:**
1. Consider the interdisciplinary approach as the most robust methodology
2. Balance theoretical insights with empirical validation
3. Account for contextual factors that may influence outcomes
4. Implement a phased approach for practical applications

This synthesis represents the collective wisdom of our roundtable discussion and provides a foundation for further exploration and implementation."""
        ]
    },
    
    'gemini': {
        'intermediate': [
            "I appreciate the thorough analysis so far. I'd like to contribute by highlighting some alternative viewpoints and potential counterarguments. It's important to consider edge cases and scenarios where the conventional wisdom might not apply.",
            
            "There are some interesting technological and contemporary angles to consider here. Recent developments in this field have opened up new possibilities that weren't available when earlier frameworks were developed.",
            
            "From a global perspective, different cultures and contexts might approach this question quite differently. These diverse viewpoints can enrich our understanding and help us avoid potential blind spots."
        ],
        'final': [
            """Synthesizing our roundtable discussion into a final comprehensive response:

## Integrated Analysis

Our collaborative exploration has produced a rich, multifaceted understanding of this complex question. Each participant has contributed unique insights that strengthen the overall analysis.

## Core Insights
- **Foundational Principles**: The theoretical groundwork establishes key concepts and relationships
- **Evidence Base**: Empirical data and historical precedents validate our analytical framework  
- **Alternative Perspectives**: Diverse viewpoints reveal assumptions and expand our conceptual boundaries
- **Contemporary Relevance**: Modern developments add new dimensions to traditional approaches

## Synthesis and Conclusions
The convergence of these different analytical lenses reveals that this question benefits significantly from collaborative examination. No single perspective captures the full complexity, but together they create a comprehensive understanding that is greater than the sum of its parts.

## Moving Forward
This integrated analysis provides a solid foundation for both theoretical understanding and practical application, demonstrating the power of diverse perspectives in tackling complex questions."""
        ]
    }
}


class MockLLMClient:
    """Mock LLM client for development and testing."""
    
    def __init__(self, provider: str, model: str):
        """Initialize mock client.
        
        Args:
            provider: Provider name (claude, openai, gemini)  
            model: Model name (used for logging but doesn't affect responses)
        """
        self.provider = provider
        self.model = model
    
    def generate_response(self, prompt: str, is_final: bool = False) -> Dict[str, Any]:
        """Generate a mock response.
        
        Args:
            prompt: Input prompt (not used in mock, but logged)
            is_final: Whether this is the final LLM in sequence
            
        Returns:
            Dict with response data
        """
        if not DEV_MODE:
            raise RuntimeError("Mock client called when DEV_MODE is False")
        
        # Simulate API delay
        delay_range = MOCK_DELAYS.get(self.provider, (1.0, 2.0))
        delay = random.uniform(*delay_range)
        time.sleep(delay)
        
        # Select response type and content
        response_type = 'final' if is_final else 'intermediate'
        responses = MOCK_RESPONSES.get(self.provider, {}).get(response_type, [])
        
        if not responses:
            # Fallback response
            if is_final:
                content = f"[{self.provider}] Final synthesis response (mock mode)"
            else:
                content = f"[{self.provider}] Intermediate discussion response (mock mode)"
        else:
            content = random.choice(responses)
        
        return {
            'content': content,
            'provider': self.provider,
            'model': self.model,
            'tokens_used': random.randint(150, 800),
            'response_time': delay,
            'is_final': is_final,
            'mock': True
        }


def get_mock_client(provider: str, model: str) -> MockLLMClient:
    """Factory function to create mock clients.
    
    Args:
        provider: Provider name
        model: Model name
        
    Returns:
        MockLLMClient instance
    """
    if not DEV_MODE:
        raise RuntimeError("Mock clients are only available in DEV_MODE")
    
    return MockLLMClient(provider, model)
