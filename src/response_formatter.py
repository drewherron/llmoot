"""Response formatting and display utilities."""

import json
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .llm_client import LLMResponse
from .discussion_orchestrator import DiscussionResult


class OutputFormat(Enum):
    """Available output formats."""
    PLAIN = "plain"
    MARKDOWN = "markdown"
    JSON = "json"
    STRUCTURED = "structured"


@dataclass
class FormattedResponse:
    """A formatted response with metadata."""
    content: str
    format: OutputFormat
    timestamp: str
    metadata: Dict[str, Any]


class ResponseFormatter:
    """Formats discussion responses for display and export."""
    
    def __init__(self):
        """Initialize the response formatter."""
        self.provider_display_names = {
            'claude': 'Claude (Anthropic)',
            'openai': 'ChatGPT (OpenAI)', 
            'gemini': 'Gemini (Google)'
        }
    
    def format_discussion_result(self, result: DiscussionResult, 
                               format_type: OutputFormat = OutputFormat.PLAIN,
                               include_metadata: bool = True,
                               include_intermediate: bool = False) -> FormattedResponse:
        """
        Format a complete discussion result.
        
        Args:
            result: DiscussionResult to format
            format_type: Output format to use
            include_metadata: Whether to include response metadata
            include_intermediate: Whether to show intermediate responses
            
        Returns:
            FormattedResponse with formatted content
        """
        if format_type == OutputFormat.PLAIN:
            content = self._format_plain(result, include_metadata, include_intermediate)
        elif format_type == OutputFormat.MARKDOWN:
            content = self._format_markdown(result, include_metadata, include_intermediate)
        elif format_type == OutputFormat.JSON:
            content = self._format_json(result, include_metadata, include_intermediate)
        elif format_type == OutputFormat.STRUCTURED:
            content = self._format_structured(result, include_metadata, include_intermediate)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
        
        metadata = {
            'success': result.success,
            'total_responses': len(result.responses),
            'execution_steps': len(result.execution_steps),
            'final_provider': result.execution_steps[-1].provider_name if result.execution_steps else None,
            'error_message': result.error_message
        }
        
        return FormattedResponse(
            content=content,
            format=format_type,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        )
    
    def _format_plain(self, result: DiscussionResult, include_metadata: bool, 
                     include_intermediate: bool) -> str:
        """Format result as plain text."""
        lines = []
        
        if not result.success:
            lines.append(f"Discussion failed: {result.error_message}")
            return "\n".join(lines)
        
        if include_intermediate and result.responses:
            lines.append("=" * 60)
            lines.append("DISCUSSION TRANSCRIPT")
            lines.append("=" * 60)
            lines.append("")
            
            for i, (response, step) in enumerate(zip(result.responses, result.execution_steps)):
                provider_name = self.provider_display_names.get(step.provider_name, step.provider_name)
                is_final = i == len(result.responses) - 1
                
                if is_final:
                    lines.append(f"FINAL RESPONSE - {provider_name}")
                else:
                    lines.append(f"STEP {i+1} - {provider_name}")
                
                lines.append("-" * 40)
                lines.append(response.content)
                
                if include_metadata:
                    lines.append("")
                    lines.append(f"Model: {response.model}")
                    if hasattr(response, 'tokens_used') and response.tokens_used:
                        lines.append(f"Tokens: {response.tokens_used}")
                    if hasattr(response, 'response_time') and response.response_time:
                        lines.append(f"Time: {response.response_time:.2f}s")
                
                lines.append("")
                lines.append("")
        else:
            lines.append("=" * 60)
            lines.append("FINAL RESPONSE")
            lines.append("=" * 60)
            lines.append("")
            lines.append(result.final_response)
            
            if include_metadata and result.responses:
                final_response = result.responses[-1]
                final_step = result.execution_steps[-1]
                provider_name = self.provider_display_names.get(final_step.provider_name, final_step.provider_name)
                
                lines.append("")
                lines.append("-" * 40)
                lines.append(f"Provider: {provider_name}")
                lines.append(f"Model: {final_response.model}")
                if hasattr(final_response, 'tokens_used') and final_response.tokens_used:
                    lines.append(f"Tokens: {final_response.tokens_used}")
                if hasattr(final_response, 'response_time') and final_response.response_time:
                    lines.append(f"Time: {final_response.response_time:.2f}s")
                lines.append(f"Total steps: {len(result.execution_steps)}")
        
        return "\n".join(lines)
    
    def _format_markdown(self, result: DiscussionResult, include_metadata: bool,
                        include_intermediate: bool) -> str:
        """Format result as Markdown."""
        lines = []
        
        if not result.success:
            lines.append(f"# Discussion Failed")
            lines.append("")
            lines.append(f"**Error:** {result.error_message}")
            return "\n".join(lines)
        
        if include_intermediate and result.responses:
            lines.append("# Discussion Transcript")
            lines.append("")
            
            for i, (response, step) in enumerate(zip(result.responses, result.execution_steps)):
                provider_name = self.provider_display_names.get(step.provider_name, step.provider_name)
                is_final = i == len(result.responses) - 1
                
                if is_final:
                    lines.append(f"## Final Response - {provider_name}")
                else:
                    lines.append(f"## Step {i+1} - {provider_name}")
                
                lines.append("")
                lines.append(response.content)
                
                if include_metadata:
                    lines.append("")
                    lines.append("### Metadata")
                    lines.append(f"- **Model:** {response.model}")
                    if hasattr(response, 'tokens_used') and response.tokens_used:
                        lines.append(f"- **Tokens:** {response.tokens_used}")
                    if hasattr(response, 'response_time') and response.response_time:
                        lines.append(f"- **Response Time:** {response.response_time:.2f}s")
                
                lines.append("")
                lines.append("---")
                lines.append("")
        else:
            lines.append("# Final Response")
            lines.append("")
            lines.append(result.final_response)
            
            if include_metadata and result.responses:
                final_response = result.responses[-1]
                final_step = result.execution_steps[-1]
                provider_name = self.provider_display_names.get(final_step.provider_name, final_step.provider_name)
                
                lines.append("")
                lines.append("## Metadata")
                lines.append(f"- **Provider:** {provider_name}")
                lines.append(f"- **Model:** {final_response.model}")
                if hasattr(final_response, 'tokens_used') and final_response.tokens_used:
                    lines.append(f"- **Tokens:** {final_response.tokens_used}")
                if hasattr(final_response, 'response_time') and final_response.response_time:
                    lines.append(f"- **Response Time:** {final_response.response_time:.2f}s")
                lines.append(f"- **Total Steps:** {len(result.execution_steps)}")
        
        return "\n".join(lines)
    
    def _format_json(self, result: DiscussionResult, include_metadata: bool,
                    include_intermediate: bool) -> str:
        """Format result as JSON."""
        data = {
            'success': result.success,
            'final_response': result.final_response,
            'error_message': result.error_message,
            'timestamp': datetime.now().isoformat(),
            'execution_summary': {
                'total_steps': len(result.execution_steps),
                'providers': [step.provider_name for step in result.execution_steps],
                'final_provider': result.execution_steps[-1].provider_name if result.execution_steps else None
            }
        }
        
        if include_intermediate:
            data['responses'] = []
            for i, (response, step) in enumerate(zip(result.responses, result.execution_steps)):
                response_data = {
                    'step_number': i + 1,
                    'provider': step.provider_name,
                    'content': response.content,
                    'is_final': step.is_final
                }
                
                if include_metadata:
                    response_data['metadata'] = {
                        'model': response.model,
                        'tokens_used': getattr(response, 'tokens_used', None),
                        'response_time': getattr(response, 'response_time', None)
                    }
                
                data['responses'].append(response_data)
        
        if include_metadata and result.responses:
            final_response = result.responses[-1]
            data['metadata'] = {
                'final_model': final_response.model,
                'final_tokens': getattr(final_response, 'tokens_used', None),
                'final_response_time': getattr(final_response, 'response_time', None),
                'total_tokens': sum(getattr(r, 'tokens_used', 0) for r in result.responses),
                'total_time': sum(getattr(r, 'response_time', 0) for r in result.responses)
            }
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def _format_structured(self, result: DiscussionResult, include_metadata: bool,
                          include_intermediate: bool) -> str:
        """Format result in structured format with clear sections."""
        lines = []
        
        if not result.success:
            lines.append("╔═══════════════════════════════════════════════╗")
            lines.append("║                DISCUSSION FAILED               ║")
            lines.append("╚═══════════════════════════════════════════════╝")
            lines.append("")
            lines.append(f"Error: {result.error_message}")
            return "\n".join(lines)
        
        # Header
        lines.append("╔═══════════════════════════════════════════════╗")
        lines.append("║             LLMOOT DISCUSSION RESULT          ║")
        lines.append("╚═══════════════════════════════════════════════╝")
        lines.append("")
        
        # Execution summary
        if result.execution_steps:
            provider_chain = " → ".join(
                self.provider_display_names.get(step.provider_name, step.provider_name) 
                for step in result.execution_steps
            )
            lines.append(f"Execution Order: {provider_chain}")
            lines.append(f"Total Steps: {len(result.execution_steps)}")
            lines.append("")
        
        if include_intermediate and result.responses:
            for i, (response, step) in enumerate(zip(result.responses, result.execution_steps)):
                provider_name = self.provider_display_names.get(step.provider_name, step.provider_name)
                is_final = i == len(result.responses) - 1
                
                if is_final:
                    lines.append("┌─────────────── FINAL RESPONSE ───────────────┐")
                else:
                    lines.append(f"┌─────────────── STEP {i+1:2d} RESPONSE ───────────────┐")
                
                lines.append(f"│ Provider: {provider_name:<35} │")
                if include_metadata:
                    lines.append(f"│ Model: {response.model:<38} │")
                lines.append("└───────────────────────────────────────────────┘")
                lines.append("")
                
                # Content with indentation
                content_lines = response.content.split('\n')
                for content_line in content_lines:
                    if content_line.strip():
                        lines.append(f"  {content_line}")
                    else:
                        lines.append("")
                
                if include_metadata:
                    lines.append("")
                    lines.append("  Metadata:")
                    if hasattr(response, 'tokens_used') and response.tokens_used:
                        lines.append(f"    Tokens: {response.tokens_used}")
                    if hasattr(response, 'response_time') and response.response_time:
                        lines.append(f"    Time: {response.response_time:.2f}s")
                
                lines.append("")
                if not is_final:
                    lines.append("─" * 50)
                    lines.append("")
        else:
            lines.append("┌─────────────── FINAL RESPONSE ───────────────┐")
            if result.execution_steps:
                final_step = result.execution_steps[-1]
                provider_name = self.provider_display_names.get(final_step.provider_name, final_step.provider_name)
                lines.append(f"│ Provider: {provider_name:<35} │")
            lines.append("└───────────────────────────────────────────────┘")
            lines.append("")
            
            # Content with indentation
            content_lines = result.final_response.split('\n')
            for content_line in content_lines:
                if content_line.strip():
                    lines.append(f"  {content_line}")
                else:
                    lines.append("")
            
            if include_metadata and result.responses:
                final_response = result.responses[-1]
                lines.append("")
                lines.append("  Metadata:")
                lines.append(f"    Model: {final_response.model}")
                if hasattr(final_response, 'tokens_used') and final_response.tokens_used:
                    lines.append(f"    Tokens: {final_response.tokens_used}")
                if hasattr(final_response, 'response_time') and final_response.response_time:
                    lines.append(f"    Time: {final_response.response_time:.2f}s")
        
        return "\n".join(lines)
    
    def generate_summary(self, result: DiscussionResult) -> str:
        """Generate a brief summary of the discussion result."""
        if not result.success:
            return f"Discussion failed: {result.error_message}"
        
        if not result.execution_steps:
            return "No execution steps found"
        
        provider_names = [
            self.provider_display_names.get(step.provider_name, step.provider_name)
            for step in result.execution_steps
        ]
        
        summary_parts = [
            f"Discussion completed successfully",
            f"Order: {' → '.join(provider_names)}",
            f"Steps: {len(result.execution_steps)}",
            f"Final response: {len(result.final_response)} characters"
        ]
        
        if result.responses:
            total_tokens = sum(getattr(r, 'tokens_used', 0) for r in result.responses if hasattr(r, 'tokens_used'))
            total_time = sum(getattr(r, 'response_time', 0) for r in result.responses if hasattr(r, 'response_time'))
            
            if total_tokens > 0:
                summary_parts.append(f"Total tokens: {total_tokens}")
            if total_time > 0:
                summary_parts.append(f"Total time: {total_time:.1f}s")
        
        return " | ".join(summary_parts)
    
    def export_to_file(self, formatted_response: FormattedResponse, 
                      filename: str) -> bool:
        """
        Export formatted response to a file.
        
        Args:
            formatted_response: FormattedResponse to export
            filename: Target filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(formatted_response.content)
                
                # Add metadata as comments for non-JSON formats
                if formatted_response.format != OutputFormat.JSON:
                    f.write(f"\n\n# Generated by llmoot on {formatted_response.timestamp}\n")
                    f.write(f"# Format: {formatted_response.format.value}\n")
                    if formatted_response.metadata.get('success'):
                        f.write(f"# Total responses: {formatted_response.metadata['total_responses']}\n")
                        f.write(f"# Final provider: {formatted_response.metadata['final_provider']}\n")
            
            return True
        except Exception as e:
            print(f"Failed to export to {filename}: {e}")
            return False