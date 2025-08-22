"""Comprehensive logging system for llmoot discussions."""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from .llm_client import LLMRequest, LLMResponse
from .order_parser import ExecutionStep


@dataclass
class LogEntry:
    """Single log entry for a discussion event."""
    timestamp: str
    event_type: str  # 'request', 'response', 'error', 'discussion_start', 'discussion_end'
    step_number: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DiscussionLogger:
    """Handles logging for llmoot discussions with timestamped files."""
    
    def __init__(self, log_dir: str = "logs", enable_console: bool = False):
        """
        Initialize the discussion logger.
        
        Args:
            log_dir: Directory to store log files
            enable_console: Whether to also log to console
        """
        self.log_dir = Path(log_dir)
        self.enable_console = enable_console
        self.session_id = self._generate_session_id()
        self.log_file_path = self._setup_log_file()
        self.log_entries: List[LogEntry] = []
        
        # Setup Python logging
        self._setup_python_logging()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID based on timestamp."""
        return datetime.now().strftime("llmoot_%Y-%m-%d_%H-%M-%S")
    
    def _setup_log_file(self) -> Path:
        """Create log directory and file path."""
        # Ensure log directory exists
        self.log_dir.mkdir(exist_ok=True)
        
        # Create log file path
        log_filename = f"{self.session_id}.log"
        return self.log_dir / log_filename
    
    def _setup_python_logging(self):
        """Setup Python logging to file and optionally console."""
        # Create logger
        self.logger = logging.getLogger(f'llmoot.{self.session_id}')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler (optional)
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def log_discussion_start(self, prompt: str, order: str, quality_level: int, 
                           execution_steps: List[ExecutionStep], attribution: bool = False):
        """Log the start of a discussion."""
        metadata = {
            'prompt': prompt,
            'order': order,
            'quality_level': quality_level,
            'total_steps': len(execution_steps),
            'execution_plan': [
                {
                    'step': step.step_number,
                    'provider': step.provider_name,
                    'is_final': step.is_final
                } for step in execution_steps
            ],
            'attribution_enabled': attribution
        }
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            event_type='discussion_start',
            content=prompt,
            metadata=metadata
        )
        
        self._write_log_entry(entry)
        self.logger.info(f"Discussion started - Order: {order}, Steps: {len(execution_steps)}")
    
    def log_request(self, step_number: int, provider: str, model: str, 
                   request: LLMRequest, execution_step: ExecutionStep):
        """Log an LLM request."""
        metadata = {
            'model': model,
            'temperature': request.temperature,
            'max_tokens': request.max_tokens,
            'is_final': execution_step.is_final,
            'system_prompt': request.system_prompt,
            'prompt_length': len(request.prompt)
        }
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            event_type='request',
            step_number=step_number,
            provider=provider,
            model=model,
            content=request.prompt,
            metadata=metadata
        )
        
        self._write_log_entry(entry)
        self.logger.info(f"Request sent to {provider} ({model}) - Step {step_number}")
    
    def log_response(self, step_number: int, provider: str, response: LLMResponse):
        """Log an LLM response."""
        metadata = {
            'model': response.model,
            'tokens_used': response.tokens_used,
            'response_time': response.response_time,
            'is_final': response.is_final,
            'content_length': len(response.content),
            'has_error': response.error is not None
        }
        
        if response.error:
            metadata['error'] = response.error
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            event_type='response',
            step_number=step_number,
            provider=provider,
            model=response.model,
            content=response.content,
            metadata=metadata
        )
        
        self._write_log_entry(entry)
        
        if response.error:
            self.logger.warning(f"Response from {provider} with error - Step {step_number}: {response.error}")
        else:
            self.logger.info(f"Response received from {provider} - Step {step_number}, {response.tokens_used} tokens, {response.response_time:.2f}s")
    
    def log_error(self, step_number: int, provider: str, error: Exception, 
                 recovery_attempted: bool = False, recovery_successful: bool = False):
        """Log an error during discussion."""
        metadata = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'recovery_attempted': recovery_attempted,
            'recovery_successful': recovery_successful
        }
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            event_type='error',
            step_number=step_number,
            provider=provider,
            content=str(error),
            metadata=metadata
        )
        
        self._write_log_entry(entry)
        self.logger.error(f"Error in step {step_number} ({provider}): {error}")
    
    def log_discussion_end(self, success: bool, final_response: str, 
                          total_responses: int, error_message: Optional[str] = None,
                          partial_success: bool = False):
        """Log the end of a discussion."""
        metadata = {
            'success': success,
            'partial_success': partial_success,
            'total_responses': total_responses,
            'final_response_length': len(final_response) if final_response else 0,
            'session_duration': self._get_session_duration()
        }
        
        if error_message:
            metadata['error_message'] = error_message
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            event_type='discussion_end',
            content=final_response if final_response else error_message,
            metadata=metadata
        )
        
        self._write_log_entry(entry)
        
        status = "completed successfully"
        if partial_success:
            status = "completed with errors"
        elif not success:
            status = "failed"
        
        self.logger.info(f"Discussion {status} - {total_responses} responses, Duration: {metadata['session_duration']}")
    
    def _write_log_entry(self, entry: LogEntry):
        """Write a log entry to the structured log file."""
        self.log_entries.append(entry)
        
        # Also write as JSON to a structured log file
        json_log_path = self.log_file_path.with_suffix('.jsonl')
        
        with open(json_log_path, 'a', encoding='utf-8') as f:
            json.dump(asdict(entry), f, ensure_ascii=False)
            f.write('\n')
    
    def _get_session_duration(self) -> str:
        """Calculate session duration from first log entry."""
        if not self.log_entries:
            return "0s"
        
        start_time = datetime.fromisoformat(self.log_entries[0].timestamp)
        end_time = datetime.now()
        duration = end_time - start_time
        
        total_seconds = int(duration.total_seconds())
        minutes, seconds = divmod(total_seconds, 60)
        
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Generate a summary of the current logging session."""
        if not self.log_entries:
            return {"session_id": self.session_id, "entries": 0}
        
        entry_counts = {}
        providers_used = set()
        total_tokens = 0
        errors = 0
        
        for entry in self.log_entries:
            entry_counts[entry.event_type] = entry_counts.get(entry.event_type, 0) + 1
            
            if entry.provider:
                providers_used.add(entry.provider)
            
            if entry.event_type == 'response' and entry.metadata:
                total_tokens += entry.metadata.get('tokens_used', 0)
            
            if entry.event_type == 'error':
                errors += 1
        
        return {
            'session_id': self.session_id,
            'log_file': str(self.log_file_path),
            'structured_log': str(self.log_file_path.with_suffix('.jsonl')),
            'total_entries': len(self.log_entries),
            'entry_types': entry_counts,
            'providers_used': list(providers_used),
            'total_tokens': total_tokens,
            'errors': errors,
            'session_duration': self._get_session_duration()
        }
    
    def export_discussion_log(self, export_path: Optional[str] = None) -> str:
        """Export the complete discussion log to a file."""
        if export_path is None:
            export_path = self.log_dir / f"{self.session_id}_complete.json"
        else:
            export_path = Path(export_path)
        
        log_data = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': self.log_entries[0].timestamp if self.log_entries else None,
                'end_time': datetime.now().isoformat(),
                'duration': self._get_session_duration()
            },
            'summary': self.get_log_summary(),
            'entries': [asdict(entry) for entry in self.log_entries]
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        return str(export_path)


class LoggingManager:
    """Manages logging configuration and logger instances."""
    
    def __init__(self):
        """Initialize the logging manager."""
        self._current_logger: Optional[DiscussionLogger] = None
        self._log_dir = "logs"
        self._console_logging = False
    
    def configure(self, log_dir: str = "logs", enable_console: bool = False):
        """Configure logging settings."""
        self._log_dir = log_dir
        self._console_logging = enable_console
    
    def start_discussion_logging(self) -> DiscussionLogger:
        """Start logging for a new discussion."""
        self._current_logger = DiscussionLogger(
            log_dir=self._log_dir,
            enable_console=self._console_logging
        )
        return self._current_logger
    
    def get_current_logger(self) -> Optional[DiscussionLogger]:
        """Get the current discussion logger."""
        return self._current_logger
    
    def list_log_files(self) -> List[Dict[str, Any]]:
        """List all log files in the log directory."""
        log_path = Path(self._log_dir)
        if not log_path.exists():
            return []
        
        log_files = []
        for log_file in log_path.glob("llmoot_*.log"):
            try:
                stat = log_file.stat()
                log_files.append({
                    'filename': log_file.name,
                    'path': str(log_file),
                    'size': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception:
                continue
        
        return sorted(log_files, key=lambda x: x['created'], reverse=True)


# Global logging manager instance
logging_manager = LoggingManager()