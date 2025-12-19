"""
Telemetry module for LLM client.
Provides structured logging and metrics collection.
"""

import json
import logging
import time
from typing import Any, Dict, Optional
from functools import wraps

from llm_baseclient.logger import get_logger

logger = get_logger()


class TelemetryCollector:
    """Collects and logs telemetry data for LLM operations."""

    def __init__(self):
        self.logger = get_logger("telemetry")
        self.metrics = {}

    def log_request(
        self,
        model: str,
        provider: str,
        request_type: str,
        request_id: str,
        start_time: float,
        end_time: float,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log request details and metrics."""
        duration = end_time - start_time
        log_data = {
            "request_id": request_id,
            "model": model,
            "provider": provider,
            "request_type": request_type,
            "duration_seconds": duration,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": error,
        }
        
        if error:
            self.logger.error(f"LLM Request failed: {json.dumps(log_data)}")
        else:
            self.logger.info(f"LLM Request completed: {json.dumps(log_data)}")
            
        # Store metrics for potential aggregation
        self.metrics[request_id] = log_data

    def get_metrics(self) -> Dict[str, Any]:
        """Return collected metrics."""
        return self.metrics.copy()

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        self.metrics.clear()


# Global telemetry collector instance
telemetry_collector = TelemetryCollector()


def with_telemetry(request_type: str):
    """Decorator to add telemetry to LLM client methods."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = f"{request_type}_{int(start_time * 1000000)}"
            
            # Try to extract model info from arguments
            model = kwargs.get('model', 'unknown')
            provider = model.split('/')[0] if '/' in model else 'unknown'
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                
                # Try to extract token usage from result
                input_tokens = None
                output_tokens = None
                if hasattr(result, 'usage'):
                    input_tokens = getattr(result.usage, 'prompt_tokens', None)
                    output_tokens = getattr(result.usage, 'completion_tokens', None)
                
                telemetry_collector.log_request(
                    model=model,
                    provider=provider,
                    request_type=request_type,
                    request_id=request_id,
                    start_time=start_time,
                    end_time=end_time,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )
                
                return result
            except Exception as e:
                end_time = time.time()
                telemetry_collector.log_request(
                    model=model,
                    provider=provider,
                    request_type=request_type,
                    request_id=request_id,
                    start_time=start_time,
                    end_time=end_time,
                    error=str(e)
                )
                raise
        return wrapper
    return decorator
