"""
Base classes for LLM clients.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class Usage:
    """
    Dataclass to track token usage statistics for LLM calls.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __add__(self, other):
        """Allow Usage objects to be added together."""
        if other is None:
            return self
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )
    
    def __radd__(self, other):
        """Support sum() function on lists of Usage objects."""
        if other == 0:
            return self
        return self.__add__(other)


class BaseClient:
    """
    Base class for all LLM clients.
    
    This class defines the interface that all LLM clients should implement.
    """
    
    def __init__(self, model_name: str = None, temperature: float = 0.0):
        """
        Initialize the client.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature for generation (0.0 to 1.0)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.supports_response_format = False
    
    def chat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Process a chat request with the model.
        
        Args:
            messages: A list of message dictionaries or a single message dictionary
            **kwargs: Additional arguments to pass to the underlying API
            
        Returns:
            A tuple of (responses, usage, completion_reasons)
        """
        raise NotImplementedError("Subclasses must implement chat()") 