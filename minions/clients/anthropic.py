import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import anthropic

from minions.usage import Usage


class AnthropicClient:
    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        stream: bool = False,
    ):
        """
        Initialize the Anthropic client.

        Args:
            model_name: The name of the model to use (default: "claude-3-sonnet-20240229")
            api_key: Anthropic API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.2)
            max_tokens: Maximum number of tokens to generate (default: 2048)
            stream: Whether to stream the response (default: False)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.logger = logging.getLogger("AnthropicClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def chat(self, messages: List[Dict[str, Any]], stream_callback=None, **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the Anthropic API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            stream_callback: Optional callback function to receive streaming chunks
            **kwargs: Additional arguments to pass to client.messages.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": self.stream,
                **kwargs,
            }

            if self.stream and stream_callback:
                # Handle streaming response
                response_content = ""
                prompt_tokens = 0
                completion_tokens = 0
                
                with self.client.messages.stream(**params) as stream:
                    for chunk in stream:
                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                            chunk_text = chunk.delta.text
                            response_content += chunk_text
                            stream_callback(chunk_text)
                
                # Since streaming doesn't provide token usage, we need to get it from a separate API call or estimate
                # For simplicity, we'll just create a minimal usage object
                usage = Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
                
                return [response_content], usage
            else:
                # Handle non-streaming response
                response = self.client.messages.create(**params)
                usage = Usage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens
                )
                return [response.content[0].text], usage
            
        except Exception as e:
            self.logger.error(f"Error during Anthropic API call: {e}")
            raise 