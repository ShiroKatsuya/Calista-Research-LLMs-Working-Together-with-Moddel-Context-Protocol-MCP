import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import openai

from minions.usage import Usage


# TODO: define one dataclass for what is returned from all the clients
class OpenAIClient:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stream: bool = False,
    ):
        """
        Initialize the OpenAI client.

        Args:
            model_name: The name of the model to use (default: "gpt-4o")
            api_key: OpenAI API key (optional, falls back to environment variable if not provided)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum number of tokens to generate (default: 4096)
            stream: Whether to stream the response (default: False)
        """
        self.model_name = model_name
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.logger = logging.getLogger("OpenAIClient")
        self.logger.setLevel(logging.INFO)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream

    def chat(self, messages: List[Dict[str, Any]], stream_callback=None, **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the OpenAI API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            stream_callback: Optional callback function to receive streaming chunks
            **kwargs: Additional arguments to pass to openai.chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
                "stream": self.stream,
                **kwargs,
            }

            # Only add temperature if NOT using the reasoning models (e.g., o3-mini model)
            if "o1" not in self.model_name and "o3" not in self.model_name:
                params["temperature"] = self.temperature

            if self.stream and stream_callback:
                # Handle streaming response
                response_content = ""
                prompt_tokens = 0
                completion_tokens = 0
                
                response = openai.chat.completions.create(**params)
                for chunk in response:
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content is not None:
                            response_content += delta.content
                            stream_callback(delta.content)
                    
                    # Update token count if available
                    if hasattr(chunk, 'usage') and chunk.usage is not None:
                        prompt_tokens = getattr(chunk.usage, 'prompt_tokens', 0)
                        completion_tokens = getattr(chunk.usage, 'completion_tokens', 0)
                
                # Create usage object with accumulated tokens
                usage = Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
                
                return [response_content], usage
            else:
                # Handle non-streaming response
                response = openai.chat.completions.create(**params)
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )
                return [choice.message.content for choice in response.choices], usage
                
        except Exception as e:
            self.logger.error(f"Error during OpenAI API call: {e}")
            raise