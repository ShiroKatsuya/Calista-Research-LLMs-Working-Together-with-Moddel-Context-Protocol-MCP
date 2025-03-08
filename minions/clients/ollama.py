import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import asyncio
import functools

from pydantic import BaseModel

from minions.clients.base import Usage


class OllamaClient:
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        num_ctx: int = 4096,
        structured_output_schema: Optional[BaseModel] = None,
        use_async: bool = False,
    ):
        """Initialize Ollama Client."""
        self.model_name = model_name
        self.logger = logging.getLogger("OllamaClient")
        self.logger.setLevel(logging.INFO)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.use_async = use_async

        # If we want structured schema output:
        self.format_structured_output = None
        if structured_output_schema:
            self.format_structured_output = structured_output_schema.model_json_schema()

        # For async calls - lazy initialized
        self._async_client = None
        
        # Cache for model availability check
        self._model_available = False

        # Ensure model is pulled
        self._ensure_model_available()

    @property
    def async_client(self):
        """Lazy initialization of async client."""
        if self._async_client is None and self.use_async:
            from ollama import AsyncClient
            self._async_client = AsyncClient()
        return self._async_client

    @functools.lru_cache(maxsize=1)
    def _prepare_options(self) -> Dict[str, Any]:
        """Prepare options for Ollama API call with caching."""
        options = {}
        
        if self.format_structured_output:
            options["format"] = "json"
            options["system"] = (
                f"Format your entire response as JSON. "
                f"Use the following JSON schema: {self.format_structured_output}"
            )
            
        return options

    def _ensure_model_available(self) -> None:
        """Ensure the specified model is available locally."""
        if self._model_available:
            return
            
        import ollama
        try:
            # Use a quick, minimal test to check if model is available
            ollama.chat(
                model=self.model_name,
                messages=[{"role": "system", "content": "test"}]
            )
            self._model_available = True
        except ollama.ResponseError as e:
            if "no model found with name" in str(e).lower():
                self.logger.info(f"Model {self.model_name} not found. Attempting to pull...")
                try:
                    ollama.pull(self.model_name)
                    self._model_available = True
                    self.logger.info(f"Successfully pulled model {self.model_name}")
                except Exception as pull_error:
                    self.logger.error(f"Failed to pull model {self.model_name}: {pull_error}")
                    raise
            else:
                self.logger.error(f"Error checking model availability: {e}")
                raise

    async def _achat_internal(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Internal async chat implementation. Takes a list of messages and returns responses.
        """
        # Import here to avoid importing if not needed
        import asyncio
        
        # Make sure we have the async client
        if not self.async_client:
            from ollama import AsyncClient
            self._async_client = AsyncClient()
            
        # If the user provided a single dictionary, wrap it
        if isinstance(messages, dict):
            messages = [messages]

        chat_kwargs = self._prepare_options()
        
        # Filter out temperature from kwargs as it's not supported by ollama.chat()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'temperature'}
        
        # Combine with any passed kwargs
        chat_kwargs.update(filtered_kwargs)

        async def process_one(msg):
            try:
                response = await self.async_client.chat(
                    model=self.model_name,
                    messages=msg,
                    **chat_kwargs,
                )
                return (
                    response["message"]["content"],
                    Usage(
                        prompt_tokens=response["prompt_eval_count"],
                        completion_tokens=response["eval_count"],
                    ),
                    response["done_reason"],
                )
            except Exception as e:
                self.logger.error(f"Error during async Ollama API call: {e}")
                raise

        # Use asyncio.gather for parallel processing if multiple message sets
        results = await asyncio.gather(*(process_one([m]) for m in messages))
        
        # Unzip the results
        contents, usages, done_reasons = zip(*results)
        
        # Sum up all usages
        total_usage = sum(usages, Usage())
        
        return list(contents), total_usage, list(done_reasons)

    async def achat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], List[Usage], List[str]]:
        """
        Handle asynchronous chat completions. If you pass a list of message dicts,
        we do one call for that entire conversation. If you pass a single dict,
        we wrap it in a list.
        """
        return await self._achat_internal(messages, **kwargs)

    def schat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Synchronous implementation of chat completions.
        
        Args:
            messages: Either a single message dict or a list of message dicts.
            **kwargs: Additional parameters to pass to the Ollama API.
            
        Returns:
            Tuple of (responses, usage, done_reasons)
        """
        # Import ollama here
        import ollama
        
        # Ensure messages is a list of dicts
        if isinstance(messages, dict):
            messages = [messages]

        # Now messages is a list of dicts, so we can pass it to Ollama in one go
        chat_kwargs = self._prepare_options()
        
        # Filter out temperature from kwargs as it's not supported by ollama.chat()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'temperature'}

        responses = []
        usage_total = Usage()
        done_reasons = []

        try:
            # Process all messages in a single call for efficiency
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                **chat_kwargs,
                **filtered_kwargs,
            )
            responses.append(response["message"]["content"])
            usage_total += Usage(
                prompt_tokens=response["prompt_eval_count"],
                completion_tokens=response["eval_count"],
            )
            done_reasons.append(response["done_reason"])

        except Exception as e:
            self.logger.error(f"Error during Ollama API call: {e}")
            raise

        return responses, usage_total, done_reasons
    
    def chat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions. Delegates to async or sync implementation based on configuration.
        """
        if self.use_async:
            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.achat(messages, **kwargs))
        else:
            return self.schat(messages, **kwargs)
