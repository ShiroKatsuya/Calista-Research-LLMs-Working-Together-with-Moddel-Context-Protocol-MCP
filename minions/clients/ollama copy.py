import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import asyncio

from pydantic import BaseModel

from minions.usage import Usage


class OllamaClient:
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        num_ctx: int = 4096,
        structured_output_schema: Optional[BaseModel] = None,
        use_async: bool = False,
        stream = True,
    ):
        """Initialize Ollama Client."""
        self.model_name = model_name
        self.stream = stream
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

        # For async calls
        from ollama import AsyncClient
        self.client = AsyncClient() if use_async else None

        # Ensure model is pulled
        self._ensure_model_available()

    def _ensure_model_available(self):
        import ollama
        try:
            ollama.chat(
                model=self.model_name,
                stream=self.stream,
                messages=[{"role": "system", "content": "test"}]
            )
        except ollama.ResponseError as e:
            if e.status_code == 404:
                self.logger.info(
                    f"Model {self.model_name} not found locally. Pulling..."
                )
                ollama.pull(self.model_name)
                self.logger.info(f"Successfully pulled model {self.model_name}")
            else:
                raise

    def _prepare_options(self):
        """Common chat options for both sync and async calls."""
        opts = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "num_ctx": self.num_ctx,
        }
        chat_kwargs = {"options": opts}
        if self.format_structured_output:
            chat_kwargs["format"] = self.format_structured_output
        return chat_kwargs

    #
    #  ASYNC
    #
    def achat(
            self,
            messages: Union[List[Dict[str, Any]], Dict[str, Any]],
            stream_callback=None,
            **kwargs,
        ) -> Tuple[List[str], List[Usage], List[str]]:
            """
            Wrapper for async chat. Runs `asyncio.run()` internally to simplify usage.
            """
            if not self.use_async:
                raise RuntimeError("This client is not in async mode. Set `use_async=True`.")

            try:
                return asyncio.run(self._achat_internal(messages, stream_callback=stream_callback, **kwargs))
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    # Create a new event loop and set it as the current one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(self._achat_internal(messages, stream_callback=stream_callback, **kwargs))
                    finally:
                        loop.close()
                raise

    async def _achat_internal(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        stream_callback=None,
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle async chat with multiple messages in parallel.
        """
        # If the user provided a single dictionary, wrap it in a list.
        if isinstance(messages, dict):
            messages = [messages]

        # Now we have a list of dictionaries. We'll call them in parallel.
        chat_kwargs = self._prepare_options()
        async def process_one(msg):
            resp = await self.client.chat(
                model=self.model_name,
                messages=[msg],  # each call with exactly one message
                stream=False,  # Force non-streaming mode
                **chat_kwargs,
                **kwargs
            )
            
            return resp
        
        # Run them all in parallel
        results = await asyncio.gather(*(process_one(m) for m in messages))

        # Gather them back
        texts = []
        usage_total = Usage()
        done_reasons = []
        for r in results:
            # Handle ollama._types.ChatResponse
            if hasattr(r, 'message') and hasattr(r.message, 'content'):
                texts.append(r.message.content)
                
                # Get usage info if available
                if hasattr(r, 'prompt_eval_count') and hasattr(r, 'eval_count'):
                    usage_total += Usage(
                        prompt_tokens=r.prompt_eval_count,
                        completion_tokens=r.eval_count,
                    )
                if hasattr(r, 'done_reason'):
                    done_reasons.append(r.done_reason)
            elif isinstance(r, dict):
                texts.append(r["message"]["content"])
                usage_total += Usage(prompt_tokens=r["prompt_eval_count"],
                                 completion_tokens=r["eval_count"])
                done_reasons.append(r["done_reason"])

        # Handle streaming with callback if provided
        if self.stream and stream_callback and hasattr(results[-1], '__iter__'):
            content = ""
            for chunk in results[-1]:
                if 'message' in chunk and 'content' in chunk['message']:
                    chunk_content = chunk['message']['content']
                    # Only stream the new content
                    new_content = chunk_content[len(content):]
                    if new_content:
                        content = chunk_content
                        stream_callback(new_content)
            
            # Ensure we have the full response
            if content:
                texts.append(content)
            # Get usage info from the last chunk
            if 'prompt_eval_count' in chunk and 'eval_count' in chunk:
                usage_total += Usage(
                    prompt_tokens=chunk['prompt_eval_count'],
                    completion_tokens=chunk['eval_count'],
                )
            if 'done_reason' in chunk:
                done_reasons.append(chunk['done_reason'])

        # Ensure we always return at least one response
        if not texts:
            self.logger.warning("No responses generated in async method. Returning empty placeholder.")
            texts = [""]
            if not done_reasons:
                done_reasons = ["empty"]

        return texts, usage_total, done_reasons


    def schat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        stream_callback=None,
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle synchronous chat completions. If you pass a list of message dicts,
        we do one call for that entire conversation. If you pass a single dict,
        we wrap it in a list so there's no error.
        """
        import ollama
        # If the user provided a single dictionary, wrap it
        if isinstance(messages, dict):
            messages = [messages]

        # Now messages is a list of dicts, so we can pass it to Ollama in one go
        chat_kwargs = self._prepare_options()

        responses = []
        usage_total = Usage()
        done_reasons = []

        try:
            # We do one single call if you pass the entire conversation:
            #   messages=[{'role': 'user', 'content': ...},
            #             {'role': 'system', 'content': ...}, ...]
            # If you want multiple calls, you can either:
            #   (a) loop outside of this function, or
            #   (b) pass a list-of-lists approach that you handle similarly
            
            # Set stream to False to avoid the streaming issues
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=False,  # Force non-streaming mode
                **chat_kwargs,
                **kwargs,
            )
            
            # Handle non-streaming response
            
            # Handle ollama._types.ChatResponse
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                responses.append(response.message.content)
                
                # Get usage info if available
                if hasattr(response, 'prompt_eval_count') and hasattr(response, 'eval_count'):
                    usage_total += Usage(
                        prompt_tokens=response.prompt_eval_count,
                        completion_tokens=response.eval_count,
                    )
                if hasattr(response, 'done_reason'):
                    done_reasons.append(response.done_reason)
            elif isinstance(response, dict):
                if 'message' in response and 'content' in response['message']:
                    responses.append(response["message"]["content"])
                
                if 'prompt_eval_count' in response and 'eval_count' in response:
                    usage_total += Usage(
                        prompt_tokens=response["prompt_eval_count"],
                        completion_tokens=response["eval_count"],
                    )
                if 'done_reason' in response:
                    done_reasons.append(response["done_reason"])

        except Exception as e:
            self.logger.error(f"Error during Ollama API call: {e}")
            raise
            
        # Ensure we always return at least one response
        if not responses:
            self.logger.warning("No responses generated. Returning empty placeholder.")
            responses = [""]
            if not done_reasons:
                done_reasons = ["empty"]

        return responses, usage_total, done_reasons
    
    def chat(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        stream_callback=None,
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle synchronous chat completions. If you pass a list of message dicts,
        we do one call for that entire conversation. If you pass a single dict,
        we wrap it in a list so there's no error.
        """
        if self.use_async:
            return self.achat(messages, stream_callback=stream_callback, **kwargs)
        else:
            return self.schat(messages, stream_callback=stream_callback, **kwargs)
