"""
Unified LLM Client for Self-Evolver.

Provides a consistent interface for interacting with OpenAI-compatible LLM APIs.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import LLMConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant", or "tool"
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None


@dataclass
class LLMResponse:
    """Response from LLM API call."""

    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)
    
    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)
    
    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


class LLMClient:
    """
    Unified LLM client supporting OpenAI-compatible APIs.
    
    Usage:
        client = LLMClient()
        response = client.chat([
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello!")
        ])
        print(response.content)
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM client.
        
        Args:
            config: LLM configuration. If None, uses global config.
        """
        self.config = config or get_config().llm
        self._client: Optional[OpenAI] = None
        self._total_tokens_used = 0
    
    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        return self._client
    
    @property
    def total_tokens_used(self) -> int:
        """Total tokens used across all calls."""
        return self._total_tokens_used
    
    def reset_token_count(self) -> None:
        """Reset the token counter."""
        self._total_tokens_used = 0
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            messages: List of messages in the conversation.
            model: Model to use. Defaults to config model.
            temperature: Sampling temperature. Defaults to config temperature.
            max_tokens: Maximum tokens in response. Defaults to config max_tokens.
            **kwargs: Additional parameters passed to the API.
            
        Returns:
            LLMResponse containing the model's response.
        """
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        # Convert Message objects to OpenAI chat message dicts.
        message_dicts: List[Dict[str, Any]] = []
        for message in messages:
            item: Dict[str, Any] = {"role": message.role, "content": message.content}
            if message.tool_calls:
                item["tool_calls"] = message.tool_calls
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            message_dicts.append(item)
        
        logger.debug(f"Sending chat request to {model} with {len(messages)} messages")
        
        response = self.client.chat.completions.create(
            model=model,
            messages=message_dicts,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        # Extract response data
        choice = response.choices[0]
        response_message = choice.message
        tool_calls = []
        for tool_call in response_message.tool_calls or []:
            tool_calls.append({
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            })
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        } if response.usage else {}
        
        # Update total token count
        self._total_tokens_used += usage.get("total_tokens", 0)
        
        logger.debug(f"Received response: {usage.get('total_tokens', 0)} tokens used")
        
        return LLMResponse(
            content=response_message.content or "",
            model=response.model,
            usage=usage,
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
        )
    
    def chat_with_system(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Convenience method for simple system + user message pattern.
        
        Args:
            system_prompt: The system prompt.
            user_message: The user's message.
            **kwargs: Additional parameters passed to chat().
            
        Returns:
            LLMResponse containing the model's response.
        """
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message),
        ]
        return self.chat(messages, **kwargs)
