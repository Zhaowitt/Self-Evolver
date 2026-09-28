"""Client for OpenAI-compatible chat completion APIs."""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import openai
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import LLMConfig, get_config

logger = logging.getLogger(__name__)

_TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


@dataclass
class Message:
    role: str  # "system", "user", "assistant", or "tool"
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_config().llm
        self._client: Optional[OpenAI] = None
        self._total_tokens_used = 0

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            # Retries live in tenacity below; the SDK default (600 s x 3) can stall a GRPO step.
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", "120")),
                max_retries=0,
            )
        return self._client

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens_used

    def reset_token_count(self) -> None:
        self._total_tokens_used = 0

    @retry(
        # Only transient failures are retried; 400/401/parse errors fail fast.
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
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
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        message_dicts: List[Dict[str, Any]] = []
        for message in messages:
            item: Dict[str, Any] = {"role": message.role, "content": message.content}
            if message.tool_calls:
                item["tool_calls"] = message.tool_calls
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            message_dicts.append(item)

        logger.debug(f"Sending chat request to {model} with {len(messages)} messages")

        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": message_dicts,
            "temperature": temperature,
            **kwargs,
        }
        # GPT-5 models reject max_tokens in favor of max_completion_tokens.
        token_parameter = (
            "max_completion_tokens" if model.lower().startswith("gpt-5") else "max_tokens"
        )
        request_kwargs[token_parameter] = max_tokens

        response = self.client.chat.completions.create(
            **request_kwargs,
        )

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
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message),
        ]
        return self.chat(messages, **kwargs)
