"""
Base Worker class for all agent workers.

Provides common functionality for LLM-based workers.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, TypeVar

from src.environment.models import ExecutionContext
from src.llm.client import LLMClient, LLMResponse, Message

logger = logging.getLogger(__name__)

# Type variable for worker results
T = TypeVar("T")


@dataclass
class WorkerResult(Generic[T]):
    """Generic result wrapper for worker outputs."""
    
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    llm_response: Optional[LLMResponse] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def tokens_used(self) -> int:
        """Get total tokens used in this result."""
        if self.llm_response:
            return self.llm_response.total_tokens
        return 0


class BaseWorker(ABC):
    """
    Abstract base class for all workers.
    
    Workers are specialized agents that perform specific tasks
    in the code repair workflow.
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the worker.
        
        Args:
            llm_client: LLM client for API calls. Creates new one if None.
            name: Worker name for logging. Uses class name if None.
        """
        self.llm_client = llm_client or LLMClient()
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for this worker."""
        pass
    
    @abstractmethod
    def execute(self, context: ExecutionContext) -> WorkerResult:
        """
        Execute the worker's task.
        
        Args:
            context: Execution context with issue and environment info.
            
        Returns:
            WorkerResult containing the output or error.
        """
        pass
    
    def _build_messages(
        self,
        user_message: str,
        additional_context: Optional[str] = None,
    ) -> list[Message]:
        """
        Build message list for LLM call.
        
        Args:
            user_message: The main user message.
            additional_context: Optional additional context to append.
            
        Returns:
            List of Message objects.
        """
        messages = [Message(role="system", content=self.system_prompt)]
        
        if additional_context:
            user_message = f"{user_message}\n\n{additional_context}"
        
        messages.append(Message(role="user", content=user_message))
        return messages
    
    def _call_llm(
        self,
        user_message: str,
        additional_context: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Make an LLM call with the worker's system prompt.
        
        Args:
            user_message: The user message to send.
            additional_context: Optional additional context.
            **kwargs: Additional arguments for the LLM call.
            
        Returns:
            LLMResponse from the API.
        """
        messages = self._build_messages(user_message, additional_context)
        self.logger.debug(f"Calling LLM with {len(messages)} messages")
        return self.llm_client.chat(messages, **kwargs)
    
    def _format_error_context(self, context: ExecutionContext) -> str:
        """
        Format previous errors for context.
        
        Args:
            context: Execution context with error history.
            
        Returns:
            Formatted string of previous errors.
        """
        if not context.previous_errors:
            return ""
        
        error_parts = ["## Previous Errors:"]
        for i, error in enumerate(context.previous_errors, 1):
            error_parts.append(f"\n### Attempt {i}:\n{error}")
        
        return "\n".join(error_parts)
    
    def _format_test_results(self, context: ExecutionContext) -> str:
        """
        Format test results for context.
        
        Args:
            context: Execution context with test history.
            
        Returns:
            Formatted string of test results.
        """
        if not context.test_results:
            return ""
        
        result_parts = ["## Test Results:"]
        for i, result in enumerate(context.test_results, 1):
            status = "PASSED" if result.passed else "FAILED"
            result_parts.append(f"\n### Attempt {i} ({status}):")
            if result.error_logs:
                result_parts.append(f"```\n{result.error_logs[:2000]}\n```")
            if result.output:
                result_parts.append(f"Output:\n```\n{result.output[:1000]}\n```")
        
        return "\n".join(result_parts)
