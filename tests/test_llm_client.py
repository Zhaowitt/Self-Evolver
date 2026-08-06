"""LLM request compatibility tests."""

from types import SimpleNamespace

from src.config import LLMConfig
from src.llm.client import LLMClient, Message


class _Completions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="OK", tool_calls=[])
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=None, model=kwargs["model"])


def _client(model: str):
    client = LLMClient(LLMConfig(api_key="test", model=model))
    completions = _Completions()
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return client, completions


def test_gpt5_uses_max_completion_tokens():
    client, completions = _client("gpt-5.4")

    client.chat([Message(role="user", content="hello")], max_tokens=64)

    assert completions.kwargs["max_completion_tokens"] == 64
    assert "max_tokens" not in completions.kwargs


def test_legacy_models_keep_max_tokens():
    client, completions = _client("gpt-4o-mini")

    client.chat([Message(role="user", content="hello")], max_tokens=64)

    assert completions.kwargs["max_tokens"] == 64
    assert "max_completion_tokens" not in completions.kwargs
