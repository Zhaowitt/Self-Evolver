from src.config import get_config, reset_config


def test_configuration_is_cached_until_explicit_reset(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "first-test-model")
    reset_config()
    first = get_config()
    assert first.llm.model == "first-test-model"
    monkeypatch.setenv("OPENAI_MODEL", "second-test-model")
    assert get_config() is first
    reset_config()
    second = get_config()
    assert second is not first
    assert second.llm.model == "second-test-model"
