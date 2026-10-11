"""Configuration loaded from environment variables and .env."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class LLMConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))
    base_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS_PER_CALL", "4096"))
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0"))
    )


class ControllerConfig(BaseModel):
    api_key: str = Field(
        default_factory=lambda: os.getenv(
            "CONTROLLER_API_KEY",
            os.getenv("OPENAI_API_KEY", ""),
        )
    )
    model: str = Field(
        default_factory=lambda: os.getenv(
            "CONTROLLER_MODEL",
            os.getenv("OPENAI_MODEL", "gpt-4o"),
        )
    )
    base_url: Optional[str] = Field(
        default_factory=lambda: os.getenv(
            "CONTROLLER_BASE_URL",
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("CONTROLLER_MAX_TOKENS", "1024"))
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("CONTROLLER_TEMPERATURE", "0.2"))
    )


class AgentConfig(BaseModel):
    max_iterations: int = Field(
        default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "3"))
    )
    timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_TIMEOUT", "300"))
    )
    inspector_max_tool_calls: int = Field(
        default_factory=lambda: int(os.getenv("INSPECTOR_MAX_TOOL_CALLS", "8"))
    )
    inspector_read_max_lines: int = Field(
        default_factory=lambda: int(os.getenv("INSPECTOR_READ_MAX_LINES", "400"))
    )
    inspector_read_max_chars: int = Field(
        default_factory=lambda: int(os.getenv("INSPECTOR_READ_MAX_CHARS", "20000"))
    )
    patch_generator_max_tool_calls: int = Field(
        default_factory=lambda: int(os.getenv("PATCH_GENERATOR_MAX_TOOL_CALLS", "2"))
    )


class EnvironmentConfig(BaseModel):
    workspace_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("WORKSPACE_DIR", "./workspace"))
    )
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )


class DockerConfig(BaseModel):
    """Docker configuration for SWE-bench."""

    timeout: int = Field(
        default_factory=lambda: int(os.getenv("DOCKER_TIMEOUT", "600"))
    )


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)

    def validate_api_key(self) -> bool:
        return bool(self.llm.api_key and self.llm.api_key != "sk-your-api-key-here")


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Clear the lazy configuration cache (used to isolate tests)."""
    global _config
    _config = None
