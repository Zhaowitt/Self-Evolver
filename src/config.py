"""
Configuration management for Self-Evolver.

Loads configuration from environment variables and .env file.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


# Load .env file if exists
load_dotenv()


class LLMConfig(BaseModel):
    """LLM configuration settings."""
    
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


class AgentConfig(BaseModel):
    """Agent behavior configuration."""
    
    max_iterations: int = Field(
        default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "3"))
    )
    timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_TIMEOUT", "300"))
    )


class EnvironmentConfig(BaseModel):
    """Environment configuration."""
    
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
    image_prefix: str = Field(default="swebench")


class Config(BaseModel):
    """Main configuration container."""
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    
    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment."""
        return cls()
    
    def validate_api_key(self) -> bool:
        """Check if API key is configured."""
        return bool(self.llm.api_key and self.llm.api_key != "sk-your-api-key-here")


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def reset_config() -> None:
    """Reset the global configuration (mainly for testing)."""
    global _config
    _config = None
