"""
Configuration management for OdinOracle.
Uses pydantic-settings for type-safe, centralized configuration.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Uses pydantic-settings for validation and type safety.
    """

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'  # Allow extra fields in .env for flexibility
    )

    # Database
    database_url: str = "sqlite:///odin_oracle.db"
    db_echo: bool = False

    # OpenAI / Cloud LLM Configuration
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    openai_base_url: Optional[str] = None

    # Local LLM Configuration (Ollama)
    local_model: str = "qwen2.5:14b"
    local_llm_url: str = "http://localhost:11434/v1"

    # Email / SMTP Configuration
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    from_email: Optional[str] = None

    @property
    def email_from(self) -> Optional[str]:
        """Get the from email address, defaulting to smtp_username."""
        return self.from_email or self.smtp_username

    @property
    def is_email_configured(self) -> bool:
        """Check if email service is properly configured."""
        return all([
            self.smtp_username,
            self.smtp_password,
            self.email_from
        ])

    @property
    def is_openai_configured(self) -> bool:
        """Check if OpenAI cloud mode is properly configured."""
        return all([
            self.openai_api_key,
            self.openai_model
        ])


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment (useful for testing)."""
    global _settings
    _settings = Settings()
    return _settings
