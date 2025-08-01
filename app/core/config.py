"""
Application configuration module.

Centralizes environment variables and runtime configuration.
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field
import os


class Settings(BaseSettings):
    """
    Global configuration settings.
    """

    PROJECT_NAME: str = "Sentiment Analysis API"
    API_VERSION: str = "1.0.0"

    MODEL_NAME: str = "bert-base-uncased"
    MODEL_PATH: str = "./saved_model"

    DEVICE: str = "cuda"  # "cpu" or "cuda"
    MAX_SEQUENCE_LENGTH: int = 256

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra='allow'
    )


def get_settings() -> Settings:
    """
    Returns application settings instance.
    Includes basic error handling.
    """
    try:
        return Settings()
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {str(e)}")