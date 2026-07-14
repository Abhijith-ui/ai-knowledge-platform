from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    """

    app_name: str = "AI Knowledge Platform"
    app_version: str = "0.1.0"
    app_description: str = "Production-grade AI Knowledge Platform API"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Return a cached Settings instance.
    """
    return Settings()
