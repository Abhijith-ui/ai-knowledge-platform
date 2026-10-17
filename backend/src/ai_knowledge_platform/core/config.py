from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    """

    # ==========================================================
    # Application
    # ==========================================================

    app_name: str = "AI Knowledge Platform"
    app_version: str = "0.1.0"
    app_description: str = "Production-grade AI Knowledge Platform API"

    # ==========================================================
    # PostgreSQL
    # ==========================================================

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ai_knowledge_platform"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    @property
    def database_url(self) -> str:
        """Build the SQLAlchemy database URL."""
        return (
            f"postgresql+psycopg://"
            f"{self.postgres_user}:"
            f"{self.postgres_password}@"
            f"{self.postgres_host}:"
            f"{self.postgres_port}/"
            f"{self.postgres_db}"
        )

    # ==========================================================
    # Document Storage and Upload Limits
    # ==========================================================

    document_storage_dir: Path = Path("var/documents")

    max_document_upload_bytes: int = Field(
        default=10 * 1024 * 1024,
        gt=0,
    )

    max_document_request_bytes: int = Field(
        default=11 * 1024 * 1024,
        gt=0,
    )

    # ==========================================================
    # Authentication
    # ==========================================================

    jwt_algorithm: str = "RS256"

    jwt_access_token_expire_minutes: int = Field(
        default=15,
        gt=0,
    )

    jwt_refresh_token_expire_days: int = Field(
        default=30,
        gt=0,
    )

    jwt_private_key_path: str = "secrets/jwt_private.pem"
    jwt_public_key_path: str = "secrets/jwt_public.pem"

    @field_validator("jwt_algorithm")
    @classmethod
    def validate_jwt_algorithm(cls, value: str) -> str:
        if value != "RS256":
            raise ValueError("Only RS256 is supported.")

        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
