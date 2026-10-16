import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_knowledge_platform.core.config import get_settings
from ai_knowledge_platform.db.base import Base
from ai_knowledge_platform.models.document import Document  # noqa: F401
from ai_knowledge_platform.models.user import User  # noqa: F401
from ai_knowledge_platform.models.user_credential import UserCredential  # noqa: F401

# Alembic Config object.
config = context.config

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

# Configure logging when a configuration file is available.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Importing the models registers their tables in Base.metadata.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without creating a database engine."""
    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
