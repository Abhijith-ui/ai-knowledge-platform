import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Add backend/src to Python's import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_knowledge_platform import models  # noqa: F401
from ai_knowledge_platform.core.config import get_settings
from ai_knowledge_platform.db.base import Base

# Alembic Config object.
config = context.config

# Configure the database URL if it was not supplied externally.
if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option(
        "sqlalchemy.url",
        get_settings().database_url,
    )

# Configure logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Importing models registers their tables in Base.metadata.
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

    supplied_connection = config.attributes.get("connection")

    if supplied_connection is not None:
        # Use the externally supplied connection.
        context.configure(
            connection=supplied_connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()

        return

    # Create a new engine for normal Alembic CLI execution.
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


# Select the migration execution mode.
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
