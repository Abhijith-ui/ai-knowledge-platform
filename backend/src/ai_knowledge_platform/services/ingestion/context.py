from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from uuid import UUID


@dataclass(frozen=True, slots=True)
class StageContext:
    """Provide immutable execution context to an ingestion stage."""

    ingestion_run_id: UUID
    document_id: UUID
    lease_token: UUID
    config_snapshot: Mapping[str, object]

    def __post_init__(self) -> None:
        """Prevent mutation of the top-level configuration mapping."""
        object.__setattr__(
            self,
            "config_snapshot",
            MappingProxyType(dict(self.config_snapshot)),
        )
