from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, Mapping, TypeVar

OutputT_co = TypeVar("OutputT_co", covariant=True)


@dataclass(frozen=True, slots=True)
class StageResult(Generic[OutputT_co]):
    """Represent the output and metrics produced by an ingestion stage."""

    output: OutputT_co
    metrics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Prevent mutation of the top-level metrics mapping."""
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(dict(self.metrics)),
        )
