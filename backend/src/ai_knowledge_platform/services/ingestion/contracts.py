from typing import Protocol, TypeVar

from ai_knowledge_platform.services.ingestion.context import StageContext
from ai_knowledge_platform.services.ingestion.results import StageResult

OutputT_co = TypeVar("OutputT_co", covariant=True)


class StageExecutor(Protocol[OutputT_co]):
    """Define the interface implemented by ingestion stage executors."""

    def execute(self, context: StageContext) -> StageResult[OutputT_co]:
        """Execute stage-specific processing and return its result."""
        ...
