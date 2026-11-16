from abc import ABC, abstractmethod

from ai_knowledge_platform.services.tokenization.types import TokenOffset


class Tokenizer(ABC):
    @abstractmethod
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:

        raise NotImplementedError

    @abstractmethod
    def count_tokens(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def max_input_tokens(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def special_tokens_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def encode_with_offsets(self, text: str) -> list[TokenOffset]:
        """Encode text and return token-level source character offsets."""
        raise NotImplementedError
