from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ai_knowledge_platform.services.tokenization.base import Tokenizer
from ai_knowledge_platform.services.tokenization.types import TokenOffset


class HuggingFaceTokenizer(Tokenizer):
    """Tokenizer adapter backed by Hugging Face Transformers."""

    def __init__(
        self,
        model_name: str,
        max_input_tokens: int,
    ) -> None:
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive.")

        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        if not self._tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for offset-based chunking.")

        self._max_input_tokens = max_input_tokens

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        token_ids = self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )

        if not isinstance(token_ids, list) or not all(
            isinstance(token_id, int) for token_id in token_ids
        ):
            raise TypeError("Tokenizer encode() must return a list of integers.")

        return token_ids

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        decoded = self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

        if not isinstance(decoded, str):
            raise TypeError("Tokenizer decode() must return a string.")

        return decoded

    def count_tokens(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> int:
        return len(
            self.encode(
                text,
                add_special_tokens=add_special_tokens,
            )
        )

    @property
    def max_input_tokens(self) -> int:
        return self._max_input_tokens

    @property
    def special_tokens_count(self) -> int:
        return self._tokenizer.num_special_tokens_to_add(pair=False)

    def encode_with_offsets(self, text: str) -> list[TokenOffset]:
        """Return token IDs and their source character positions."""

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )

        token_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        return [
            TokenOffset(
                token_id=token_id,
                start_char=start,
                end_char=end,
            )
            for token_id, (start, end) in zip(
                token_ids,
                offsets,
                strict=True,
            )
            if end > start
        ]
