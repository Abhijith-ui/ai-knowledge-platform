from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenOffset:
    """A token and its character span in the original source text."""

    token_id: int
    start_char: int
    end_char: int
