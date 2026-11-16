import pytest
from ai_knowledge_platform.services.tokenization.huggingface import (
    HuggingFaceTokenizer,
)


@pytest.fixture(scope="module")
def tokenizer() -> HuggingFaceTokenizer:
    return HuggingFaceTokenizer(
        model_name="intfloat/multilingual-e5-base",
        max_input_tokens=512,
    )


def test_encode_returns_token_ids(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    token_ids = tokenizer.encode("Machine learning is interesting.")

    assert isinstance(token_ids, list)
    assert len(token_ids) > 0
    assert all(isinstance(token_id, int) for token_id in token_ids)


def test_decode_returns_text(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    text = "Machine learning is interesting."

    token_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(token_ids)

    assert decoded_text == text


def test_count_tokens_matches_encoding(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    text = "Understanding tokenization is important."

    assert tokenizer.count_tokens(text) == len(tokenizer.encode(text))


def test_special_tokens_count(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    assert tokenizer.special_tokens_count > 0


def test_model_input_limit(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    assert tokenizer.max_input_tokens == 512


def test_invalid_input_limit() -> None:
    with pytest.raises(ValueError):
        HuggingFaceTokenizer(
            model_name="intfloat/multilingual-e5-base",
            max_input_tokens=0,
        )


def test_encode_with_offsets(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    text = "Machine learning is interesting."

    tokens = tokenizer.encode_with_offsets(text)

    assert len(tokens) == tokenizer.count_tokens(text)

    for token in tokens:
        assert token.start_char < token.end_char
        assert token.start_char >= 0
        assert token.end_char <= len(text)


def test_token_offsets_preserve_unicode(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    text = "Machine learning with café and नमस्ते."

    tokens = tokenizer.encode_with_offsets(text)

    assert len(tokens) > 0

    for token in tokens:
        assert text[token.start_char : token.end_char]


def test_token_offsets_preserve_whitespace(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    text = "Machine   learning\n\nis interesting."

    tokens = tokenizer.encode_with_offsets(text)

    assert len(tokens) == tokenizer.count_tokens(text)

    for token in tokens:
        assert 0 <= token.start_char < token.end_char <= len(text)
