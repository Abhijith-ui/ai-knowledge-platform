from ai_knowledge_platform.security.password import (
    hash_password,
    verify_password,
)


def test_hash_password() -> None:
    password = "StrongPassword123!"

    password_hash = hash_password(password)

    assert password_hash != password
    assert password_hash.startswith("$argon2id$")


def test_verify_password_with_correct_password() -> None:
    password = "StrongPassword123!"

    password_hash = hash_password(password)

    assert verify_password(password, password_hash) is True


def test_verify_password_with_wrong_password() -> None:
    password_hash = hash_password("StrongPassword123!")

    assert verify_password("WrongPassword123!", password_hash) is False


def test_same_password_produces_different_hashes() -> None:
    password = "StrongPassword123!"

    first_hash = hash_password(password)
    second_hash = hash_password(password)

    assert first_hash != second_hash
    assert verify_password(password, first_hash) is True
    assert verify_password(password, second_hash) is True
