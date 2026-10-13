import pytest
from ai_knowledge_platform.security.password import (
    hash_password,
    password_needs_rehash,
    verify_password,
)
from argon2 import PasswordHasher


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


@pytest.mark.parametrize(
    "malformed_hash",
    [
        "",
        "not-a-valid-hash",
        "$argon2id$malformed",
    ],
)
def test_verify_password_with_malformed_hash(
    malformed_hash: str,
) -> None:
    assert verify_password("SomePassword123!", malformed_hash) is False


def test_current_hash_does_not_need_rehash() -> None:
    password_hash = hash_password("StrongPassword123!")

    assert password_needs_rehash(password_hash) is False


def test_hash_with_outdated_parameters_needs_rehash() -> None:
    legacy_hasher = PasswordHasher(
        time_cost=1,
        memory_cost=1024,
        parallelism=1,
    )
    old_hash = legacy_hasher.hash("StrongPassword123!")

    assert password_needs_rehash(old_hash) is True
