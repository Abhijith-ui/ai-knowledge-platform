from pathlib import Path

import pytest
from ai_knowledge_platform.security.keys import (
    KeyLoadingError,
    load_rsa_keys,
)
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def write_key_pair(
    directory: Path,
    private_key: rsa.RSAPrivateKey | None = None,
) -> tuple[Path, Path]:
    if private_key is None:
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

    private_path = directory / "private.pem"
    public_path = directory / "public.pem"

    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    return private_path, public_path


def test_load_valid_rsa_key_pair(tmp_path: Path) -> None:
    private_path, public_path = write_key_pair(tmp_path)

    private_key, public_key = load_rsa_keys(private_path, public_path)

    assert private_key.key_size == 2048
    assert public_key.key_size == 2048
    assert private_key.public_key().public_numbers() == public_key.public_numbers()


def test_missing_private_key_raises_error(tmp_path: Path) -> None:
    _, public_path = write_key_pair(tmp_path)

    with pytest.raises(KeyLoadingError, match="private key file"):
        load_rsa_keys(tmp_path / "missing.pem", public_path)


def test_missing_public_key_raises_error(tmp_path: Path) -> None:
    private_path, _ = write_key_pair(tmp_path)

    with pytest.raises(KeyLoadingError, match="public key file"):
        load_rsa_keys(private_path, tmp_path / "missing.pem")


def test_invalid_private_key_raises_error(tmp_path: Path) -> None:
    private_path, public_path = write_key_pair(tmp_path)
    private_path.write_text("not a valid PEM key", encoding="utf-8")

    with pytest.raises(KeyLoadingError, match="private key"):
        load_rsa_keys(private_path, public_path)


def test_invalid_public_key_raises_error(tmp_path: Path) -> None:
    private_path, public_path = write_key_pair(tmp_path)
    public_path.write_text("not a valid PEM key", encoding="utf-8")

    with pytest.raises(KeyLoadingError, match="public key"):
        load_rsa_keys(private_path, public_path)


def test_mismatched_key_pair_raises_error(tmp_path: Path) -> None:
    private_path, public_path = write_key_pair(tmp_path)
    another_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    public_path.write_bytes(
        another_private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    with pytest.raises(KeyLoadingError, match="do not match"):
        load_rsa_keys(private_path, public_path)
