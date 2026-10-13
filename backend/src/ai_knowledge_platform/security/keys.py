from pathlib import Path

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateKey,
    RSAPublicKey,
)


class KeyLoadingError(RuntimeError):
    """Raised when configured RSA keys cannot be loaded or validated."""


def load_rsa_keys(
    private_key_path: str | Path,
    public_key_path: str | Path,
) -> tuple[RSAPrivateKey, RSAPublicKey]:
    """Load and validate an RSA private/public key pair."""
    private_path = Path(private_key_path)
    public_path = Path(public_key_path)

    try:
        private_data = private_path.read_bytes()
    except OSError as exc:
        raise KeyLoadingError(
            f"Cannot read RSA private key file: {private_path}"
        ) from exc

    try:
        public_data = public_path.read_bytes()
    except OSError as exc:
        raise KeyLoadingError(
            f"Cannot read RSA public key file: {public_path}"
        ) from exc

    try:
        private_key = serialization.load_pem_private_key(
            private_data,
            password=None,
        )
    except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
        raise KeyLoadingError(
            "Configured private key is not a valid unencrypted PEM key."
        ) from exc

    try:
        public_key = serialization.load_pem_public_key(public_data)
    except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
        raise KeyLoadingError("Configured public key is not a valid PEM key.") from exc

    if not isinstance(private_key, RSAPrivateKey):
        raise KeyLoadingError("Configured private key must be an RSA key.")

    if not isinstance(public_key, RSAPublicKey):
        raise KeyLoadingError("Configured public key must be an RSA key.")

    if private_key.key_size < 2048 or public_key.key_size < 2048:
        raise KeyLoadingError("RSA keys must be at least 2048 bits.")

    if private_key.public_key().public_numbers() != public_key.public_numbers():
        raise KeyLoadingError("Configured RSA private and public keys do not match.")

    return private_key, public_key
