from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
import pytest
from ai_knowledge_platform.security.jwt import (
    JWTService,
    TokenType,
    TokenValidationError,
)
from cryptography.hazmat.primitives.asymmetric import rsa


@pytest.fixture
def key_pair():
    """Generate a temporary RSA key pair for testing."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    return private_key, private_key.public_key()


@pytest.fixture
def jwt_service(key_pair):
    """Create a JWT service with test keys and token lifetimes."""
    private_key, public_key = key_pair

    return JWTService(
        private_key=private_key,
        public_key=public_key,
        access_token_lifetime=timedelta(minutes=15),
        refresh_token_lifetime=timedelta(days=30),
    )


def test_create_and_decode_access_token(jwt_service):
    user_id = uuid4()

    token = jwt_service.create_access_token(user_id)
    payload = jwt_service.decode_token(token, TokenType.ACCESS)

    assert payload["sub"] == str(user_id)
    assert payload["type"] == "access"
    assert "iat" in payload
    assert "exp" in payload
    assert "jti" in payload


def test_create_and_decode_refresh_token(jwt_service):
    user_id = uuid4()

    token = jwt_service.create_refresh_token(user_id)
    payload = jwt_service.decode_token(token, TokenType.REFRESH)

    assert payload["sub"] == str(user_id)
    assert payload["type"] == "refresh"


def test_refresh_token_cannot_be_used_as_access_token(jwt_service):
    token = jwt_service.create_refresh_token(uuid4())

    with pytest.raises(TokenValidationError, match="Token type"):
        jwt_service.decode_token(token, TokenType.ACCESS)


def test_access_token_cannot_be_used_as_refresh_token(jwt_service):
    token = jwt_service.create_access_token(uuid4())

    with pytest.raises(TokenValidationError, match="Token type"):
        jwt_service.decode_token(token, TokenType.REFRESH)


def test_expired_token_is_rejected(jwt_service, key_pair):
    private_key, _ = key_pair
    now = datetime.now(timezone.utc)

    token = jwt.encode(
        {
            "sub": str(uuid4()),
            "type": "access",
            "iat": now - timedelta(minutes=2),
            "exp": now - timedelta(minutes=1),
            "jti": str(uuid4()),
        },
        private_key,
        algorithm="RS256",
    )

    with pytest.raises(TokenValidationError, match="Invalid token"):
        jwt_service.decode_token(token, TokenType.ACCESS)


def test_token_signed_by_another_key_is_rejected(jwt_service):
    other_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    now = datetime.now(timezone.utc)

    token = jwt.encode(
        {
            "sub": str(uuid4()),
            "type": "access",
            "iat": now,
            "exp": now + timedelta(minutes=15),
            "jti": str(uuid4()),
        },
        other_private_key,
        algorithm="RS256",
    )

    with pytest.raises(TokenValidationError, match="Invalid token"):
        jwt_service.decode_token(token, TokenType.ACCESS)


def test_token_missing_required_claim_is_rejected(jwt_service, key_pair):
    private_key, _ = key_pair

    token = jwt.encode(
        {"sub": str(uuid4()), "type": "access"},
        private_key,
        algorithm="RS256",
    )

    with pytest.raises(TokenValidationError, match="Invalid token"):
        jwt_service.decode_token(token, TokenType.ACCESS)


def test_malformed_token_is_rejected(jwt_service):
    with pytest.raises(TokenValidationError, match="Invalid token"):
        jwt_service.decode_token("not-a-valid-jwt", TokenType.ACCESS)


def test_empty_token_is_rejected(jwt_service):
    with pytest.raises(TokenValidationError, match="Invalid token"):
        jwt_service.decode_token("", TokenType.ACCESS)


def test_mismatched_keys_are_rejected(key_pair):
    private_key, _ = key_pair
    unrelated_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    with pytest.raises(ValueError, match="do not match"):
        JWTService(
            private_key=private_key,
            public_key=unrelated_private_key.public_key(),
            access_token_lifetime=timedelta(minutes=15),
            refresh_token_lifetime=timedelta(days=30),
        )


def test_non_positive_token_lifetime_is_rejected(key_pair):
    private_key, public_key = key_pair

    with pytest.raises(ValueError, match="must be positive"):
        JWTService(
            private_key=private_key,
            public_key=public_key,
            access_token_lifetime=timedelta(0),
            refresh_token_lifetime=timedelta(days=30),
        )


def test_weak_rsa_keys_are_rejected():
    weak_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=1024,
    )

    with pytest.raises(ValueError, match="at least 2048 bits"):
        JWTService(
            private_key=weak_private_key,
            public_key=weak_private_key.public_key(),
            access_token_lifetime=timedelta(minutes=15),
            refresh_token_lifetime=timedelta(days=30),
        )


def test_empty_subject_is_rejected(jwt_service):
    with pytest.raises(ValueError, match="subject must not be empty"):
        jwt_service.create_access_token("")


def test_whitespace_subject_is_rejected(jwt_service):
    with pytest.raises(ValueError, match="subject must not be empty"):
        jwt_service.create_access_token("   ")
