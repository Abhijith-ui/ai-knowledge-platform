import pytest
from ai_knowledge_platform.core.config import Settings
from ai_knowledge_platform.core.lifespan import lifespan
from ai_knowledge_platform.security.jwt import TokenType
from ai_knowledge_platform.security.keys import KeyLoadingError
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def valid_key_paths(tmp_path):
    private_key_path = tmp_path / "private.pem"
    public_key_path = tmp_path / "public.pem"

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    private_key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    public_key_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    return private_key_path, public_key_path


def test_lifespan_initializes_and_releases_jwt_service(
    monkeypatch,
    valid_key_paths,
):
    private_key_path, public_key_path = valid_key_paths

    settings = Settings(
        _env_file=None,
        jwt_private_key_path=str(private_key_path),
        jwt_public_key_path=str(public_key_path),
    )

    monkeypatch.setattr(
        "ai_knowledge_platform.core.lifespan.get_settings",
        lambda: settings,
    )

    app = FastAPI(lifespan=lifespan)

    with TestClient(app):
        service = app.state.jwt_service
        token = service.create_access_token("test-user")
        payload = service.decode_token(token, TokenType.ACCESS)

        assert payload["sub"] == "test-user"

    assert not hasattr(app.state, "jwt_service")


def test_lifespan_fails_when_private_key_is_missing(
    monkeypatch,
    tmp_path,
):
    settings = Settings(
        _env_file=None,
        jwt_private_key_path=str(tmp_path / "missing-private.pem"),
        jwt_public_key_path=str(tmp_path / "missing-public.pem"),
    )

    monkeypatch.setattr(
        "ai_knowledge_platform.core.lifespan.get_settings",
        lambda: settings,
    )

    app = FastAPI(lifespan=lifespan)

    with pytest.raises(KeyLoadingError):
        with TestClient(app):
            pass
