from pathlib import Path

import pytest
from ai_knowledge_platform.core.config import Settings
from ai_knowledge_platform.db.session import get_db
from ai_knowledge_platform.main import app
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


@pytest.fixture(scope="session")
def jwt_key_paths(tmp_path_factory):
    """Generate temporary RSA keys for API tests."""
    key_directory = tmp_path_factory.mktemp("jwt-keys")

    private_key_path = key_directory / "private.pem"
    public_key_path = key_directory / "public.pem"

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


@pytest.fixture
def client(
    db: Session,
    jwt_key_paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    private_key_path, public_key_path = jwt_key_paths

    settings = Settings(
        _env_file=None,
        jwt_private_key_path=str(private_key_path),
        jwt_public_key_path=str(public_key_path),
        document_storage_dir=tmp_path / "documents",
        max_document_upload_bytes=1024 * 1024,
        max_document_request_bytes=1024,
    )

    monkeypatch.setattr(
        "ai_knowledge_platform.core.lifespan.get_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        "ai_knowledge_platform.services.document.get_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        "ai_knowledge_platform.core.config.get_settings",
        lambda: settings,
    )

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
