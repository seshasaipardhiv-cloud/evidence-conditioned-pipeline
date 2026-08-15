import os
from backend.app.config import Settings

def test_config(monkeypatch):
    monkeypatch.setenv("APP_NAME", "test-app")
    monkeypatch.setenv("APP_VERSION", "1.0.0")
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    
    settings = Settings()
    assert settings.app_name == "test-app"
    assert settings.app_version == "1.0.0"
    assert settings.environment == "test"
    assert settings.database_url == "postgresql+psycopg://test:test@localhost:5432/test"
    assert settings.log_level == "DEBUG"
