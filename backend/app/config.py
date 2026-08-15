from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    app_name: str = Field("evidence-conditioned-pipeline", alias="APP_NAME")
    app_version: str = Field("0.1.0", alias="APP_VERSION")
    environment: str = Field("development", alias="ENVIRONMENT")
    database_url: str = Field("postgresql+psycopg://postgres:postgres@localhost:5432/evidence_pipeline", alias="DATABASE_URL")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
