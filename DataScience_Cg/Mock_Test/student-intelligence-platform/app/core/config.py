from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Student Intelligence Platform"
    app_env: str = "development"
    database_url: str = "sqlite:///./student_platform.db"
    log_level: str = "INFO"
    model_path: str = "artifacts/student_model.joblib"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
