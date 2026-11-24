from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # Local LLM (llama.cpp server)
    local_llm_base_url: str = "http://localhost:8080/v1"
    local_llm_model: str = "qwen3-4b-instruct"
    local_llm_timeout_sec: int = 45

    # Cloud LLM
    cloud_llm_base_url: str = "https://openrouter.ai/api/v1"
    cloud_llm_api_key: str = Field(default="")
    cloud_llm_model: str = "google/gemini-2.5-flash"
    cloud_llm_timeout_sec: int = 30

    # Storage
    sqlite_db_path: Path = Path("data/research_assistant.db")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "research_papers"

    grobid_url: str = "http://localhost:8070"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"

@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings