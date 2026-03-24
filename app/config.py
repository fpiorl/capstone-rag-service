from functools import lru_cache
from typing import ClassVar


from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openrouter_api_key: str
    openrouter_embedding_model: str = "openai/text-embedding-3-small"
    openrouter_chat_model: str = "openai/gpt-4o-mini"
    chroma_persist_dir: str = "./chroma_data"
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 5
    host: str = "0.0.0.0"
    port: int = 8081

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
