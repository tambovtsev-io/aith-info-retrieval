import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    qdrant_path: Path = data_dir / "qdrant_mirage01"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 1024
    embedding_device: str = "cuda"
    embedding_normalize: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_batch_size: int = 1024
    hf_cache_dir: Path | None = Path.home() / ".cache" / "huggingface"
    hf_local_files_only: bool = False
    log_level: int = logging.INFO

    model_config = SettingsConfigDict(
        env_file=project_root / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
