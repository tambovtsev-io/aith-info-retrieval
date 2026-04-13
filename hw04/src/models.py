from collections.abc import Sequence

from pydantic import BaseModel, Field


class Doc(BaseModel):
    doc_id: str
    text: str
    dataset: str


class Query(BaseModel):
    query_id: str
    text: str


class Qrel(BaseModel):
    query_id: str
    doc_id: str
    relevance: int = Field(..., description="Relevance label (score).")


class RunRow(BaseModel):
    query_id: str
    doc_id: str
    score: float


class EmbeddingConfig(BaseModel):
    collection_prefix: str = ""
    model_name: str
    batch_size: int = 1024
    device: str = "auto"
    normalize_embeddings: bool = True
    show_progress: bool = True


class EvaluationConfig(BaseModel):
    metrics: Sequence[str] = ("P@1", "P@10", "P@20", "MAP@20", "nDCG@20")
