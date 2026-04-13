import re
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from .settings import settings
from .models import EmbeddingConfig


def sanitize_model_name(model_name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", model_name.strip())
    return cleaned.strip("_").lower()


def collection_name(prefix: str, model_name: str) -> str:
    return f"{prefix}_{sanitize_model_name(model_name)}"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_qdrant_client(qdrant_path: Path | None = None) -> QdrantClient:
    qdrant_path = Path(qdrant_path) if qdrant_path is not None else settings.qdrant_path
    qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(qdrant_path))


def _load_model(model_name: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


def embed_texts(
    texts: Iterable[str],
    config: EmbeddingConfig,
) -> np.ndarray:
    device = resolve_device(config.device)
    model = _load_model(config.model_name, device=device)
    embeddings = model.encode(
        list(texts),
        batch_size=config.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=config.normalize_embeddings,
        show_progress_bar=config.show_progress,
    )
    return np.asarray(embeddings)


def _coerce_point_id(item_id: str | int) -> str | int:
    if isinstance(item_id, int):
        return item_id
    item_str = str(item_id)
    if item_str.isdigit():
        return int(item_str)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, item_str))


def _sanitize_payload_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_payloads(
    df: pd.DataFrame,
    id_column: str,
    text_column: str,
    extra_columns: list[str] | None = None,
) -> list[dict[str, object]]:
    if extra_columns is None:
        extra_columns = [
            column for column in df.columns if column not in {id_column, text_column}
        ]
    payloads: list[dict[str, object]] = []
    for _, row in df.iterrows():
        payload = {"item_id": str(row[id_column]), "text": str(row[text_column])}
        for column in extra_columns:
            payload[column] = _sanitize_payload_value(row[column])
        payloads.append(payload)
    return payloads


@retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3), reraise=True)
def ensure_collection(
    client: QdrantClient,
    name: str,
    vector_size: int,
) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE,
        ),
    )


@retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3), reraise=True)
def upsert_embeddings(
    client: QdrantClient,
    collection_name: str,
    item_ids: Iterable[str],
    texts: Iterable[str],
    embeddings: np.ndarray,
    payloads: list[dict[str, object]] | None = None,
) -> None:
    ids_list = list(item_ids)
    texts_list = list(texts)
    if len(ids_list) != len(texts_list) or len(ids_list) != embeddings.shape[0]:
        raise ValueError("Item ids, texts, and embeddings must have matching lengths")
    if payloads is not None and len(payloads) != len(ids_list):
        raise ValueError("payloads length must match item_ids length")
    points: list[qdrant_models.PointStruct] = []
    for idx, vector in enumerate(embeddings):
        point_id = _coerce_point_id(ids_list[idx])
        extra_payload = payloads[idx] if payloads is not None else {}
        payload = {
            "item_id": str(ids_list[idx]),
            "text": texts_list[idx],
            **extra_payload,
        }
        points.append(
            qdrant_models.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload,
            )
        )
    client.upsert(collection_name=collection_name, points=points, wait=True)


def vectorize_and_store(
    docs_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    config: EmbeddingConfig | None = None,
    qdrant_path: Path | None = None,
    doc_payload_columns: list[str] | None = None,
    query_payload_columns: list[str] | None = None,
    docs_collection_name: str | None = None,
    queries_collection_name: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray, str, str]:
    if "doc_id" not in docs_df.columns or "text" not in docs_df.columns:
        raise ValueError("docs_df must include doc_id and text columns")
    if "query_id" not in queries_df.columns or "text" not in queries_df.columns:
        raise ValueError("queries_df must include query_id and text columns")

    config = config or EmbeddingConfig(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
        normalize_embeddings=settings.embedding_normalize,
    )

    doc_ids = docs_df["doc_id"].astype(str).tolist()
    doc_texts = docs_df["text"].astype(str).tolist()
    query_ids = queries_df["query_id"].astype(str).tolist()
    query_texts = queries_df["text"].astype(str).tolist()

    if not doc_ids:
        raise ValueError("docs_df must contain at least one document")
    if not query_ids:
        raise ValueError("queries_df must contain at least one query")

    client = get_qdrant_client(qdrant_path=qdrant_path)
    if docs_collection_name is not None:
        docs_collection = docs_collection_name
        if not client.collection_exists(docs_collection):
            raise ValueError(f"docs collection '{docs_collection}' does not exist")
        doc_embeddings = None
    else:
        doc_embeddings = embed_texts(doc_texts, config)
        docs_collection = collection_name(
            f"{config.collection_prefix}_docs", config.model_name
        )
        ensure_collection(client, docs_collection, vector_size=doc_embeddings.shape[1])
        doc_payloads = _build_payloads(
            docs_df,
            id_column="doc_id",
            text_column="text",
            extra_columns=doc_payload_columns,
        )
        upsert_embeddings(
            client,
            docs_collection,
            doc_ids,
            doc_texts,
            doc_embeddings,
            payloads=doc_payloads,
        )

    query_embeddings = embed_texts(query_texts, config)
    if queries_collection_name is not None:
        queries_collection = queries_collection_name
        if not client.collection_exists(queries_collection):
            raise ValueError(f"queries collection '{queries_collection}' does not exist")
    else:
        queries_collection = collection_name(
            f"{config.collection_prefix}_queries", config.model_name
        )
        query_payloads = _build_payloads(
            queries_df,
            id_column="query_id",
            text_column="text",
            extra_columns=query_payload_columns,
        )
        ensure_collection(
            client, queries_collection, vector_size=query_embeddings.shape[1]
        )
        upsert_embeddings(
            client,
            queries_collection,
            query_ids,
            query_texts,
            query_embeddings,
            payloads=query_payloads,
        )

    return doc_embeddings, query_embeddings, docs_collection, queries_collection
