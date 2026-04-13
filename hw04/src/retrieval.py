from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest, SearchParams
from rank_bm25 import BM25Okapi

from .vectorize import get_qdrant_client


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def bm25_retrieve(
    docs_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    k: int = 10,
    tokenizer: Callable[[str], list[str]] | None = None,
    k1: float = 1.2,
    b: float = 0.75,
    epsilon: float = 0.25,
) -> pd.DataFrame:
    if k <= 0:
        return pd.DataFrame(columns=["query_id", "doc_id", "score"])

    tokenizer = tokenizer or simple_tokenize
    doc_ids = docs_df["doc_id"].astype(str).tolist()
    doc_texts = docs_df["text"].astype(str).tolist()
    tokenized_docs = [tokenizer(text) for text in doc_texts]

    bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b, epsilon=epsilon)
    rows = []
    for query_id, query_text in queries_df[["query_id", "text"]].itertuples(
        index=False, name=None
    ):
        scores = bm25.get_scores(tokenizer(str(query_text)))
        if len(scores) == 0:
            continue
        top_indices = np.argsort(scores)[-min(k, len(scores)) :][::-1]
        for idx in top_indices:
            rows.append(
                {
                    "query_id": str(query_id),
                    "doc_id": str(doc_ids[idx]),
                    "score": float(scores[idx]),
                }
            )
    return pd.DataFrame(rows, columns=["query_id", "doc_id", "score"])


def vector_retrieve(
    query_ids: list[str],
    query_embeddings: np.ndarray,
    collection_name: str,
    top_k: int = 10,
    client: QdrantClient | None = None,
    query_filter: object | None = None,
) -> pd.DataFrame:
    if top_k <= 0:
        return pd.DataFrame(columns=["query_id", "doc_id", "score"])
    if len(query_ids) != query_embeddings.shape[0]:
        raise ValueError("query_ids must match the number of query embeddings")

    client = client or get_qdrant_client()
    search_params = SearchParams(exact=True)
    requests = [
        QueryRequest(
            query=vector.tolist(),
            filter=query_filter,
            limit=top_k,
            with_payload=True,
            params=search_params,
        )
        for vector in query_embeddings
    ]
    responses = client.query_batch_points(
        collection_name=collection_name, requests=requests
    )
    rows = []
    for query_id, response in zip(query_ids, responses, strict=True):
        for point in response.points:
            payload = point.payload or {}
            doc_id = payload.get("item_id", point.id)
            rows.append(
                {
                    "query_id": str(query_id),
                    "doc_id": str(doc_id),
                    "score": float(point.score),
                }
            )
    return pd.DataFrame(rows, columns=["query_id", "doc_id", "score"])


def _min_max_normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def mix_runs_alpha(
    bm25_run: pd.DataFrame,
    vector_run: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    bm25 = bm25_run.copy()
    vector = vector_run.copy()

    bm25["bm25_norm"] = bm25.groupby("query_id")["score"].transform(_min_max_normalize)
    bm25 = bm25.rename(columns={"score": "bm25_score"})
    vector = vector.rename(columns={"score": "vector_score"})

    merged = pd.merge(bm25, vector, on=["query_id", "doc_id"], how="outer")
    merged["bm25_norm"] = merged["bm25_norm"].fillna(0.0)
    merged["vector_score"] = merged["vector_score"].fillna(0.0)

    merged["score"] = (
        alpha * merged["bm25_norm"]
        + (1.0 - alpha) * merged["vector_score"]
    )

    return merged[["query_id", "doc_id", "score"]]
