from typing import Iterable

import ir_measures as irm
import numpy as np
import pandas as pd

from .settings import settings
from .evaluate import DEFAULT_METRICS, evaluate_run
from .models import EmbeddingConfig
from .rerank import rerank_run
from .retrieval import bm25_retrieve, mix_runs_alpha, vector_retrieve
from .vectorize import vectorize_and_store


def tune_alpha(
    bm25_run: pd.DataFrame,
    vector_run: pd.DataFrame,
    qrels_df: pd.DataFrame,
    alphas: Iterable[float] | None = None,
    metric: irm.Measure | str = irm.nDCG @ 20,
) -> tuple[float, pd.DataFrame]:
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 21)

    metric_obj = irm.parse_measure(metric) if isinstance(metric, str) else metric
    metric_key = str(metric_obj)

    best_alpha = 0.0
    best_score = float("-inf")
    rows = []
    for alpha in alphas:
        mixed_run = mix_runs_alpha(bm25_run, vector_run, alpha=float(alpha))
        metrics, _ = evaluate_run(mixed_run, qrels_df, metrics=[metric_obj])
        score = metrics.get(metric_key, float("-inf"))
        rows.append({"alpha": float(alpha), metric_key: score})
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)

    return best_alpha, pd.DataFrame(rows)


def _query_id_set(queries_df: pd.DataFrame | None) -> set[str] | None:
    if queries_df is None:
        return None
    return set(queries_df["query_id"].astype(str).tolist())


def _with_split(queries_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    queries_with_split = queries_df.copy()
    queries_with_split["split"] = split_name
    return queries_with_split


def _filter_run_by_query_ids(
    run_df: pd.DataFrame,
    query_ids: set[str] | None,
) -> pd.DataFrame:
    if query_ids is None:
        return run_df
    return run_df[run_df["query_id"].astype(str).isin(query_ids)].copy()


def _evaluate_split_run(
    split_name: str,
    run_name: str,
    run_df: pd.DataFrame,
    qrels_df: pd.DataFrame | None,
    query_ids: set[str] | None,
    metrics: Iterable[irm.Measure | str],
) -> dict[str, float | str] | None:
    if qrels_df is None or query_ids is None:
        return None
    split_run = _filter_run_by_query_ids(run_df, query_ids)
    split_metrics, _ = evaluate_run(
        split_run, qrels_df, metrics=metrics, return_intermediate=False
    )
    return {"split": split_name, "run": run_name, **split_metrics}


def _evaluate_metrics_over_splits(
    *,
    reranked_run: pd.DataFrame,
    bm25_run: pd.DataFrame,
    vector_run: pd.DataFrame,
    qrels_df: pd.DataFrame,
    validation_qrels_df: pd.DataFrame | None,
    test_qrels_df: pd.DataFrame | None,
    train_query_ids: set[str] | None,
    validation_query_ids: set[str] | None,
    test_query_ids: set[str] | None,
    metrics: Iterable[irm.Measure | str],
) -> tuple[dict[str, float], dict[str, dict[str, float]], pd.DataFrame]:
    train_run = _filter_run_by_query_ids(reranked_run, train_query_ids)
    metrics_result, _ = evaluate_run(
        train_run, qrels_df, metrics=metrics, return_intermediate=False
    )
    split_metrics: dict[str, dict[str, float]] = {"train": metrics_result}
    if validation_qrels_df is not None and validation_query_ids is not None:
        validation_run = _filter_run_by_query_ids(reranked_run, validation_query_ids)
        validation_metrics, _ = evaluate_run(
            validation_run,
            validation_qrels_df,
            metrics=metrics,
            return_intermediate=False,
        )
        split_metrics["validation"] = validation_metrics
    if test_qrels_df is not None and test_query_ids is not None:
        test_run = _filter_run_by_query_ids(reranked_run, test_query_ids)
        test_metrics, _ = evaluate_run(
            test_run,
            test_qrels_df,
            metrics=metrics,
            return_intermediate=False,
        )
        split_metrics["test"] = test_metrics

    split_run_rows: list[dict[str, float | str]] = []
    for split_name, split_qrels, split_query_ids in (
        ("train", qrels_df, train_query_ids),
        ("validation", validation_qrels_df, validation_query_ids),
        ("test", test_qrels_df, test_query_ids),
    ):
        for run_name, run_df in (
            ("bm25", bm25_run),
            ("vector", vector_run),
            ("reranked", reranked_run),
        ):
            row = _evaluate_split_run(
                split_name=split_name,
                run_name=run_name,
                run_df=run_df,
                qrels_df=split_qrels,
                query_ids=split_query_ids,
                metrics=metrics,
            )
            if row is not None:
                split_run_rows.append(row)
    split_run_metrics = pd.DataFrame(split_run_rows)
    return metrics_result, split_metrics, split_run_metrics


def run_pipeline(
    docs_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    validation_queries_df: pd.DataFrame | None = None,
    validation_qrels_df: pd.DataFrame | None = None,
    test_queries_df: pd.DataFrame | None = None,
    test_qrels_df: pd.DataFrame | None = None,
    embedding_model: str | None = None,
    bm25_k: int = 100,
    vector_k: int = 100,
    alpha: float | None = None,
    tune_alpha_on_train: bool = False,
    alpha_candidates: Iterable[float] | None = None,
    rerank_enabled: bool = True,
    rerank_top_k: int | None = 50,
    rerank_model: str | None = None,
    metrics: Iterable[irm.Measure | str] | None = None,
    collection_prefix: str = "",
    docs_collection_name: str | None = None,
    queries_collection_name: str | None = None,
) -> dict[str, object]:
    embedding_config = EmbeddingConfig(
        model_name=embedding_model,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
        normalize_embeddings=settings.embedding_normalize,
        collection_prefix=collection_prefix,
    )

    queries_with_split = [_with_split(queries_df, "train")]
    if validation_queries_df is not None:
        queries_with_split.append(_with_split(validation_queries_df, "validation"))
    if test_queries_df is not None:
        queries_with_split.append(_with_split(test_queries_df, "test"))
    all_queries_df = pd.concat(queries_with_split, ignore_index=True)

    _, query_embeddings, docs_collection, queries_collection = vectorize_and_store(
        docs_df,
        all_queries_df,
        config=embedding_config,
        query_payload_columns=["split"],
        docs_collection_name=docs_collection_name,
        queries_collection_name=queries_collection_name,
    )

    bm25_run = bm25_retrieve(docs_df, all_queries_df, k=bm25_k)
    query_ids = all_queries_df["query_id"].astype(str).tolist()
    vector_run = vector_retrieve(
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        collection_name=docs_collection,
        top_k=vector_k,
    )

    train_query_ids = _query_id_set(queries_df)
    validation_query_ids = _query_id_set(validation_queries_df)
    test_query_ids = _query_id_set(test_queries_df)

    alpha_results = None
    if tune_alpha_on_train:
        tuning_bm25_run = _filter_run_by_query_ids(bm25_run, train_query_ids)
        tuning_vector_run = _filter_run_by_query_ids(vector_run, train_query_ids)
        alpha, alpha_results = tune_alpha(
            bm25_run=tuning_bm25_run,
            vector_run=tuning_vector_run,
            qrels_df=qrels_df,
            alphas=alpha_candidates,
            metric=irm.nDCG @ 20,
        )

    mixed_run = (
        mix_runs_alpha(bm25_run, vector_run, alpha=alpha)
        if alpha is not None
        else vector_run
    )

    reranked_run = rerank_run(
        mixed_run,
        all_queries_df,
        docs_df,
        model_name=rerank_model or settings.rerank_model,
        top_k=rerank_top_k,
        enabled=rerank_enabled,
    )

    metrics_to_use = metrics or DEFAULT_METRICS
    metrics_result, split_metrics, split_run_metrics = _evaluate_metrics_over_splits(
        reranked_run=reranked_run,
        bm25_run=bm25_run,
        vector_run=vector_run,
        qrels_df=qrels_df,
        validation_qrels_df=validation_qrels_df,
        test_qrels_df=test_qrels_df,
        train_query_ids=train_query_ids,
        validation_query_ids=validation_query_ids,
        test_query_ids=test_query_ids,
        metrics=metrics_to_use,
    )

    return {
        "bm25_run": bm25_run,
        "vector_run": vector_run,
        "mixed_run": mixed_run,
        "reranked_run": reranked_run,
        "metrics": metrics_result,
        "split_metrics": split_metrics,
        "split_run_metrics": split_run_metrics,
        "alpha": alpha,
        "alpha_results": alpha_results,
        "collections": {
            "docs": docs_collection,
            "queries": queries_collection,
        },
    }
