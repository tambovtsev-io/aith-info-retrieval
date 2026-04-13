import pandas as pd
from sentence_transformers import CrossEncoder

from .settings import settings
from .vectorize import resolve_device


def rerank_run(
    run_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    docs_df: pd.DataFrame,
    model_name: str | None = None,
    top_k: int | None = 50,
    batch_size: int | None = None,
    enabled: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    if not enabled:
        return run_df.copy()
    if "query_id" not in run_df.columns or "doc_id" not in run_df.columns:
        raise ValueError("run_df must include query_id and doc_id columns")
    if "query_id" not in queries_df.columns or "text" not in queries_df.columns:
        raise ValueError("queries_df must include query_id and text columns")
    if "doc_id" not in docs_df.columns or "text" not in docs_df.columns:
        raise ValueError("docs_df must include doc_id and text columns")

    model_name = model_name or settings.rerank_model
    batch_size = batch_size or settings.rerank_batch_size
    device = resolve_device(settings.embedding_device)
    model = CrossEncoder(model_name, device=device)

    query_texts = queries_df.set_index("query_id")["text"].astype(str).to_dict()
    doc_texts = docs_df.set_index("doc_id")["text"].astype(str).to_dict()

    rows = []
    for query_id, group in run_df.groupby("query_id"):
        query_text = query_texts.get(query_id)
        if query_text is None:
            continue
        group_sorted = group.sort_values("score", ascending=False)
        if top_k is None or top_k <= 0:
            rerank_slice = group_sorted
            remainder = group_sorted.iloc[0:0]
        else:
            rerank_slice = group_sorted.head(top_k)
            remainder = group_sorted.iloc[top_k:]

        pairs = []
        ids = []
        for _, row in rerank_slice.iterrows():
            doc_id = row["doc_id"]
            doc_text = doc_texts.get(doc_id)
            if doc_text is None:
                continue
            pairs.append((query_text, doc_text))
            ids.append(doc_id)

        if pairs:
            scores = model.predict(
                pairs, batch_size=batch_size, show_progress_bar=show_progress
            )
            scored = list(zip(ids, scores, strict=True))
            scored.sort(key=lambda item: item[1], reverse=True)
            for doc_id, score in scored:
                rows.append(
                    {"query_id": str(query_id), "doc_id": str(doc_id), "score": float(score)}
                )

        for _, row in remainder.iterrows():
            rows.append(
                {
                    "query_id": str(query_id),
                    "doc_id": str(row["doc_id"]),
                    "score": float(row["score"]),
                }
            )

    return pd.DataFrame(rows, columns=["query_id", "doc_id", "score"])
