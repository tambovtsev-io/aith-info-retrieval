from __future__ import annotations

import re
import typing as tp
from datetime import datetime
from pathlib import Path

import ir_measures as irm
import numpy as np
import nltk
import pandas as pd
from catboost import CatBoostRanker, Pool
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import retrieval

try:
    from pydantic import BaseModel, ConfigDict

    class _ArbitraryTypesBaseModel(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

except ImportError:  # pragma: no cover - fallback for older pydantic
    from pydantic import BaseModel

    class _ArbitraryTypesBaseModel(BaseModel):
        class Config:
            arbitrary_types_allowed = True


DEFAULT_TARGET_K = 20
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
DEFAULT_CATBOOST_SUBDIR = Path("catboost_models") / "wikir"

FeatureFn = tp.Callable[[list[str], list[str]], float]


class LtrEvaluation(BaseModel):
    measure: str
    linear_ndcg: float
    catboost_ndcg: float
    train_size: int
    eval_size: int


class LtrModels(_ArbitraryTypesBaseModel):
    feature_columns: list[str]
    linear_model: LinearRegression
    linear_imputer: SimpleImputer
    linear_scaler: StandardScaler | None
    catboost_model: CatBoostRanker


class LtrTrainingResult(_ArbitraryTypesBaseModel):
    models: LtrModels
    evaluation: LtrEvaluation


def _validate_dataframe_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    df_name: str,
) -> None:
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{df_name} is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )


def simple_tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


class Bm25SparseVectorizer:
    def __init__(
        self,
        tokenizer: tp.Callable[[str], list[str]] | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.tokenizer = tokenizer or simple_tokenize
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(
            tokenizer=self.tokenizer,
            lowercase=False,
            token_pattern=None,
        )

    def fit(self, documents: tp.Sequence[str]) -> "Bm25SparseVectorizer":
        term_matrix = self.vectorizer.fit_transform(documents)
        self._fit_stats(term_matrix)
        return self

    def _fit_stats(self, term_matrix: sparse.spmatrix) -> None:
        doc_len = term_matrix.sum(axis=1)
        self.doc_len_ = np.asarray(doc_len).ravel()
        self.avgdl_ = float(self.doc_len_.mean()) if self.doc_len_.size > 0 else 0.0
        df = np.asarray((term_matrix > 0).sum(axis=0)).ravel()
        n_docs = term_matrix.shape[0]
        self.idf_ = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def _bm25_from_tf(
        self, tf_matrix: sparse.spmatrix, doc_len: np.ndarray
    ) -> sparse.csr_matrix:
        if tf_matrix.shape[0] == 0:
            return sparse.csr_matrix(tf_matrix)

        avgdl = self.avgdl_ if self.avgdl_ > 0 else 1.0
        norm = self.k1 * (1 - self.b + self.b * (doc_len / avgdl))

        tf = tf_matrix.tocoo()
        data = tf.data
        norm_rows = norm[tf.row]
        data = data * (self.k1 + 1) / (data + norm_rows)
        data = data * self.idf_[tf.col]
        return sparse.csr_matrix((data, (tf.row, tf.col)), shape=tf_matrix.shape)

    def transform_docs(self, documents: tp.Sequence[str]) -> sparse.csr_matrix:
        tf_matrix = self.vectorizer.transform(documents)
        doc_len = np.asarray(tf_matrix.sum(axis=1)).ravel()
        return self._bm25_from_tf(tf_matrix, doc_len)

    def transform_queries(self, queries: tp.Sequence[str]) -> sparse.csr_matrix:
        return self.vectorizer.transform(queries)


def feature_query_length(query_tokens: list[str], doc_tokens: list[str]) -> float:
    return float(len(query_tokens))


def feature_doc_length(query_tokens: list[str], doc_tokens: list[str]) -> float:
    return float(len(doc_tokens))


def feature_min_term_distance(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not query_tokens or not doc_tokens:
        return float(len(doc_tokens))

    unique_terms = list(dict.fromkeys(query_tokens))
    positions: dict[str, list[int]] = {term: [] for term in unique_terms}
    for idx, token in enumerate(doc_tokens):
        if token in positions:
            positions[token].append(idx)

    matched_terms = [term for term in unique_terms if positions[term]]
    if len(matched_terms) < 2:
        return float(len(doc_tokens))

    min_distance = float(len(doc_tokens))
    for i, term_a in enumerate(matched_terms):
        for term_b in matched_terms[i + 1 :]:
            for pos_a in positions[term_a]:
                for pos_b in positions[term_b]:
                    distance = abs(pos_a - pos_b)
                    if distance < min_distance:
                        min_distance = distance
    return float(min_distance)


def feature_n_intersecting_terms(
    query_tokens: list[str],
    doc_tokens: list[str],
) -> float:
    return float(len(set(query_tokens) & set(doc_tokens)))


def feature_frac_intersecting_terms(
    query_tokens: list[str],
    doc_tokens: list[str],
) -> float:
    return (
        feature_n_intersecting_terms(query_tokens, doc_tokens)
        / len(query_tokens)
    )


def feature_n_hits(
    query_tokens: list[str],
    doc_tokens: list[str],
) -> float:
    n_hits = 0
    for token in query_tokens:
        if token in doc_tokens:
            n_hits += 1
    return float(n_hits)


def feature_frac_hits(
    query_tokens: list[str],
    doc_tokens: list[str],
) -> float:
    n_hits = feature_n_hits(query_tokens, doc_tokens)
    return n_hits / len(doc_tokens)


def feature_bigram_overlap(
    query_tokens: list[str],
    doc_tokens: list[str],
) -> float:
    query_bigrams = set(nltk.bigrams(query_tokens))
    doc_bigrams = set(nltk.bigrams(doc_tokens))
    return float(len(query_bigrams & doc_bigrams))


def feature_bigram_overlap_frac(
    query_tokens: list[str],
    doc_tokens: list[str],
) -> float:
    query_bigrams = set(nltk.bigrams(query_tokens))
    if not query_bigrams:
        return 0.0
    doc_bigrams = set(nltk.bigrams(doc_tokens))
    return float(len(query_bigrams & doc_bigrams)) / len(doc_bigrams)


def feature_bigram_hits(
    query_tokens: list[str],
    doc_tokens: list[str],
) -> float:
    query_bigrams = list(nltk.bigrams(query_tokens))
    doc_bigrams = list(nltk.bigrams(doc_tokens))
    n_hits = 0
    for bigram in query_bigrams:
        if bigram in doc_bigrams:
            n_hits += 1
    return float(n_hits)


def feature_bigram_hits_frac(
    query_tokens: list[str],
    doc_tokens: list[str],
) -> float:
    doc_bigrams = list(nltk.bigrams(doc_tokens))
    if not doc_bigrams:
        return 0.0
    n_hits = feature_bigram_hits(query_tokens, doc_tokens)
    return n_hits / len(doc_bigrams)


DEFAULT_FEATURE_FUNCTIONS: list[FeatureFn] = [
    feature_query_length,
    feature_doc_length,
    feature_min_term_distance,
    feature_n_intersecting_terms,
    feature_frac_intersecting_terms,
    feature_n_hits,
    feature_frac_hits,
    feature_bigram_overlap,
    feature_bigram_hits,
    feature_bigram_overlap_frac,
    feature_bigram_hits_frac,
]


def infer_bm25_columns(
    df: pd.DataFrame,
    prefix: str = "bm25_",
) -> list[str]:
    return [col for col in df.columns if col.lower().startswith(prefix)]


def add_ltr_features(
    df: pd.DataFrame,
    query_col: str = "query_text",
    doc_col: str = "doc_text",
    tokenizer: tp.Callable[[str], list[str]] | None = None,
    feature_fns: list[FeatureFn] | None = None,
    feature_prefix: str = "f",
) -> pd.DataFrame:
    _validate_dataframe_columns(df, {query_col, doc_col}, "df")
    tokenizer = tokenizer or simple_tokenize
    feature_fns = feature_fns or DEFAULT_FEATURE_FUNCTIONS

    query_tokens = df[query_col].map(lambda text: tokenizer(str(text)))
    doc_tokens = df[doc_col].map(lambda text: tokenizer(str(text)))

    features: dict[str, list[float]] = {}
    for idx, feature_fn in enumerate(feature_fns, start=1):
        features[f"{feature_prefix}{idx}"] = [
            feature_fn(q_tokens, d_tokens)
            for q_tokens, d_tokens in zip(query_tokens, doc_tokens)
        ]

    return df.assign(**features)


def build_bm25_sparse_features(
    candidates_df: pd.DataFrame,
    vectorizer: Bm25SparseVectorizer,
    query_id_col: str = "query_id",
    query_text_col: str = "query_text",
    doc_id_col: str = "doc_id",
    doc_text_col: str = "doc_text",
) -> sparse.csr_matrix:
    _validate_dataframe_columns(
        candidates_df,
        {query_id_col, query_text_col, doc_id_col, doc_text_col},
        "candidates_df",
    )

    doc_df = candidates_df[[doc_id_col, doc_text_col]].drop_duplicates(doc_id_col)
    query_df = candidates_df[[query_id_col, query_text_col]].drop_duplicates(
        query_id_col
    )

    doc_ids = doc_df[doc_id_col].astype(str).tolist()
    query_ids = query_df[query_id_col].astype(str).tolist()

    doc_matrix = vectorizer.transform_docs(doc_df[doc_text_col].astype(str).tolist())
    query_matrix = vectorizer.transform_queries(
        query_df[query_text_col].astype(str).tolist()
    )

    doc_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    query_index = {query_id: idx for idx, query_id in enumerate(query_ids)}

    rows: list[sparse.spmatrix] = []
    for query_id, doc_id in candidates_df[[query_id_col, doc_id_col]].itertuples(
        index=False, name=None
    ):
        q_idx = query_index[str(query_id)]
        d_idx = doc_index[str(doc_id)]
        rows.append(doc_matrix[d_idx].multiply(query_matrix[q_idx]))

    return sparse.vstack(rows, format="csr")


def build_ltr_feature_matrix(
    df: pd.DataFrame,
    bm25_columns: list[str] | None = None,
    query_col: str = "query_text",
    doc_col: str = "doc_text",
    tokenizer: tp.Callable[[str], list[str]] | None = None,
    feature_fns: list[FeatureFn] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    feature_fns = feature_fns or DEFAULT_FEATURE_FUNCTIONS
    df_features = add_ltr_features(
        df,
        query_col=query_col,
        doc_col=doc_col,
        tokenizer=tokenizer,
        feature_fns=feature_fns,
    )
    bm25_columns = bm25_columns or infer_bm25_columns(df_features)
    if not bm25_columns:
        raise ValueError("No BM25 feature columns found in input dataframe.")

    extra_feature_cols = [f"f{idx}" for idx in range(1, len(feature_fns) + 1)]
    feature_columns = bm25_columns + extra_feature_cols
    return df_features, feature_columns


def add_relevance_labels(
    df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    query_col: str = "query_id",
    doc_col: str = "doc_id",
    relevance_col: str = "relevance",
    missing_value: float = 0.0,
) -> pd.DataFrame:
    _validate_dataframe_columns(qrels_df, {query_col, doc_col, relevance_col}, "qrels_df")
    merged = df.merge(
        qrels_df[[query_col, doc_col, relevance_col]],
        on=[query_col, doc_col],
        how="left",
    )
    merged[relevance_col] = merged[relevance_col].fillna(missing_value)
    return merged


def sample_balanced_judgments(
    df: pd.DataFrame,
    label_col: str = "relevance",
    group_col: str = "query_id",
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Sample a balanced set of positive and negative document judgments per query group.

    For each group (typically a query), the function samples all positive (relevant) examples and an equal number of negative (non-relevant) examples, randomly selecting among the negatives. Returns the balanced DataFrame and the list of selected indices.
    """
    rng = np.random.RandomState(random_state)
    sampled_indices: list[int] = []

    for _, group in df.groupby(group_col, sort=False):
        pos_idx = group[group[label_col] > 0].index
        if pos_idx.empty:
            continue
        neg_idx = group[group[label_col] <= 0].index
        if len(neg_idx) > 0:
            sample_size = min(len(pos_idx), len(neg_idx))
            neg_sample = rng.choice(neg_idx.to_numpy(), size=sample_size, replace=False)
        else:
            neg_sample = []
        sampled_indices.extend(pos_idx.tolist())
        sampled_indices.extend(list(neg_sample))

    sampled_indices = sorted(set(sampled_indices))
    return df.loc[sampled_indices].copy(), sampled_indices


def ndcg_at_k(
    df: pd.DataFrame,
    scores: tp.Sequence[float],
    k: int = DEFAULT_TARGET_K,
    query_col: str = "query_id",
    doc_col: str = "doc_id",
    relevance_col: str = "relevance",
) -> float:
    qrels = [
        irm.Qrel(str(q), str(d), int(r))
        for q, d, r in zip(df[query_col], df[doc_col], df[relevance_col])
    ]
    run = [
        irm.ScoredDoc(str(q), str(d), float(s))
        for q, d, s in zip(df[query_col], df[doc_col], scores)
    ]
    measure = irm.nDCG @ k
    return float(irm.calc_aggregate([measure], qrels, run)[measure])


def split_by_query_group(
    df: pd.DataFrame,
    group_col: str = "query_id",
    label_col: str = "relevance",
    train_size: float = 0.7,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1, train_size=train_size, random_state=random_state
    )
    train_idx, eval_idx = next(splitter.split(df, df[label_col], df[group_col]))
    return df.iloc[train_idx], df.iloc[eval_idx]


def _combine_dense_sparse(
    dense_matrix: tp.Any, sparse_matrix: sparse.spmatrix | None
) -> tp.Any:
    if sparse_matrix is None:
        return dense_matrix
    dense_sparse = sparse.csr_matrix(dense_matrix)
    if dense_sparse.shape[0] != sparse_matrix.shape[0]:
        raise ValueError("Dense and sparse feature matrices must have same rows.")
    return sparse.hstack([dense_sparse, sparse_matrix], format="csr")


def fit_feature_transformer(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    sparse_features: sparse.spmatrix | None = None,
    normalize: bool = True,
) -> tuple[SimpleImputer, StandardScaler | None, tp.Any]:
    imputer = SimpleImputer(strategy="mean")
    dense = imputer.fit_transform(train_df[feature_columns])
    X_train = _combine_dense_sparse(dense, sparse_features)
    scaler = None
    if normalize:
        with_mean = not sparse.issparse(X_train)
        scaler = StandardScaler(with_mean=with_mean)
        X_train = scaler.fit_transform(X_train)
    return imputer, scaler, X_train


def transform_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    imputer: SimpleImputer,
    scaler: StandardScaler | None,
    sparse_features: sparse.spmatrix | None = None,
) -> tp.Any:
    dense = imputer.transform(df[feature_columns])
    X = _combine_dense_sparse(dense, sparse_features)
    if scaler is not None:
        X = scaler.transform(X)
    return X


def train_linear_ranker(
    X_train: tp.Any,
    y_train: tp.Sequence[float],
) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_catboost_ranker(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    label_col: str = "relevance",
    group_col: str = "query_id",
    eval_df: pd.DataFrame | None = None,
    target_k: int = DEFAULT_TARGET_K,
    params: dict[str, tp.Any] | None = None,
    train_features: tp.Any | None = None,
    eval_features: tp.Any | None = None,
) -> CatBoostRanker:
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir = Path(__file__).resolve().parent / DEFAULT_CATBOOST_SUBDIR / f"catboost_{dt_str}"
    train_dir.mkdir(parents=True, exist_ok=True)
    default_params = {
        "iterations": 200,
        "loss_function": "YetiRank",
        "eval_metric": f"NDCG:top={target_k}",
        "custom_metric": [f"NDCG:top={target_k}"],
        "verbose": False,
        "random_seed": 0,
        "depth": 6,
        "learning_rate": 0.1,
        "train_dir": str(train_dir),
    }
    if params:
        default_params.update(params)
    train_features = train_features if train_features is not None else train_df[feature_columns]
    train_pool = Pool(
        train_features,
        train_df[label_col],
        group_id=train_df[group_col].astype(str),
    )
    eval_pool = None
    if eval_df is not None:
        eval_features = eval_features if eval_features is not None else eval_df[feature_columns]
        eval_pool = Pool(
            eval_features,
            eval_df[label_col],
            group_id=eval_df[group_col].astype(str),
        )

    ranker = CatBoostRanker(**default_params)
    ranker.fit(train_pool, eval_set=eval_pool, plot=True)
    return ranker


def train_and_evaluate_rankers(
    judgments_df: pd.DataFrame,
    feature_columns: list[str],
    label_col: str = "relevance",
    group_col: str = "query_id",
    eval_split: float = 0.3,
    random_state: int = 42,
    target_k: int = DEFAULT_TARGET_K,
    catboost_params: dict[str, tp.Any] | None = None,
    sparse_features: sparse.spmatrix | None = None,
    normalize: bool = True,
) -> LtrTrainingResult:
    splitter = GroupShuffleSplit(
        n_splits=1, train_size=1.0 - eval_split, random_state=random_state
    )
    train_idx, eval_idx = next(
        splitter.split(judgments_df, judgments_df[label_col], judgments_df[group_col])
    )
    train_df = judgments_df.iloc[train_idx]
    eval_df = judgments_df.iloc[eval_idx]

    sparse_train = sparse_features[train_idx] if sparse_features is not None else None
    sparse_eval = sparse_features[eval_idx] if sparse_features is not None else None

    linear_imputer, linear_scaler, X_train = fit_feature_transformer(
        train_df,
        feature_columns,
        sparse_features=sparse_train,
        normalize=normalize,
    )
    y_train = train_df[label_col].astype(float).to_numpy()
    linear_model = train_linear_ranker(X_train, y_train)
    X_eval = transform_features(
        eval_df,
        feature_columns,
        linear_imputer,
        linear_scaler,
        sparse_features=sparse_eval,
    )
    eval_linear_scores = linear_model.predict(X_eval)
    linear_ndcg = ndcg_at_k(
        eval_df,
        eval_linear_scores,
        k=target_k,
        query_col=group_col,
        doc_col="doc_id",
        relevance_col=label_col,
    )

    catboost_model = train_catboost_ranker(
        train_df,
        feature_columns,
        label_col=label_col,
        group_col=group_col,
        eval_df=eval_df,
        target_k=target_k,
        params=catboost_params,
        train_features=X_train,
        eval_features=X_eval,
    )
    eval_catboost_scores = catboost_model.predict(X_eval)
    catboost_ndcg = ndcg_at_k(
        eval_df,
        eval_catboost_scores,
        k=target_k,
        query_col=group_col,
        doc_col="doc_id",
        relevance_col=label_col,
    )

    evaluation = LtrEvaluation(
        measure=f"nDCG@{target_k}",
        linear_ndcg=linear_ndcg,
        catboost_ndcg=catboost_ndcg,
        train_size=len(train_df),
        eval_size=len(eval_df),
    )
    models = LtrModels(
        feature_columns=feature_columns,
        linear_model=linear_model,
        linear_imputer=linear_imputer,
        linear_scaler=linear_scaler,
        catboost_model=catboost_model,
    )
    return LtrTrainingResult(models=models, evaluation=evaluation)


def build_bm25_candidates(
    corpus_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    retrieval_depth: int,
    tokenizer: tp.Callable[[str], list[str]] | None = None,
    bm25_params: dict[str, tp.Any] | None = None,
    score_col: str = "bm25_1",
) -> pd.DataFrame:
    _validate_dataframe_columns(corpus_df, {"doc_id", "text"}, "corpus_df")
    _validate_dataframe_columns(queries_df, {"query_id", "text"}, "queries_df")
    if retrieval_depth <= 0:
        raise ValueError("retrieval_depth must be positive")

    corpus = [
        (str(doc_id), str(text))
        for doc_id, text in corpus_df[["doc_id", "text"]].itertuples(
            index=False, name=None
        )
    ]
    retriever = retrieval.BM25Retriever(
        corpus=corpus,
        tokenizer=tokenizer,
        **(bm25_params or {}),
    )

    rows: list[dict[str, tp.Any]] = []
    for query_id, query_text in queries_df[["query_id", "text"]].itertuples(
        index=False, name=None
    ):
        retrieval_result = retriever.retrieve(
            query_id=str(query_id),
            query=str(query_text),
            k=retrieval_depth,
        )
        for doc in retrieval_result.retrieved_docs:
            rows.append(
                {
                    "query_id": retrieval_result.query_id,
                    "query_text": retrieval_result.query,
                    "doc_id": doc.doc_id,
                    "doc_text": doc.text,
                    score_col: doc.score,
                }
            )

    return pd.DataFrame(rows)


def add_bm25_scores(
    candidates_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    tokenizer: tp.Callable[[str], list[str]] | None = None,
    bm25_params: dict[str, tp.Any] | None = None,
    score_col: str = "bm25_1",
    query_id_col: str = "query_id",
    query_text_col: str = "query_text",
    doc_id_col: str = "doc_id",
) -> pd.DataFrame:
    _validate_dataframe_columns(corpus_df, {"doc_id", "text"}, "corpus_df")
    _validate_dataframe_columns(
        candidates_df,
        {query_id_col, query_text_col, doc_id_col},
        "candidates_df",
    )

    corpus = [
        (str(doc_id), str(text))
        for doc_id, text in corpus_df[["doc_id", "text"]].itertuples(
            index=False, name=None
        )
    ]
    retriever = retrieval.BM25Retriever(
        corpus=corpus,
        tokenizer=tokenizer,
        **(bm25_params or {}),
    )
    doc_index = {doc_id: idx for idx, doc_id in enumerate(retriever.doc_ids)}

    scored_df = candidates_df.copy()
    for query_id, group in scored_df.groupby(query_id_col, sort=False):
        query_text = str(group[query_text_col].iloc[0])
        tokenized_query = retriever.tokenizer(query_text)
        scores = retriever.bm25.get_scores(tokenized_query)
        scored_df.loc[group.index, score_col] = [
            float(scores[doc_index[str(doc_id)]]) for doc_id in group[doc_id_col]
        ]

    return scored_df


def rerank_candidates(
    candidates_df: pd.DataFrame,
    scores: tp.Sequence[float],
    top_k: int,
    query_col: str = "query_id",
    query_text_col: str = "query_text",
    doc_col: str = "doc_id",
    doc_text_col: str = "doc_text",
    score_col: str = "score",
) -> list[retrieval.RetrievalResult]:
    if len(candidates_df) != len(scores):
        raise ValueError("scores length must match candidates_df length")

    df = candidates_df.copy()
    df[score_col] = scores
    results: list[retrieval.RetrievalResult] = []
    for query_id, group in df.groupby(query_col, sort=False):
        group_sorted = group.sort_values(score_col, ascending=False).head(top_k)
        query_text = (
            str(group_sorted[query_text_col].iloc[0])
            if query_text_col in group_sorted.columns
            else ""
        )
        retrieved_docs = [
            retrieval.RetrievalDoc(
                doc_id=str(row[doc_col]),
                text=str(row[doc_text_col]),
                score=float(row[score_col]),
            )
            for _, row in group_sorted.iterrows()
        ]
        results.append(
            retrieval.RetrievalResult(
                query_id=str(query_id),
                query=query_text,
                retrieved_docs=retrieved_docs,
            )
        )
    return results


def run_reranking_pipeline(
    candidates_df: pd.DataFrame,
    scores: tp.Sequence[float],
    qrels_df: pd.DataFrame,
    metrics: tp.Iterable[tp.Any] | None = None,
    ks: tp.Iterable[int] | None = None,
    retriever_name: str = "LTRRanker",
    retrieval_depth: int | None = None,
    return_intermediate: bool = False,
) -> retrieval.RetrievalPipelineResult:
    metrics = list(metrics) if metrics is not None else [irm.nDCG @ DEFAULT_TARGET_K]
    ks = list(ks) if ks is not None else [DEFAULT_TARGET_K]
    top_k = max(int(k) for k in ks if int(k) > 0)
    retrieval_depth = retrieval_depth or top_k

    reranked_results = rerank_candidates(candidates_df, scores, top_k=top_k)
    metric_scores, intermediate_results = retrieval.evaluate_retrieval(
        retrieval_results=reranked_results,
        measures=metrics,
        qrels=retrieval.qrels_from_dataframe(qrels_df),
        return_intermediate=return_intermediate,
    )

    unique_ks = sorted({int(k) for k in ks if int(k) > 0})
    return retrieval.RetrievalPipelineResult(
        retriever_name=retriever_name,
        tokenizer_name=None,
        ks=unique_ks,
        retrieval_depth=retrieval_depth,
        retrieval_results=reranked_results,
        metrics={measure: float(metric_scores[measure]) for measure in metrics},
        intermediate_results=intermediate_results,
    )
