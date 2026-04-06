import typing as tp
import ir_measures as irm
import pandas as pd
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from functools import lru_cache
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


DocumentId = str
CorpusEntry = tuple[str, str]


class RetrievalDoc(BaseModel):
    doc_id: str
    text: str
    score: float


class RetrievalResult(BaseModel):
    query_id: str
    query: str
    retrieved_docs: list[RetrievalDoc]

    def to_run_rows(self) -> list[dict[str, tp.Any]]:
        return [
            {
                "query_id": self.query_id,
                "doc_id": doc.doc_id,
                "score": doc.score,
            }
            for doc in self.retrieved_docs
        ]

    def to_scored_docs(self) -> list[irm.ScoredDoc]:
        return [
            irm.ScoredDoc(self.query_id, doc.doc_id, doc.score)
            for doc in self.retrieved_docs
        ]


class RetrievalPipelineResult(BaseModel):
    retriever_name: str
    tokenizer_name: str | None
    ks: list[int]
    retrieval_depth: int
    retrieval_results: list[RetrievalResult]
    metrics: dict[tp.Any, float]
    intermediate_results: list[dict[str, tp.Any]] | None

    def to_runs_dataframe(self) -> pd.DataFrame:
        rows = []
        for retrieval_result in self.retrieval_results:
            rows.extend(retrieval_result.to_run_rows())
        return pd.DataFrame(rows, columns=["query_id", "doc_id", "score"])

    def to_metrics_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.metrics])

    @property
    def intermediate_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.intermediate_results)


class BM25Retriever:
    def __init__(
        self,
        corpus: list[CorpusEntry],
        tokenizer: tp.Callable[[str], list[str]] | None = None,
        k1: float = 1.2,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.doc_ids = [doc_id for doc_id, _ in corpus]
        self.documents = [text for _, text in corpus]
        self.tokenizer = tokenizer if tokenizer is not None else lambda x: x.split()
        self.tokenized_documents = [
            self.tokenizer(document) for document in self.documents
        ]
        self.bm25 = BM25Okapi(
            corpus=self.tokenized_documents,
            k1=k1,
            b=b,
            epsilon=epsilon,
        )

    def retrieve(
        self,
        query_id: str,
        query: str,
        k: int = 10,
    ) -> RetrievalResult:
        if k <= 0:
            return RetrievalResult(
                query_id=query_id,
                query=query,
                retrieved_docs=[],
            )

        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-min(k, len(self.doc_ids)) :][::-1]
        return RetrievalResult(
            query_id=query_id,
            query=query,
            retrieved_docs=[
                RetrievalDoc(
                    doc_id=self.doc_ids[idx],
                    text=self.documents[idx],
                    score=float(scores[idx]),
                )
                for idx in top_indices
            ],
        )


def _ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


@lru_cache(maxsize=1)
def get_stopwords(langs: list[str] = ["english"]) -> frozenset[str]:
    _ensure_nltk_resource("corpora/stopwords", "stopwords")
    sw = set()
    for lang in langs:
        sw.update(stopwords.words(lang))
    return frozenset(sw)


@lru_cache(maxsize=1)
def get_lemmatizer() -> WordNetLemmatizer:
    _ensure_nltk_resource("corpora/wordnet", "wordnet")
    _ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def get_stemmer() -> PorterStemmer:
    return PorterStemmer()


def remove_punctuation(text: str) -> str:
    return "".join([char for char in text if char not in string.punctuation])


def remove_stopwords(text: str) -> str:
    return " ".join([word for word in text.split() if word not in get_stopwords()])


def lemmatize(text: str) -> str:
    return " ".join([get_lemmatizer().lemmatize(word) for word in text.split()])


def stem(text: str) -> str:
    return " ".join([get_stemmer().stem(word) for word in text.split()])


def sequential_transforms(
    transforms: list[tp.Callable[[str], str]],
) -> tp.Callable[[str], list[str]]:
    def func(text: str) -> list[str]:
        for transform in transforms:
            text = transform(text)
        return text.split()

    return func


tok_simple = sequential_transforms([remove_punctuation, remove_stopwords])
tok_lemmatize = sequential_transforms([remove_punctuation, remove_stopwords, lemmatize])
tok_stem = sequential_transforms([remove_punctuation, remove_stopwords, stem])


def evaluate_retrieval(
    retrieval_results: list[RetrievalResult],
    measures: list[tp.Any],
    qrels: list[irm.Qrel],
    return_intermediate: bool = False,
) -> tuple[dict[tp.Any, float], pd.DataFrame | None]:
    run = []
    for retrieval_result in retrieval_results:
        run.extend(retrieval_result.to_scored_docs())
    aggregate = irm.calc_aggregate(measures, qrels, run)
    if return_intermediate:
        rows = []
        for metric in irm.iter_calc(measures, qrels, run):
            rows.append({
                "query_id": metric.query_id,
                "measure": str(metric.measure),
                "value": metric.value,
            })
        intermediate_df = pd.DataFrame(rows).pivot(
            index="query_id", columns="measure", values="value"
        ).reset_index()
        intermediate_df.columns.name = None
        return aggregate, intermediate_df.to_dict(orient="records")
    return aggregate, None


def qrels_from_dataframe(qrels_df: pd.DataFrame) -> list[irm.Qrel]:
    qrels = []
    for i, row in qrels_df.iterrows():
        qrels.append(
            irm.Qrel(
                str(row["query_id"]),
                str(row["doc_id"]),
                int(row["relevance"]),
            )
        )
    return qrels


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


def run_retrieval_pipeline(
    retriever_cls: type[tp.Any],
    tokenizer: tp.Callable[[str], list[str]] | None,
    corpus_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    metrics: tp.Iterable[tp.Any],
    ks: tp.Iterable[int],
    return_intermediate: bool = False,
    retriever_kwargs: dict[str, tp.Any] | None = None,
) -> RetrievalPipelineResult:
    _validate_dataframe_columns(corpus_df, {"doc_id", "text"}, "corpus_df")
    _validate_dataframe_columns(queries_df, {"query_id", "text"}, "queries_df")

    corpus = [
        (str(doc_id), str(text))
        for doc_id, text in corpus_df[["doc_id", "text"]].itertuples(
            index=False, name=None
        )
    ]
    if not corpus:
        raise ValueError("corpus_df must contain at least one document")

    retriever = retriever_cls(
        corpus=corpus,
        tokenizer=tokenizer,
        **(retriever_kwargs or {}),
    )
    retrieval_depth = max(ks)
    retrieval_results = [
        retriever.retrieve(
            query_id=str(query_id),
            query=str(query_text),
            k=retrieval_depth,
        )
        for query_id, query_text in queries_df[["query_id", "text"]].itertuples(
            index=False, name=None
        )
    ]

    measures = list(metrics)
    metric_scores, intermediate_results = evaluate_retrieval(
        retrieval_results=retrieval_results,
        measures=measures,
        qrels=qrels_from_dataframe(qrels_df),
        return_intermediate=return_intermediate,
    )

    unique_ks = sorted({int(k) for k in ks if int(k) > 0})
    return RetrievalPipelineResult(
        retriever_name=retriever_cls.__name__,
        tokenizer_name=getattr(tokenizer, "__name__", None),
        ks=unique_ks,
        retrieval_depth=retrieval_depth,
        retrieval_results=retrieval_results,
        metrics={measure: float(metric_scores[measure]) for measure in measures},
        intermediate_results=intermediate_results,
    )