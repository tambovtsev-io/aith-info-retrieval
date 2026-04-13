from typing import Iterable

import ir_measures as irm
import pandas as pd


DEFAULT_METRICS = [
    irm.P @ 1,
    irm.P @ 10,
    irm.P @ 20,
    irm.MAP @ 20,
    irm.nDCG @ 20,
]


def _resolve_metrics(metrics: Iterable[irm.Measure | str] | None) -> list[irm.Measure]:
    if metrics is None:
        return list(DEFAULT_METRICS)
    resolved: list[irm.Measure] = []
    for metric in metrics:
        if isinstance(metric, str):
            resolved.append(irm.parse_measure(metric))
        else:
            resolved.append(metric)
    return resolved


def qrels_from_dataframe(qrels_df: pd.DataFrame) -> list[irm.Qrel]:
    qrels = []
    for _, row in qrels_df.iterrows():
        relevance = row["relevance"] if "relevance" in qrels_df.columns else row["score"]
        qrels.append(
            irm.Qrel(
                str(row["query_id"]),
                str(row["doc_id"]),
                int(relevance),
            )
        )
    return qrels


def run_from_dataframe(run_df: pd.DataFrame) -> list[irm.ScoredDoc]:
    return [
        irm.ScoredDoc(str(row["query_id"]), str(row["doc_id"]), float(row["score"]))
        for _, row in run_df.iterrows()
    ]


def evaluate_run(
    run_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    metrics: Iterable[irm.Measure | str] | None = None,
    return_intermediate: bool = False,
) -> tuple[dict[str, float], pd.DataFrame | None]:
    resolved_metrics = _resolve_metrics(metrics)
    qrels = qrels_from_dataframe(qrels_df)
    run = run_from_dataframe(run_df)

    aggregate = irm.calc_aggregate(resolved_metrics, qrels, run)
    aggregate_out = {str(metric): value for metric, value in aggregate.items()}

    if not return_intermediate:
        return aggregate_out, None

    rows = []
    for metric in irm.iter_calc(resolved_metrics, qrels, run):
        rows.append(
            {
                "query_id": metric.query_id,
                "measure": str(metric.measure),
                "value": metric.value,
            }
        )
    intermediate_df = pd.DataFrame(rows).pivot(
        index="query_id", columns="measure", values="value"
    ).reset_index()
    intermediate_df.columns.name = None
    return aggregate_out, intermediate_df
