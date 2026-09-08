"""Profile retrieval metrics and reliability-fusion lift."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "evaluation" / "retrieval" / "v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def _format_metrics(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    for column in ("precision", "recall", "ndcg", "hit_rate", "mrr"):
        output[column] = output[column].map(lambda value: f"{value:.4f}")
    if "catalog_coverage" in output:
        output["catalog_coverage"] = output["catalog_coverage"].map(
            lambda value: f"{value:.2%}"
        )
    return output


def main() -> None:
    args = parse_args()
    directory = args.data_dir
    metrics = pd.read_parquet(directory / "retrieval_metrics.parquet")
    by_coverage = pd.read_parquet(
        directory / "retrieval_metrics_by_coverage.parquet"
    )
    predictions = pd.read_parquet(directory / "retrieval_predictions.parquet")

    print(f"Retrieval snapshot: {directory.resolve()}")
    print("\nOVERALL METRICS")
    print(_format_metrics(metrics).to_string(index=False))
    catalog_size = int(metrics["catalog_size"].iloc[0])
    print(
        "\nRANDOM-RANKING REFERENCE"
        f"\n  expected Recall@30: {min(30 / catalog_size, 1.0):.4f}"
    )

    primary = metrics[metrics["k"].eq(30)].pivot(
        index="split", columns="strategy", values=["recall", "ndcg"]
    )
    if {"reliability_weighted", "uniform_mean"} <= set(
        metrics["strategy"]
    ):
        print("\nRELIABILITY LIFT AT K=30")
        for split in primary.index:
            recall_lift = (
                primary.loc[split, ("recall", "reliability_weighted")]
                - primary.loc[split, ("recall", "uniform_mean")]
            )
            ndcg_lift = (
                primary.loc[split, ("ndcg", "reliability_weighted")]
                - primary.loc[split, ("ndcg", "uniform_mean")]
            )
            print(
                f"  {split}: Recall@30 {recall_lift:+.4f}; "
                f"NDCG@30 {ndcg_lift:+.4f}"
            )

    print("\nMETRICS BY AVAILABLE-TOWER COVERAGE AT K=30")
    coverage_30 = by_coverage[by_coverage["k"].eq(30)]
    print(_format_metrics(coverage_30).to_string(index=False))

    print("\nPREDICTIONS")
    print(f"  rows: {len(predictions):,}")
    print(f"  candidates: {predictions['candidate_id'].nunique():,}")
    print(f"  unique retrieved skills: {predictions['skill_id'].nunique():,}")


if __name__ == "__main__":
    main()
