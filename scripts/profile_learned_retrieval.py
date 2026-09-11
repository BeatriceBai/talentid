"""Profile learned exact/ANN retrieval and baseline lift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "evaluation" / "learned_retrieval" / "v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.data_dir
    manifest = json.loads((directory / "learned_retrieval_manifest.json").read_text())
    metrics = pd.read_parquet(directory / "learned_retrieval_metrics.parquet")
    fidelity = pd.read_parquet(directory / "ann_fidelity.parquet")
    benchmark = pd.read_parquet(directory / "search_benchmark.parquet")
    comparison = pd.read_parquet(directory / "baseline_comparison.parquet")

    print(f"Learned retrieval snapshot: {directory.resolve()}")
    print(f"Evaluation version: {manifest['evaluation_version']}")
    print("\nRANKING METRICS")
    shown = metrics.copy()
    for column in (
        "catalog_coverage",
        "precision",
        "recall",
        "ndcg",
        "hit_rate",
        "mrr",
    ):
        shown[column] = shown[column].map(lambda value: f"{value:.4f}")
    print(shown.to_string(index=False))

    print("\nANN FIDELITY")
    fidelity_shown = fidelity.copy()
    fidelity_shown["neighbor_recall"] = fidelity_shown["neighbor_recall"].map(
        lambda value: f"{value:.4f}"
    )
    print(fidelity_shown.to_string(index=False))

    print("\nSEARCH BENCHMARK")
    benchmark_shown = benchmark.copy()
    for column in ("index_build_seconds", "search_seconds", "queries_per_second"):
        benchmark_shown[column] = benchmark_shown[column].map(
            lambda value: f"{value:.4f}"
        )
    print(benchmark_shown.to_string(index=False))

    print("\nFROZEN-TO-LEARNED COMPARISON AT K=30")
    at_30 = comparison[comparison["k"].eq(30)].copy()
    for column in (
        "frozen_baseline",
        "learned_model",
        "absolute_lift",
        "relative_lift",
    ):
        at_30[column] = at_30[column].map(lambda value: f"{value:.4f}")
    print(at_30.to_string(index=False))

    max_k = max(manifest["config"]["cutoffs"])
    hnsw_recall = fidelity.loc[
        fidelity["engine"].eq("faiss_hnsw") & fidelity["k"].eq(max_k),
        "neighbor_recall",
    ].iloc[0]
    minimum = manifest["config"]["minimum_recall_at_max_k"]
    print("\nCONTRACT CHECKS")
    print(f"  indexed skills: {manifest['outputs']['hnsw_index']['vectors']:,}")
    print(f"  HNSW recall@{max_k}: {hnsw_recall:.4f} (minimum {minimum:.4f})")
    print(f"  test labels used: {manifest['test_labels_used']}")


if __name__ == "__main__":
    main()
