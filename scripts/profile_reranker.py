"""Profile the selected evidence- and diversity-aware reranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "evaluation" / "reranking" / "v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.data_dir
    manifest = json.loads((directory / "reranking_manifest.json").read_text())
    trials = pd.read_parquet(directory / "reranking_trials.parquet")
    comparison = pd.read_parquet(directory / "reranking_comparison.parquet")
    predictions = pd.read_parquet(directory / "reranked_predictions.parquet")
    selected = trials[trials["is_selected"]].iloc[0]

    print(f"Reranking snapshot: {directory.resolve()}")
    print(f"Reranker version: {manifest['reranker_version']}")
    print("\nSELECTED CONFIGURATION")
    print(f"  trial: {int(selected['trial'])}")
    print(f"  evidence weight: {selected['evidence_weight']:.2f}")
    print(f"  popularity penalty: {selected['popularity_penalty']:.2f}")
    print(f"  diversity penalty: {selected['diversity_penalty']:.2f}")
    print(f"  NDCG floor: {selected['ndcg_floor']:.4f}")

    print("\nRETRIEVAL VS RERANKING")
    columns = [
        "strategy",
        "precision",
        "recall",
        "ndcg",
        "hit_rate",
        "mrr",
        "catalog_coverage",
        "mean_popularity",
        "mean_intra_list_similarity",
    ]
    shown = comparison[columns].copy()
    for column in columns[1:]:
        shown[column] = shown[column].map(lambda value: f"{value:.4f}")
    print(shown.to_string(index=False))

    print("\nSEARCH SUMMARY")
    print(f"  configurations evaluated: {len(trials)}")
    print(
        f"  configurations within NDCG budget: {trials['meets_ndcg_constraint'].sum()}"
    )
    print(
        f"  maximum observed catalog coverage: {trials['catalog_coverage'].max():.2%}"
    )

    counts = predictions.groupby("candidate_id").size()
    print("\nCONTRACT CHECKS")
    print(f"  candidates: {predictions['candidate_id'].nunique():,}")
    print(f"  predictions: {len(predictions):,}")
    print(f"  rows per candidate min/max: {counts.min()} / {counts.max()}")
    print(
        f"  duplicate candidate-skill pairs: {predictions.duplicated(['candidate_id', 'skill_id']).sum()}"
    )
    print(f"  test labels used: {manifest['test_labels_used']}")


if __name__ == "__main__":
    main()
