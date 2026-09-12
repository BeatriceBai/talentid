"""Compare the selected hybrid reranker with the v1 reranking baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V1_DIR = PROJECT_ROOT / "data" / "evaluation" / "reranking" / "v1"
DEFAULT_V2_DIR = PROJECT_ROOT / "data" / "evaluation" / "reranking" / "v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-dir", type=Path, default=DEFAULT_V1_DIR)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    return parser.parse_args()


def _selected(comparison: pd.DataFrame) -> pd.Series:
    return comparison[comparison["strategy"].eq("selected_reranker")].iloc[0]


def main() -> None:
    args = parse_args()
    v1 = pd.read_parquet(args.v1_dir / "reranking_comparison.parquet")
    v2 = pd.read_parquet(args.v2_dir / "reranking_comparison.parquet")
    trials = pd.read_parquet(args.v2_dir / "reranking_trials.parquet")
    predictions = pd.read_parquet(args.v2_dir / "reranked_predictions.parquet")
    manifest = json.loads((args.v2_dir / "reranking_manifest.json").read_text())
    chosen = trials[trials["is_selected"]].iloc[0]

    rows = []
    v1_selected = _selected(v1).copy()
    v1_selected["strategy"] = "v1_learned100_reranker"
    rows.append(v1_selected)
    v2_baseline = v2[v2["strategy"].eq("retrieval_top30")].iloc[0].copy()
    v2_baseline["strategy"] = "v2_hybrid200_unmodified"
    rows.append(v2_baseline)
    v2_selected = _selected(v2).copy()
    v2_selected["strategy"] = "v2_hybrid200_reranker"
    rows.append(v2_selected)
    comparison = pd.DataFrame(rows)

    print(f"Hybrid reranking snapshot: {args.v2_dir.resolve()}")
    print(f"Version: {manifest['reranker_version']}")
    print("\nSELECTED CONFIGURATION")
    print(f"  trial: {int(chosen['trial'])}")
    print(f"  evidence weight: {chosen['evidence_weight']:.2f}")
    print(f"  popularity penalty: {chosen['popularity_penalty']:.2f}")
    print(f"  diversity penalty: {chosen['diversity_penalty']:.2f}")
    print(f"  NDCG floor: {chosen['ndcg_floor']:.4f}")

    print("\nV1 VS HYBRID V2")
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

    old = comparison.iloc[0]
    new = comparison.iloc[2]
    print("\nV2 SELECTED LIFT OVER V1")
    for metric in (
        "precision",
        "recall",
        "ndcg",
        "hit_rate",
        "mrr",
        "catalog_coverage",
    ):
        delta = float(new[metric] - old[metric])
        print(f"  {metric}: {old[metric]:.4f} -> {new[metric]:.4f} ({delta:+.4f})")

    counts = predictions.groupby("candidate_id").size()
    print("\nSEARCH AND CONTRACT CHECKS")
    print(f"  configurations evaluated: {len(trials)}")
    print(
        f"  configurations within NDCG budget: {trials['meets_ndcg_constraint'].sum()}"
    )
    print(f"  candidates: {predictions['candidate_id'].nunique():,}")
    print(f"  predictions: {len(predictions):,}")
    print(f"  rows per candidate min/max: {counts.min()} / {counts.max()}")
    print(
        "  duplicate candidate-skill pairs: "
        f"{predictions.duplicated(['candidate_id', 'skill_id']).sum()}"
    )
    print(f"  test labels used: {manifest['test_labels_used']}")


if __name__ == "__main__":
    main()
