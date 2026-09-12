"""Profile recall and long-tail coverage of hybrid candidate generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "evaluation" / "hybrid_retrieval" / "v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (args.data_dir / "hybrid_retrieval_manifest.json").read_text()
    )
    metrics = pd.read_parquet(args.data_dir / "hybrid_retrieval_metrics.parquet")
    predictions = pd.read_parquet(
        args.data_dir / "hybrid_retrieval_predictions.parquet"
    )

    shown = metrics.copy()
    for column in (
        "catalog_coverage",
        "truth_skill_coverage",
        "truth_pair_recall",
        "mean_candidate_recall",
    ):
        shown[column] = shown[column].map(lambda value: f"{value:.2%}")

    print(f"Hybrid retrieval snapshot: {args.data_dir.resolve()}")
    print(f"Version: {manifest['retrieval_version']}")
    print("\nPOOL METRICS")
    print(shown.to_string(index=False))

    hybrid = metrics.set_index("stage")
    learned = hybrid.loc["learned_top100"]
    expanded = hybrid.loc["hybrid_top200"]
    print("\nTOP-100 TO HYBRID-200 LIFT")
    print(
        "  truth-pair recall: "
        f"{learned['truth_pair_recall']:.2%} -> "
        f"{expanded['truth_pair_recall']:.2%} "
        f"({expanded['truth_pair_recall'] - learned['truth_pair_recall']:+.2%})"
    )
    print(
        "  truth-skill coverage: "
        f"{learned['truth_skill_coverage']:.2%} -> "
        f"{expanded['truth_skill_coverage']:.2%} "
        f"({expanded['truth_skill_coverage'] - learned['truth_skill_coverage']:+.2%})"
    )
    print(
        "  catalog coverage: "
        f"{learned['catalog_coverage']:.2%} -> "
        f"{expanded['catalog_coverage']:.2%} "
        f"({expanded['catalog_coverage'] - learned['catalog_coverage']:+.2%})"
    )

    extras = predictions[~predictions["is_learned_protected"]]
    print("\nEXTRA-CANDIDATE CHANNEL SIGNALS")
    print(f"  extra rows: {len(extras):,}")
    print(f"  occupation hits: {extras['occupation_rank'].notna().mean():.2%}")
    print(f"  tower hits: {extras['tower_hit_count'].gt(0).mean():.2%}")
    print(f"  mean contributing channels: {extras['channel_count'].mean():.2f}")
    print(f"  deterministic fallback: {extras['is_fallback'].mean():.2%}")
    print(f"  learned top-100 preserved: {manifest['learned_top100_preserved']}")
    print(f"  test labels used: {manifest['test_labels_used']}")


if __name__ == "__main__":
    main()
