"""Profile candidate-skill positives and mined negatives."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "training" / "pairs" / "v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.data_dir
    positives = pd.read_parquet(directory / "positive_pairs.parquet")
    negatives = pd.read_parquet(directory / "negative_pairs.parquet")
    candidates = pd.read_parquet(directory / "training_candidate_index.parquet")

    print(f"Training-pair snapshot: {directory.resolve()}")
    print("\nTABLE ROWS")
    print(f"  positive_pairs: {len(positives):,}")
    print(f"  negative_pairs: {len(negatives):,}")
    print(f"  training_candidate_index: {len(candidates):,}")

    print("\nCANDIDATES BY SPLIT")
    print(candidates["split"].value_counts().sort_index().to_string())
    print("\nPOSITIVE PAIRS BY SPLIT")
    print(positives["split"].value_counts().sort_index().to_string())
    print("\nNEGATIVE PAIRS BY TYPE")
    print(negatives["negative_type"].value_counts().sort_index().to_string())

    positive_counts = positives.groupby("candidate_id").size()
    negative_counts = negatives.groupby("candidate_id").size()
    print("\nPOSITIVE COUNT QUANTILES")
    print(positive_counts.quantile([0, 0.25, 0.5, 0.75, 1]).to_string())
    print("\nTRAIN NEGATIVE COUNT QUANTILES")
    print(negative_counts.quantile([0, 0.5, 1]).to_string())
    print("\nPOSITIVE SAMPLE-WEIGHT QUANTILES")
    print(
        positives["sample_weight"]
        .quantile([0, 0.25, 0.5, 0.75, 1])
        .to_string()
    )

    hard_scores = negatives.loc[
        negatives["negative_type"].eq("hard_retrieval"), "retrieval_score"
    ]
    print("\nHARD-NEGATIVE SCORE QUANTILES")
    print(hard_scores.quantile([0, 0.25, 0.5, 0.75, 1]).to_string())

    positive_keys = set(zip(positives["candidate_id"], positives["skill_id"]))
    negative_keys = set(zip(negatives["candidate_id"], negatives["skill_id"]))
    print("\nCONTRACT CHECKS")
    print(f"  positive/negative overlap: {len(positive_keys & negative_keys):,}")
    print(
        "  duplicate negative keys: "
        f"{negatives.duplicated(['candidate_id', 'skill_id']).sum():,}"
    )
    print(f"  test candidates exposed: {(candidates['split'] == 'test').sum():,}")


if __name__ == "__main__":
    main()

