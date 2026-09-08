"""Profile candidate-tower and skill feature artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "features" / "v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    """Print coverage, size, reliability, and quality-flag summaries."""
    args = parse_args()
    towers = pd.read_parquet(args.data_dir / "candidate_tower_features.parquet")
    skills = pd.read_parquet(args.data_dir / "skill_features.parquet")
    candidate_index = pd.read_parquet(args.data_dir / "candidate_index.parquet")

    print(f"Feature snapshot: {args.data_dir}")
    print(f"Candidates: {len(candidate_index):,}")
    print(f"Candidate-tower rows: {len(towers):,}")
    print(f"Skill rows: {len(skills):,}")

    print("\nSOURCE COVERAGE")
    coverage = towers.groupby("source")["is_available"].mean()
    print(coverage.map(lambda value: f"{value:.1%}").to_string())

    print("\nAVAILABLE TEXT LENGTH QUANTILES (characters)")
    available = towers[towers["is_available"]]
    lengths = available.groupby("source")["character_count"].quantile(
        [0.5, 0.9, 0.99, 1.0]
    ).unstack()
    print(lengths.round(0).astype(int).to_string())

    print("\nRELIABILITY SCORE")
    reliability = available.groupby("source")["reliability_score"].agg(
        ["mean", "min", "max"]
    )
    print(reliability.round(3).to_string())

    print("\nQUALITY FLAGS")
    flags = towers.groupby("source").agg(
        stale_rate=("is_stale", "mean"),
        high_volume_rate=("is_high_volume", "mean"),
    )
    print(flags.map(lambda value: f"{value:.1%}").to_string())

    print("\nSPLITS")
    print(candidate_index["split"].value_counts().sort_index().to_string())

    print("\nSKILL TEXT LENGTH QUANTILES")
    print(skills["skill_text"].str.len().quantile([0, 0.5, 0.9, 0.99, 1]).to_string())


if __name__ == "__main__":
    main()

