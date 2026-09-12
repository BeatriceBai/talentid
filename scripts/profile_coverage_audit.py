"""Profile truth diversity, pool ceilings, and coverage concentration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "evaluation" / "coverage_audit" / "v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.data_dir
    manifest = json.loads((directory / "coverage_audit_manifest.json").read_text())
    summary = pd.read_parquet(directory / "coverage_summary.parquet")
    by_type = pd.read_parquet(directory / "coverage_by_skill_type.parquet")
    frequencies = pd.read_parquet(directory / "skill_prediction_frequency.parquet")

    print(f"Coverage audit: {directory.resolve()}")
    print(f"Audit version: {manifest['audit_version']}")
    print("\nSTAGE SUMMARY")
    columns = [
        "stage",
        "unique_skills",
        "catalog_coverage",
        "truth_unique_skills",
        "truth_catalog_coverage",
        "truth_skill_coverage",
        "truth_pair_recall",
        "mean_candidate_recall",
        "top_1pct_prediction_share",
        "normalized_entropy",
    ]
    shown = summary[columns].copy()
    for column in columns[2:]:
        if column != "truth_unique_skills":
            shown[column] = shown[column].map(lambda value: f"{value:.4f}")
    print(shown.to_string(index=False))

    print("\nRERANKED COVERAGE BY SKILL TYPE")
    selected_type = by_type[by_type["stage"].eq("reranked_top30")].copy()
    for column in ("catalog_coverage", "truth_skill_coverage"):
        selected_type[column] = selected_type[column].map(
            lambda value: "-" if pd.isna(value) else f"{value:.4f}"
        )
    print(selected_type.to_string(index=False))

    print("\nTOP 10 RERANKED SKILLS BY EXPOSURE")
    top = frequencies[frequencies["stage"].eq("reranked_top30")].nsmallest(
        10, "frequency_rank"
    )
    print(
        top[
            [
                "frequency_rank",
                "skill_id",
                "skill_name",
                "skill_type",
                "prediction_count",
                "prediction_share",
            ]
        ].to_string(index=False)
    )

    pool = summary[summary["stage"].eq("learned_top100_pool")].iloc[0]
    reranked = summary[summary["stage"].eq("reranked_top30")].iloc[0]
    print("\nDECISION CHECKS")
    print(
        "  pool truth-pair recall: "
        f"{pool['truth_pair_recall']:.2%} "
        f"(minimum {manifest['config']['minimum_pool_pair_recall']:.2%})"
    )
    print(
        "  pool truth-skill coverage: "
        f"{pool['truth_skill_coverage']:.2%} "
        f"(minimum {manifest['config']['minimum_pool_truth_skill_coverage']:.2%})"
    )
    print(
        "  reranked truth-skill coverage: "
        f"{reranked['truth_skill_coverage']:.2%} "
        f"(minimum {manifest['config']['minimum_reranked_truth_skill_coverage']:.2%})"
    )
    print(f"  recommendation: {manifest['recommendation']}")
    print(f"  test labels used: {manifest['test_labels_used']}")


if __name__ == "__main__":
    main()
