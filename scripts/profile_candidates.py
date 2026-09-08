"""Profile generated candidate tables and their simulated data problems."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "synthetic" / "candidates" / "v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    """Print concise quality and distribution checks for the snapshot."""
    args = parse_args()
    paths = sorted(args.data_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No generated candidate tables found in {args.data_dir}"
        )

    tables = {path.stem: pd.read_parquet(path) for path in paths}
    candidates = tables["candidates"]
    status = tables["source_status"]
    claims = tables["self_reported_skills"]

    print(f"Candidate snapshot: {args.data_dir}")
    print("\nTABLE ROWS")
    for name, table in tables.items():
        print(f"  {name}: {len(table):,}")

    print("\nSPLITS")
    print(candidates["split"].value_counts().sort_index().to_string())

    print("\nMISSING SOURCE RATE")
    missing = 1 - status.groupby("source")["is_available"].mean()
    print(missing.sort_index().map(lambda value: f"{value:.1%}").to_string())

    resumes = status[status["source"].eq("resume")]
    available_resumes = resumes[resumes["is_available"]]
    stale_rate = available_resumes["age_days"].gt(730).mean()
    print(f"\nSTALE RESUMES (>730 days): {stale_rate:.1%}")

    claim_counts = claims.groupby("candidate_id").size()
    print("\nSELF-REPORTED SKILL COUNT QUANTILES")
    print(claim_counts.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]).to_string())
    noisy_rate = 1 - claims["oracle_is_true_skill"].mean()
    alias_rate = claims["alias_type"].ne("canonical").mean()
    print(f"Noisy claims: {noisy_rate:.1%}")
    print(f"Noncanonical surface forms: {alias_rate:.1%}")

    print("\nPROFILE SEGMENTS")
    print(candidates["profile_segment"].value_counts(normalize=True).to_string())


if __name__ == "__main__":
    main()

