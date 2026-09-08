"""Profile embedding shapes, alignment, zero rows, and vector norms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "embeddings" / "v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def describe_array(name: str, array: np.ndarray) -> None:
    norms = np.linalg.norm(array, axis=1)
    print(f"\n{name}")
    print(f"  shape: {array.shape}")
    print(f"  dtype: {array.dtype}")
    print(f"  zero rows: {(norms == 0).sum():,} ({(norms == 0).mean():.1%})")
    nonzero = norms[norms > 0]
    if len(nonzero):
        print(
            "  nonzero norm min/mean/max: "
            f"{nonzero.min():.6f} / {nonzero.mean():.6f} / {nonzero.max():.6f}"
        )


def main() -> None:
    args = parse_args()
    directory = args.data_dir
    manifest = json.loads((directory / "embedding_manifest.json").read_text())
    tower = np.load(directory / "candidate_tower_embeddings.npy")
    skills = np.load(directory / "skill_embeddings.npy")
    candidates = np.load(directory / "candidate_embeddings.npy")
    tower_index = pd.read_parquet(
        directory / "candidate_tower_embedding_index.parquet"
    )
    skill_index = pd.read_parquet(directory / "skill_embedding_index.parquet")
    candidate_index = pd.read_parquet(
        directory / "candidate_embedding_index.parquet"
    )

    print(f"Embedding snapshot: {directory.resolve()}")
    print(
        "Model: "
        f"{manifest['model']['name']} @ {manifest['model']['revision'][:12]}"
    )
    describe_array("CANDIDATE TOWER EMBEDDINGS", tower)
    describe_array("SKILL EMBEDDINGS", skills)
    describe_array("FUSED CANDIDATE EMBEDDINGS", candidates)

    print("\nINDEX ALIGNMENT")
    print(f"  tower array/index: {len(tower):,} / {len(tower_index):,}")
    print(f"  skill array/index: {len(skills):,} / {len(skill_index):,}")
    print(
        f"  candidate array/index: {len(candidates):,} / {len(candidate_index):,}"
    )
    missing = tower_index.loc[~tower_index["is_available"]]
    zero_missing = np.linalg.norm(tower[missing["embedding_row"]], axis=1) == 0
    print(f"  missing towers encoded as zero: {zero_missing.mean():.1%}")

    print("\nTOWER AVAILABILITY")
    availability = tower_index.groupby("source")["is_available"].mean()
    print(availability.map(lambda value: f"{value:.1%}").to_string())

    print("\nSPLITS")
    print(candidate_index["split"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

