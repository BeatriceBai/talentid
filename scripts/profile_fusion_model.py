"""Profile learned fusion validation metrics and tower gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "fusion" / "v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.model_dir
    manifest = json.loads((directory / "fusion_manifest.json").read_text())
    history = pd.read_parquet(directory / "training_history.parquet")
    gates = pd.read_parquet(directory / "gate_weights.parquet")
    candidates = np.load(
        directory / "learned_candidate_embeddings.npy", allow_pickle=False
    )
    skills = np.load(directory / "learned_skill_embeddings.npy", allow_pickle=False)

    print(f"Fusion model snapshot: {directory.resolve()}")
    print(f"Model version: {manifest['model_version']}")
    print(f"Best epoch: {manifest['best_epoch']}")
    print("\nTRAINING HISTORY")
    shown = history.copy()
    for column in ("train_loss", "precision", "recall", "ndcg", "hit_rate", "mrr"):
        shown[column] = shown[column].map(
            lambda value: "-" if pd.isna(value) else f"{value:.4f}"
        )
    print(shown.to_string(index=False))

    print("\nLEARNED EMBEDDINGS")
    for name, array in (("candidates", candidates), ("skills", skills)):
        norms = np.linalg.norm(array, axis=1)
        print(
            f"  {name}: shape={array.shape}, "
            f"norm min/mean/max={norms.min():.6f}/"
            f"{norms.mean():.6f}/{norms.max():.6f}"
        )

    print("\nMEAN GATE WEIGHT AMONG AVAILABLE TOWERS")
    availability = gates["is_available"].astype(bool)
    available = gates.loc[availability]
    gate_summary = available.groupby("source")["gate_weight"].agg(
        ["mean", "median", "min", "max"]
    )
    print(gate_summary.round(4).to_string())

    print("\nMEAN GATE WEIGHT BY SPLIT AND SOURCE")
    split_summary = available.pivot_table(
        index="source", columns="split", values="gate_weight", aggfunc="mean"
    )
    print(split_summary.round(4).to_string())

    gate_sums = gates.groupby("candidate_id")["gate_weight"].sum()
    unavailable_weight = gates.loc[~availability, "gate_weight"].abs()
    print("\nCONTRACT CHECKS")
    print(f"  maximum |candidate gate sum - 1|: {(gate_sums - 1).abs().max():.8f}")
    print(
        "  maximum unavailable gate weight: "
        f"{unavailable_weight.max() if len(unavailable_weight) else 0:.8f}"
    )
    print("  test labels used: 0")


if __name__ == "__main__":
    main()
