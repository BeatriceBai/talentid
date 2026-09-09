"""Build leakage-safe positive and hard-negative candidate-skill pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from talentid.evaluation.evaluate_retrieval import retrieve_top_k

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings" / "v1"
DEFAULT_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "candidates"
    / "v1"
    / "candidate_skill_truth.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "training" / "pairs" / "v1"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "training_pairs.toml"


@dataclass(frozen=True)
class PairMiningConfig:
    """Configuration for deterministic positive and negative construction."""

    mining_version: str = "talentskill-training-pairs-v1"
    positive_splits: tuple[str, ...] = ("train", "validation")
    negative_split: str = "train"
    hard_negatives_per_candidate: int = 30
    random_negatives_per_candidate: int = 10
    query_batch_size: int = 256
    seed: int = 20_260_908
    relevance_weight: float = 0.70
    proficiency_weight: float = 0.30
    minimum_weight: float = 0.10

    def validate(self) -> None:
        valid_splits = {"train", "validation", "test"}
        if not self.mining_version.strip():
            raise ValueError("mining_version must be nonempty")
        if not self.positive_splits or not set(self.positive_splits) <= valid_splits:
            raise ValueError("positive_splits contains an unknown split")
        if "test" in self.positive_splits or self.negative_split == "test":
            raise ValueError("the test split must remain sealed")
        if self.negative_split not in self.positive_splits:
            raise ValueError("negative_split must be included in positive_splits")
        if self.hard_negatives_per_candidate < 1:
            raise ValueError("at least one hard negative is required")
        if self.random_negatives_per_candidate < 0:
            raise ValueError("random negative count cannot be negative")
        if self.query_batch_size < 1:
            raise ValueError("query_batch_size must be positive")
        if self.relevance_weight < 0 or self.proficiency_weight < 0:
            raise ValueError("positive-weight components cannot be negative")
        if not np.isclose(self.relevance_weight + self.proficiency_weight, 1.0):
            raise ValueError("positive-weight components must sum to one")
        if not 0 <= self.minimum_weight <= 1:
            raise ValueError("minimum_weight must be between zero and one")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> PairMiningConfig:
    """Load pair-mining configuration from TOML."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    mining = raw["mining"]
    weights = raw["positive_weight"]
    config = PairMiningConfig(
        mining_version=str(mining["mining_version"]),
        positive_splits=tuple(str(value) for value in mining["positive_splits"]),
        negative_split=str(mining["negative_split"]),
        hard_negatives_per_candidate=int(
            mining["hard_negatives_per_candidate"]
        ),
        random_negatives_per_candidate=int(
            mining["random_negatives_per_candidate"]
        ),
        query_batch_size=int(mining["query_batch_size"]),
        seed=int(mining["seed"]),
        relevance_weight=float(weights["relevance_weight"]),
        proficiency_weight=float(weights["proficiency_weight"]),
        minimum_weight=float(weights["minimum_weight"]),
    )
    config.validate()
    return config


def _require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _validate_embedding_rows(table: pd.DataFrame, expected_length: int, name: str) -> None:
    rows = table["embedding_row"].tolist()
    if sorted(rows) != list(range(expected_length)):
        raise ValueError(f"{name} embedding rows must be a complete permutation")


def _positive_weight(truth: pd.DataFrame, config: PairMiningConfig) -> pd.Series:
    relevance = truth["relevance_score"].astype(float).clip(0, 1)
    proficiency = truth["proficiency"].astype(float).clip(1, 5) / 5.0
    combined = (
        config.relevance_weight * relevance
        + config.proficiency_weight * proficiency
    )
    return combined.clip(config.minimum_weight, 1.0)


def _sample_random_skill_positions(
    rng: np.random.Generator,
    catalog_size: int,
    forbidden: set[int],
    count: int,
) -> list[int]:
    """Sample unique catalog positions without allocating a full complement."""
    available = catalog_size - len(forbidden)
    if count > available:
        raise ValueError("not enough skills remain for random negatives")
    selected: set[int] = set()
    while len(selected) < count:
        position = int(rng.integers(0, catalog_size))
        if position not in forbidden:
            selected.add(position)
    return sorted(selected)


def _prepare_inputs(
    candidate_embeddings: np.ndarray,
    candidate_index: pd.DataFrame,
    skill_embeddings: np.ndarray,
    skill_index: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    _require_columns(
        candidate_index,
        {"embedding_row", "candidate_id", "split"},
        "candidate_embedding_index",
    )
    _require_columns(
        skill_index,
        {"embedding_row", "skill_id"},
        "skill_embedding_index",
    )
    _require_columns(
        truth,
        {
            "candidate_id",
            "skill_id",
            "relevance_score",
            "proficiency",
            "taxonomy_version",
        },
        "candidate_skill_truth",
    )
    if len(candidate_embeddings) != len(candidate_index):
        raise ValueError("candidate embeddings and index do not align")
    if len(skill_embeddings) != len(skill_index):
        raise ValueError("skill embeddings and index do not align")
    if candidate_index["candidate_id"].duplicated().any():
        raise ValueError("candidate ids must be unique")
    if skill_index["skill_id"].duplicated().any():
        raise ValueError("skill ids must be unique")
    if truth.duplicated(["candidate_id", "skill_id"]).any():
        raise ValueError("truth candidate-skill keys must be unique")
    _validate_embedding_rows(candidate_index, len(candidate_embeddings), "candidate")
    _validate_embedding_rows(skill_index, len(skill_embeddings), "skill")

    candidates = candidate_index.sort_values("embedding_row").reset_index(drop=True)
    candidate_rows = candidates["embedding_row"].to_numpy(dtype=int)
    candidate_vectors = candidate_embeddings[candidate_rows]
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)

    skills = skill_index.sort_values("embedding_row").reset_index(drop=True)
    skill_rows = skills["embedding_row"].to_numpy(dtype=int)
    skill_vectors = skill_embeddings[skill_rows]
    skills["skill_id"] = skills["skill_id"].astype(str)

    safe_truth = truth.copy()
    safe_truth["candidate_id"] = safe_truth["candidate_id"].astype(str)
    safe_truth["skill_id"] = safe_truth["skill_id"].astype(str)
    return candidates, candidate_vectors, skills, skill_vectors, safe_truth


def build_pair_tables(
    candidate_embeddings: np.ndarray,
    candidate_index: pd.DataFrame,
    skill_embeddings: np.ndarray,
    skill_index: pd.DataFrame,
    truth: pd.DataFrame,
    config: PairMiningConfig,
) -> dict[str, pd.DataFrame]:
    """Create visible positive pairs and train-only negative pairs."""
    config.validate()
    candidates, candidate_vectors, skills, skill_vectors, safe_truth = (
        _prepare_inputs(
            candidate_embeddings,
            candidate_index,
            skill_embeddings,
            skill_index,
            truth,
        )
    )
    candidate_split = candidates.set_index("candidate_id")["split"]
    skill_ids = skills["skill_id"].to_numpy()
    skill_position = {skill_id: index for index, skill_id in enumerate(skill_ids)}

    safe_truth["split"] = safe_truth["candidate_id"].map(candidate_split)
    if safe_truth["split"].isna().any():
        raise ValueError("truth contains unknown candidate ids")
    visible_truth = safe_truth[
        safe_truth["split"].isin(config.positive_splits)
    ].copy()
    unknown_skills = set(visible_truth["skill_id"]) - set(skill_ids)
    if unknown_skills:
        raise ValueError(f"visible truth contains {len(unknown_skills)} unknown skills")

    visible_truth["sample_weight"] = _positive_weight(visible_truth, config)
    positive_pairs = visible_truth[
        [
            "candidate_id",
            "skill_id",
            "split",
            "relevance_score",
            "proficiency",
            "sample_weight",
            "taxonomy_version",
        ]
    ].sort_values(["split", "candidate_id", "skill_id"], kind="stable")
    positive_pairs["mining_version"] = config.mining_version
    positive_pairs = positive_pairs.reset_index(drop=True)

    mining_mask = candidates["split"].eq(config.negative_split).to_numpy()
    mining_positions = np.flatnonzero(mining_mask)
    mining_ids = candidates.loc[mining_positions, "candidate_id"].tolist()
    mining_truth = positive_pairs[
        positive_pairs["split"].eq(config.negative_split)
    ]
    truth_by_candidate = {
        candidate_id: set(group["skill_id"])
        for candidate_id, group in mining_truth.groupby("candidate_id", sort=False)
    }
    missing_truth = sorted(set(mining_ids) - set(truth_by_candidate))
    if missing_truth:
        raise ValueError(f"{len(missing_truth)} mining candidates have no truth skills")

    maximum_truth = max(len(values) for values in truth_by_candidate.values())
    search_depth = min(
        len(skill_ids),
        maximum_truth + config.hard_negatives_per_candidate + 20,
    )
    retrieved_positions, retrieved_scores = retrieve_top_k(
        candidate_vectors[mining_positions],
        skill_vectors,
        search_depth,
        config.query_batch_size,
    )

    rng = np.random.default_rng(config.seed)
    negative_rows: list[dict[str, Any]] = []
    for row, candidate_id in enumerate(mining_ids):
        true_ids = truth_by_candidate[candidate_id]
        true_positions = {skill_position[skill_id] for skill_id in true_ids}
        hard_positions: list[int] = []
        for rank, position in enumerate(retrieved_positions[row], 1):
            position = int(position)
            if position in true_positions:
                continue
            hard_positions.append(position)
            negative_rows.append(
                {
                    "candidate_id": candidate_id,
                    "skill_id": str(skill_ids[position]),
                    "split": config.negative_split,
                    "negative_type": "hard_retrieval",
                    "retrieval_rank": rank,
                    "retrieval_score": float(retrieved_scores[row, rank - 1]),
                    "mining_version": config.mining_version,
                }
            )
            if len(hard_positions) == config.hard_negatives_per_candidate:
                break
        if len(hard_positions) != config.hard_negatives_per_candidate:
            raise ValueError(f"insufficient hard negatives for {candidate_id}")

        forbidden = true_positions | set(hard_positions)
        random_positions = _sample_random_skill_positions(
            rng,
            len(skill_ids),
            forbidden,
            config.random_negatives_per_candidate,
        )
        for position in random_positions:
            negative_rows.append(
                {
                    "candidate_id": candidate_id,
                    "skill_id": str(skill_ids[position]),
                    "split": config.negative_split,
                    "negative_type": "random",
                    "retrieval_rank": pd.NA,
                    "retrieval_score": np.nan,
                    "mining_version": config.mining_version,
                }
            )

    negative_pairs = pd.DataFrame(negative_rows)
    negative_pairs["retrieval_rank"] = negative_pairs["retrieval_rank"].astype(
        "Int64"
    )
    negative_pairs = negative_pairs.sort_values(
        ["candidate_id", "negative_type", "retrieval_rank", "skill_id"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    truth_keys = set(
        zip(positive_pairs["candidate_id"], positive_pairs["skill_id"], strict=True)
    )
    negative_keys = set(
        zip(negative_pairs["candidate_id"], negative_pairs["skill_id"], strict=True)
    )
    if truth_keys & negative_keys:
        raise ValueError("negative pairs overlap visible truth")
    if negative_pairs.duplicated(["candidate_id", "skill_id"]).any():
        raise ValueError("negative candidate-skill keys must be unique")

    mining_index = candidates[
        candidates["split"].isin(config.positive_splits)
    ][["candidate_id", "split"]].copy()
    mining_index["positive_count"] = (
        mining_index["candidate_id"]
        .map(positive_pairs.groupby("candidate_id").size())
        .fillna(0)
        .astype(int)
    )
    mining_index["negative_count"] = (
        mining_index["candidate_id"]
        .map(negative_pairs.groupby("candidate_id").size())
        .fillna(0)
        .astype(int)
    )
    mining_index["mining_version"] = config.mining_version
    mining_index = mining_index.reset_index(drop=True)

    return {
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "training_candidate_index": mining_index,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(
    embedding_dir: Path, truth_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """Load embedding arrays, row mappings, and candidate-skill truth."""
    paths = {
        "candidate_embeddings": embedding_dir / "candidate_embeddings.npy",
        "candidate_index": embedding_dir / "candidate_embedding_index.parquet",
        "skill_embeddings": embedding_dir / "skill_embeddings.npy",
        "skill_index": embedding_dir / "skill_embedding_index.parquet",
        "truth": truth_path,
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError("Missing pair-mining inputs:\n" + lines)
    arrays = {
        "candidate_embeddings": np.load(
            paths["candidate_embeddings"], allow_pickle=False
        ),
        "skill_embeddings": np.load(paths["skill_embeddings"], allow_pickle=False),
    }
    tables = {
        "candidate_index": pd.read_parquet(paths["candidate_index"]),
        "skill_index": pd.read_parquet(paths["skill_index"]),
        "truth": pd.read_parquet(paths["truth"]),
    }
    return arrays, tables


def write_pair_tables(
    tables: dict[str, pd.DataFrame],
    embedding_dir: Path,
    truth_path: Path,
    output_dir: Path,
    config: PairMiningConfig,
) -> dict[str, Any]:
    """Write training tables and a lineage manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "mining_version": config.mining_version,
        "config": asdict(config),
        "inputs": {},
        "outputs": {},
    }
    input_paths = (
        embedding_dir / "candidate_embeddings.npy",
        embedding_dir / "candidate_embedding_index.parquet",
        embedding_dir / "skill_embeddings.npy",
        embedding_dir / "skill_embedding_index.parquet",
        truth_path,
    )
    for path in input_paths:
        manifest["inputs"][path.name] = {"sha256": _sha256(path)}
    for name, table in tables.items():
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        manifest["outputs"][name] = {
            "path": path.name,
            "rows": len(table),
            "sha256": _sha256(path),
        }
    manifest_path = output_dir / "training_pairs_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--truth-path", type=Path, default=DEFAULT_TRUTH_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    arrays, inputs = load_inputs(args.embedding_dir, args.truth_path)
    tables = build_pair_tables(
        arrays["candidate_embeddings"],
        inputs["candidate_index"],
        arrays["skill_embeddings"],
        inputs["skill_index"],
        inputs["truth"],
        config,
    )
    write_pair_tables(
        tables,
        args.embedding_dir,
        args.truth_path,
        args.output_dir,
        config,
    )
    print(f"Training-pair snapshot: {args.output_dir.resolve()}")
    for name, table in tables.items():
        print(f"{name}: {len(table):,} rows")


if __name__ == "__main__":
    main()

