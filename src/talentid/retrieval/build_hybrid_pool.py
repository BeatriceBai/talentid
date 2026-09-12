"""Build a recall-preserving hybrid candidate pool for skill reranking."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "hybrid_retrieval.toml"
DEFAULT_LEARNED_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "learned_retrieval"
    / "v1"
    / "learned_retrieval_predictions.parquet"
)
DEFAULT_EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings" / "v1"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "v1"
DEFAULT_ONET_DIR = PROJECT_ROOT / "data" / "processed" / "onet" / "31.0"
DEFAULT_PAIR_PATH = (
    PROJECT_ROOT / "data" / "training" / "pairs" / "v1" / "positive_pairs.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "hybrid_retrieval" / "v1"


@dataclass(frozen=True)
class HybridRetrievalConfig:
    """Hybrid-pool sizes, channel weights, and evaluation settings."""

    retrieval_version: str = "talentskill-hybrid-retrieval-v1"
    split: str = "validation"
    learned_k: int = 100
    pool_size: int = 200
    tower_k: int = 200
    occupation_k: int = 200
    rrf_constant: float = 60.0
    tower_weight: float = 0.75
    occupation_weight: float = 0.65
    evaluation_cutoffs: tuple[int, ...] = (30, 100, 200)

    def validate(self) -> None:
        if not self.retrieval_version.strip():
            raise ValueError("retrieval_version must be nonempty")
        if self.split != "validation":
            raise ValueError("only validation is allowed while test labels are sealed")
        if not 1 <= self.learned_k < self.pool_size:
            raise ValueError("learned_k must be positive and smaller than pool_size")
        if self.tower_k < 1 or self.occupation_k < 1:
            raise ValueError("channel cutoffs must be positive")
        if self.rrf_constant <= 0:
            raise ValueError("rrf_constant must be positive")
        if self.tower_weight < 0 or self.occupation_weight < 0:
            raise ValueError("channel weights cannot be negative")
        if not self.evaluation_cutoffs:
            raise ValueError("evaluation_cutoffs cannot be empty")
        if (
            min(self.evaluation_cutoffs) < 1
            or max(self.evaluation_cutoffs) > self.pool_size
        ):
            raise ValueError("evaluation cutoffs must be within pool_size")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> HybridRetrievalConfig:
    """Load hybrid candidate-generation settings."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)["retrieval"]
    config = HybridRetrievalConfig(
        retrieval_version=str(raw["retrieval_version"]),
        split=str(raw["split"]),
        learned_k=int(raw["learned_k"]),
        pool_size=int(raw["pool_size"]),
        tower_k=int(raw["tower_k"]),
        occupation_k=int(raw["occupation_k"]),
        rrf_constant=float(raw["rrf_constant"]),
        tower_weight=float(raw["tower_weight"]),
        occupation_weight=float(raw["occupation_weight"]),
        evaluation_cutoffs=tuple(int(value) for value in raw["evaluation_cutoffs"]),
    )
    config.validate()
    return config


def _require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("embedding arrays must be finite matrices")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _top_k(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic descending top-k positions and scores per row."""
    if values.ndim != 2 or not 1 <= k <= values.shape[1]:
        raise ValueError("invalid top-k request")
    if k == values.shape[1]:
        positions = np.broadcast_to(np.arange(k), values.shape).copy()
    else:
        positions = np.argpartition(values, -k, axis=1)[:, -k:]
    scores = np.take_along_axis(values, positions, axis=1)
    order = np.empty_like(positions)
    for row in range(len(values)):
        order[row] = np.lexsort((positions[row], -scores[row]))
    positions = np.take_along_axis(positions, order, axis=1)
    return positions, np.take_along_axis(values, positions, axis=1)


def retrieve_tower_channels(
    tower_embeddings: np.ndarray,
    tower_index: pd.DataFrame,
    skill_embeddings: np.ndarray,
    skill_index: pd.DataFrame,
    candidate_ids: list[str],
    k: int,
    batch_size: int = 128,
) -> dict[str, list[tuple[str, int, float, float, str]]]:
    """Retrieve per-source candidates from every available evidence tower."""
    _require_columns(
        tower_index,
        {
            "embedding_row",
            "candidate_id",
            "source",
            "is_available",
            "reliability_score",
        },
        "tower index",
    )
    _require_columns(skill_index, {"embedding_row", "skill_id"}, "skill index")
    if len(tower_embeddings) != len(tower_index):
        raise ValueError("tower array and index lengths do not match")
    if len(skill_embeddings) != len(skill_index):
        raise ValueError("skill array and index lengths do not match")
    if tower_index.duplicated(["candidate_id", "source"]).any():
        raise ValueError("tower candidate-source keys must be unique")
    ordered_skills = skill_index.sort_values("embedding_row")
    if ordered_skills["embedding_row"].tolist() != list(range(len(skill_index))):
        raise ValueError("skill embedding rows must be contiguous")
    skill_ids = ordered_skills["skill_id"].astype(str).to_numpy()
    skills = _normalize_rows(skill_embeddings)
    towers = _normalize_rows(tower_embeddings)
    selected = tower_index[
        tower_index["candidate_id"].astype(str).isin(candidate_ids)
        & tower_index["is_available"].astype(bool)
    ].copy()
    selected["candidate_id"] = selected["candidate_id"].astype(str)
    output: dict[str, list[tuple[str, int, float, float]]] = {
        candidate_id: [] for candidate_id in candidate_ids
    }
    k = min(k, len(skills))
    for start in range(0, len(selected), batch_size):
        block = selected.iloc[start : start + batch_size]
        rows = block["embedding_row"].to_numpy(dtype=np.int64)
        similarities = towers[rows] @ skills.T
        positions, scores = _top_k(similarities, k)
        for local_row, (_, tower) in enumerate(block.iterrows()):
            reliability = float(tower["reliability_score"])
            source = str(tower["source"])
            for rank, skill_position in enumerate(positions[local_row], start=1):
                output[str(tower["candidate_id"])].append(
                    (
                        str(skill_ids[skill_position]),
                        rank,
                        float(scores[local_row, rank - 1]),
                        reliability,
                        source,
                    )
                )
    return output


def build_occupation_channels(
    candidate_index: pd.DataFrame,
    occupation_skills: pd.DataFrame,
    candidate_ids: list[str],
    k: int,
) -> dict[str, list[tuple[str, int, float]]]:
    """Rank O*NET occupation skills without using candidate-skill labels."""
    _require_columns(
        candidate_index,
        {"candidate_id", "primary_onet_soc_code", "split"},
        "candidate index",
    )
    _require_columns(
        occupation_skills,
        {
            "onet_soc_code",
            "skill_id",
            "importance",
            "level",
            "hot_technology",
            "in_demand",
            "not_relevant",
        },
        "occupation skills",
    )
    candidates = candidate_index.copy()
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("candidate ids must be unique")
    candidates = candidates.set_index("candidate_id")
    if not set(candidate_ids) <= set(candidates.index):
        raise ValueError("candidate index does not cover the retrieval candidates")
    skills = occupation_skills[~occupation_skills["not_relevant"].fillna(False)].copy()
    skills["importance"] = pd.to_numeric(skills["importance"], errors="coerce").fillna(
        0.0
    )
    skills["level"] = pd.to_numeric(skills["level"], errors="coerce").fillna(0.0)
    skills["occupation_score"] = (
        0.60 * (skills["importance"] / 5.0).clip(0.0, 1.0)
        + 0.30 * (skills["level"] / 7.0).clip(0.0, 1.0)
        + 0.06 * skills["hot_technology"].fillna(False).astype(float)
        + 0.04 * skills["in_demand"].fillna(False).astype(float)
    )
    grouped: dict[str, list[tuple[str, int, float]]] = {}
    for occupation, group in skills.groupby("onet_soc_code"):
        ranked = (
            group.sort_values(
                ["occupation_score", "skill_id"], ascending=[False, True], kind="stable"
            )
            .drop_duplicates("skill_id")
            .head(k)
        )
        grouped[str(occupation)] = [
            (str(row.skill_id), rank, float(row.occupation_score))
            for rank, row in enumerate(ranked.itertuples(index=False), start=1)
        ]
    return {
        candidate_id: grouped.get(
            str(candidates.loc[candidate_id, "primary_onet_soc_code"]), []
        )
        for candidate_id in candidate_ids
    }


def _learned_groups(
    learned_predictions: pd.DataFrame,
    config: HybridRetrievalConfig,
) -> tuple[list[str], dict[str, list[tuple[str, int, float]]]]:
    _require_columns(
        learned_predictions,
        {"candidate_id", "split", "engine", "rank", "skill_id", "score"},
        "learned predictions",
    )
    selected = learned_predictions[
        learned_predictions["split"].astype(str).eq(config.split)
        & learned_predictions["engine"].astype(str).eq("faiss_flat")
        & learned_predictions["rank"].le(config.learned_k)
    ].copy()
    selected["candidate_id"] = selected["candidate_id"].astype(str)
    selected["skill_id"] = selected["skill_id"].astype(str)
    if selected.duplicated(["candidate_id", "skill_id"]).any():
        raise ValueError("learned predictions contain duplicate candidate-skill rows")
    counts = selected.groupby("candidate_id").size()
    if counts.empty or not counts.eq(config.learned_k).all():
        raise ValueError(
            "learned predictions must contain exactly learned_k rows per candidate"
        )
    candidate_ids = sorted(counts.index.tolist())
    groups = {
        candidate_id: [
            (str(row.skill_id), int(row.rank), float(row.score))
            for row in group.sort_values("rank").itertuples(index=False)
        ]
        for candidate_id, group in selected.groupby("candidate_id")
    }
    return candidate_ids, groups


def _channel_features(
    features: dict[str, dict[str, Any]], skill_id: str
) -> dict[str, Any]:
    return features.setdefault(
        skill_id,
        {
            "rrf_score": 0.0,
            "learned_rank": None,
            "learned_score": None,
            "occupation_rank": None,
            "occupation_score": None,
            "best_tower_rank": None,
            "best_tower_score": None,
            "tower_hit_count": 0,
            "channel_count": 0,
        },
    )


def fuse_candidate_channels(
    learned: dict[str, list[tuple[str, int, float]]],
    towers: dict[str, list[tuple[str, int, float, float, str]]],
    occupations: dict[str, list[tuple[str, int, float]]],
    catalog_skill_ids: list[str],
    config: HybridRetrievalConfig,
) -> pd.DataFrame:
    """Preserve learned top-k and add RRF-ranked candidates from new channels."""
    rows: list[dict[str, Any]] = []
    for candidate_id in sorted(learned):
        features: dict[str, dict[str, Any]] = {}
        protected: list[str] = []
        for skill_id, rank, score in learned[candidate_id]:
            item = _channel_features(features, skill_id)
            item["rrf_score"] += 1.0 / (config.rrf_constant + rank)
            item["learned_rank"] = rank
            item["learned_score"] = score
            item["channel_count"] += 1
            protected.append(skill_id)
        for skill_id, rank, score, reliability, _source in towers[candidate_id]:
            item = _channel_features(features, skill_id)
            item["rrf_score"] += (
                config.tower_weight * reliability / (config.rrf_constant + rank)
            )
            item["tower_hit_count"] += 1
            item["channel_count"] += 1
            if item["best_tower_rank"] is None or rank < item["best_tower_rank"]:
                item["best_tower_rank"] = rank
                item["best_tower_score"] = score
        for skill_id, rank, score in occupations[candidate_id]:
            item = _channel_features(features, skill_id)
            item["rrf_score"] += config.occupation_weight / (config.rrf_constant + rank)
            item["occupation_rank"] = rank
            item["occupation_score"] = score
            item["channel_count"] += 1
        protected_set = set(protected)
        extras = sorted(
            (skill_id for skill_id in features if skill_id not in protected_set),
            key=lambda skill_id: (-features[skill_id]["rrf_score"], skill_id),
        )
        needed = config.pool_size - len(protected)
        chosen = protected + extras[:needed]
        chosen_set = set(chosen)
        fallback = [
            skill_id for skill_id in catalog_skill_ids if skill_id not in chosen_set
        ]
        chosen.extend(fallback[: config.pool_size - len(chosen)])
        if len(chosen) != config.pool_size:
            raise ValueError("catalog is smaller than the requested hybrid pool")
        for rank, skill_id in enumerate(chosen, start=1):
            item = _channel_features(features, skill_id)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "split": config.split,
                    "engine": "hybrid_rrf",
                    "rank": rank,
                    "skill_id": skill_id,
                    "score": item["rrf_score"],
                    **{key: value for key, value in item.items() if key != "rrf_score"},
                    "is_learned_protected": skill_id in protected_set,
                    "is_fallback": skill_id in fallback,
                }
            )
    return pd.DataFrame(rows)


def validate_visible_truth(pairs: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return visible validation truth and reject sealed test labels."""
    _require_columns(pairs, {"candidate_id", "skill_id", "split"}, "positive pairs")
    if "test" in set(pairs["split"].astype(str)):
        raise ValueError("positive pairs expose sealed test labels")
    truth = pairs[pairs["split"].astype(str).eq(split)].copy()
    truth["candidate_id"] = truth["candidate_id"].astype(str)
    truth["skill_id"] = truth["skill_id"].astype(str)
    if truth.empty:
        raise ValueError("positive pairs contain no validation truth")
    return truth.drop_duplicates(["candidate_id", "skill_id"])


def evaluate_pools(
    predictions: pd.DataFrame,
    learned_predictions: pd.DataFrame,
    truth: pd.DataFrame,
    catalog_size: int,
    config: HybridRetrievalConfig,
) -> pd.DataFrame:
    """Compare learned and hybrid pool recall and long-tail coverage."""
    truth_pairs = set(zip(truth["candidate_id"], truth["skill_id"], strict=True))
    truth_skills = set(truth["skill_id"])
    stages: list[tuple[str, pd.DataFrame]] = []
    learned = learned_predictions[
        learned_predictions["split"].astype(str).eq(config.split)
        & learned_predictions["engine"].astype(str).eq("faiss_flat")
    ].copy()
    for cutoff in config.evaluation_cutoffs:
        if cutoff <= config.learned_k:
            stages.append((f"learned_top{cutoff}", learned[learned["rank"].le(cutoff)]))
        stages.append(
            (f"hybrid_top{cutoff}", predictions[predictions["rank"].le(cutoff)])
        )
    rows = []
    for stage, table in stages:
        table_pairs = set(
            zip(
                table["candidate_id"].astype(str),
                table["skill_id"].astype(str),
                strict=True,
            )
        )
        skills = set(table["skill_id"].astype(str))
        candidate_hits = pd.DataFrame(
            table_pairs & truth_pairs, columns=["candidate_id", "skill_id"]
        )
        hit_counts = (
            candidate_hits.groupby("candidate_id").size()
            if not candidate_hits.empty
            else pd.Series(dtype=int)
        )
        truth_counts = truth.groupby("candidate_id").size()
        mean_recall = float(
            (hit_counts.reindex(truth_counts.index, fill_value=0) / truth_counts).mean()
        )
        rows.append(
            {
                "stage": stage,
                "rows": len(table),
                "unique_skills": len(skills),
                "catalog_coverage": len(skills) / catalog_size,
                "truth_skill_coverage": len(skills & truth_skills) / len(truth_skills),
                "truth_pair_recall": len(table_pairs & truth_pairs) / len(truth_pairs),
                "mean_candidate_recall": mean_recall,
            }
        )
    return pd.DataFrame(rows)


def build_hybrid_retrieval(
    learned_predictions: pd.DataFrame,
    tower_embeddings: np.ndarray,
    tower_index: pd.DataFrame,
    skill_embeddings: np.ndarray,
    skill_index: pd.DataFrame,
    candidate_index: pd.DataFrame,
    occupation_skills: pd.DataFrame,
    positive_pairs: pd.DataFrame,
    config: HybridRetrievalConfig,
) -> dict[str, pd.DataFrame]:
    """Build and evaluate a label-free hybrid pool on validation candidates."""
    config.validate()
    candidate_ids, learned = _learned_groups(learned_predictions, config)
    _require_columns(skill_index, {"skill_id"}, "skill index")
    catalog_skill_ids = sorted(skill_index["skill_id"].astype(str).tolist())
    if len(catalog_skill_ids) != len(set(catalog_skill_ids)):
        raise ValueError("skill ids must be unique")
    catalog_skill_set = set(catalog_skill_ids)
    learned_skill_set = {
        skill_id for rows in learned.values() for skill_id, _rank, _score in rows
    }
    if not learned_skill_set <= catalog_skill_set:
        raise ValueError("learned predictions contain unknown skills")
    _require_columns(occupation_skills, {"skill_id"}, "occupation skills")
    occupation_skill_set = set(occupation_skills["skill_id"].astype(str))
    if not occupation_skill_set <= catalog_skill_set:
        raise ValueError("occupation relationships contain unknown skills")
    towers = retrieve_tower_channels(
        tower_embeddings,
        tower_index,
        skill_embeddings,
        skill_index,
        candidate_ids,
        config.tower_k,
    )
    occupations = build_occupation_channels(
        candidate_index, occupation_skills, candidate_ids, config.occupation_k
    )
    predictions = fuse_candidate_channels(
        learned, towers, occupations, catalog_skill_ids, config
    )
    truth = validate_visible_truth(positive_pairs, config.split)
    if set(truth["candidate_id"]) != set(candidate_ids):
        raise ValueError("validation truth and retrieval candidates do not align")
    metrics = evaluate_pools(
        predictions, learned_predictions, truth, len(skill_index), config
    )
    return {
        "hybrid_retrieval_predictions": predictions,
        "hybrid_retrieval_metrics": metrics,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[Path]]:
    """Load hybrid retrieval inputs and record their lineage paths."""
    paths = {
        "learned_predictions": args.learned,
        "tower_embeddings": args.embedding_dir / "candidate_tower_embeddings.npy",
        "tower_index": args.embedding_dir / "candidate_tower_embedding_index.parquet",
        "skill_embeddings": args.embedding_dir / "skill_embeddings.npy",
        "skill_index": args.embedding_dir / "skill_embedding_index.parquet",
        "candidate_index": args.feature_dir / "candidate_index.parquet",
        "occupation_skills": args.onet_dir / "occupation_skills.parquet",
        "positive_pairs": args.positive_pairs,
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing hybrid-retrieval inputs:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    arrays = {
        name: np.load(path, allow_pickle=False)
        for name, path in paths.items()
        if path.suffix == ".npy"
    }
    tables = {
        name: pd.read_parquet(path)
        for name, path in paths.items()
        if path.suffix == ".parquet"
    }
    return {**arrays, **tables}, list(paths.values())


def write_results(
    results: dict[str, pd.DataFrame],
    input_paths: list[Path],
    output_dir: Path,
    config: HybridRetrievalConfig,
) -> None:
    """Persist hybrid pools, metrics, and reproducibility metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "retrieval_version": config.retrieval_version,
        "config": asdict(config),
        "test_labels_used": 0,
        "learned_top100_preserved": True,
        "inputs": {str(path): {"sha256": _sha256(path)} for path in input_paths},
        "outputs": {},
    }
    for name, table in results.items():
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        manifest["outputs"][name] = {
            "path": path.name,
            "rows": len(table),
            "sha256": _sha256(path),
        }
    (output_dir / "hybrid_retrieval_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned", type=Path, default=DEFAULT_LEARNED_PATH)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--onet-dir", type=Path, default=DEFAULT_ONET_DIR)
    parser.add_argument("--positive-pairs", type=Path, default=DEFAULT_PAIR_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    inputs, input_paths = load_inputs(args)
    results = build_hybrid_retrieval(**inputs, config=config)
    write_results(results, input_paths, args.output_dir, config)
    print(f"Hybrid retrieval snapshot: {args.output_dir.resolve()}")
    print(results["hybrid_retrieval_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
