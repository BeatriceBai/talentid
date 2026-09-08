"""Evaluate candidate-to-skill cosine retrieval against synthetic truth."""

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
DEFAULT_EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings" / "v1"
DEFAULT_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "candidates"
    / "v1"
    / "candidate_skill_truth.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "retrieval" / "v1"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval.toml"

SOURCES = (
    "resume",
    "projects",
    "job_history",
    "courses",
    "self_reported_skills",
)


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval strategies, held-out splits, and metric cutoffs."""

    evaluation_version: str = "talentskill-retrieval-v1"
    splits: tuple[str, ...] = ("validation",)
    strategies: tuple[str, ...] = ("reliability_weighted", "uniform_mean")
    cutoffs: tuple[int, ...] = (10, 30, 100)
    query_batch_size: int = 256

    def validate(self) -> None:
        valid_splits = {"train", "validation", "test"}
        valid_strategies = {"reliability_weighted", "uniform_mean"}
        if not self.evaluation_version.strip():
            raise ValueError("evaluation_version must be nonempty")
        if not self.splits or not set(self.splits) <= valid_splits:
            raise ValueError(f"splits must be selected from {sorted(valid_splits)}")
        if not self.strategies or not set(self.strategies) <= valid_strategies:
            raise ValueError("unknown retrieval strategy")
        if not self.cutoffs or any(value < 1 for value in self.cutoffs):
            raise ValueError("cutoffs must be positive")
        if tuple(sorted(set(self.cutoffs))) != self.cutoffs:
            raise ValueError("cutoffs must be unique and increasing")
        if self.query_batch_size < 1:
            raise ValueError("query_batch_size must be positive")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> RetrievalConfig:
    """Load retrieval settings from TOML."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)["retrieval"]
    config = RetrievalConfig(
        evaluation_version=str(raw["evaluation_version"]),
        splits=tuple(str(value) for value in raw["splits"]),
        strategies=tuple(str(value) for value in raw["strategies"]),
        cutoffs=tuple(int(value) for value in raw["cutoffs"]),
        query_batch_size=int(raw["query_batch_size"]),
    )
    config.validate()
    return config


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values),
        where=norms > 0,
    )


def retrieve_top_k(
    query_embeddings: np.ndarray,
    skill_embeddings: np.ndarray,
    max_k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic cosine top-k indices and scores in query batches."""
    queries = _l2_normalize(query_embeddings)
    skills = _l2_normalize(skill_embeddings)
    if queries.ndim != 2 or skills.ndim != 2:
        raise ValueError("query and skill embeddings must be matrices")
    if queries.shape[1] != skills.shape[1]:
        raise ValueError("query and skill dimensions must match")
    if not 1 <= max_k <= len(skills):
        raise ValueError("max_k must be between 1 and the skill count")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    top_indices = np.empty((len(queries), max_k), dtype=np.int64)
    top_scores = np.empty((len(queries), max_k), dtype=np.float32)
    for start in range(0, len(queries), batch_size):
        stop = min(start + batch_size, len(queries))
        scores = queries[start:stop] @ skills.T
        for offset, row_scores in enumerate(scores):
            selected = np.argpartition(-row_scores, max_k - 1)[:max_k]
            order = np.lexsort((selected, -row_scores[selected]))
            ranked = selected[order]
            top_indices[start + offset] = ranked
            top_scores[start + offset] = row_scores[ranked]
    return top_indices, top_scores


def build_uniform_candidate_embeddings(
    tower_embeddings: np.ndarray,
    tower_index: pd.DataFrame,
    candidate_ids: list[str],
) -> np.ndarray:
    """Average all available tower vectors without reliability weighting."""
    required = {"embedding_row", "candidate_id", "source", "is_available"}
    missing = required - set(tower_index.columns)
    if missing:
        raise ValueError(f"tower index is missing columns: {sorted(missing)}")
    if len(tower_embeddings) != len(tower_index):
        raise ValueError("tower array and index lengths do not match")
    if tower_index.duplicated(["candidate_id", "source"]).any():
        raise ValueError("tower index keys must be unique")

    row_lookup = tower_index.set_index(["candidate_id", "source"])[
        ["embedding_row", "is_available"]
    ]
    output = np.zeros(
        (len(candidate_ids), tower_embeddings.shape[1]), dtype=np.float32
    )
    for candidate_position, candidate_id in enumerate(candidate_ids):
        rows = []
        for source in SOURCES:
            key = (candidate_id, source)
            if key not in row_lookup.index:
                raise ValueError(f"missing tower index row: {key}")
            metadata = row_lookup.loc[key]
            if bool(metadata["is_available"]):
                rows.append(int(metadata["embedding_row"]))
        if rows:
            output[candidate_position] = tower_embeddings[rows].mean(axis=0)
    return _l2_normalize(output)


def _discounts(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))


def compute_candidate_metrics(
    retrieved_skill_ids: np.ndarray,
    truth: pd.DataFrame,
    candidate_ids: list[str],
    cutoffs: tuple[int, ...],
) -> pd.DataFrame:
    """Compute binary precision/recall/MRR and graded NDCG per candidate."""
    required = {"candidate_id", "skill_id", "relevance_score"}
    missing = required - set(truth.columns)
    if missing:
        raise ValueError(f"truth is missing columns: {sorted(missing)}")
    if retrieved_skill_ids.shape[0] != len(candidate_ids):
        raise ValueError("retrieval rows and candidate ids must align")
    if max(cutoffs) > retrieved_skill_ids.shape[1]:
        raise ValueError("retrieval output is shorter than a requested cutoff")

    grouped = {
        str(candidate_id): dict(
            zip(
                group["skill_id"].astype(str),
                group["relevance_score"].astype(float),
                strict=True,
            )
        )
        for candidate_id, group in truth.groupby("candidate_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for position, candidate_id in enumerate(candidate_ids):
        relevance = grouped.get(candidate_id, {})
        if not relevance:
            raise ValueError(f"candidate has no truth skills: {candidate_id}")
        ranked = retrieved_skill_ids[position]
        for k in cutoffs:
            top = ranked[:k]
            gains = np.asarray(
                [relevance.get(str(skill_id), 0.0) for skill_id in top],
                dtype=np.float64,
            )
            hits = np.asarray(
                [str(skill_id) in relevance for skill_id in top], dtype=bool
            )
            hit_count = int(hits.sum())
            recall = hit_count / len(relevance)
            precision = hit_count / k
            hit_positions = np.flatnonzero(hits)
            reciprocal_rank = (
                1.0 / float(hit_positions[0] + 1) if len(hit_positions) else 0.0
            )
            dcg = float(np.sum(gains * _discounts(k)))
            ideal_gains = np.asarray(
                sorted(relevance.values(), reverse=True)[:k], dtype=np.float64
            )
            idcg = float(np.sum(ideal_gains * _discounts(len(ideal_gains))))
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "k": k,
                    "truth_count": len(relevance),
                    "hit_count": hit_count,
                    "precision": precision,
                    "recall": recall,
                    "ndcg": dcg / idcg if idcg else 0.0,
                    "hit_rate": float(hit_count > 0),
                    "mrr": reciprocal_rank,
                }
            )
    return pd.DataFrame(rows)


def _predictions_table(
    candidate_ids: list[str],
    splits: list[str],
    skill_ids: np.ndarray,
    top_indices: np.ndarray,
    top_scores: np.ndarray,
    truth_pairs: set[tuple[str, str]],
    strategy: str,
) -> pd.DataFrame:
    rows = []
    for candidate_position, candidate_id in enumerate(candidate_ids):
        for rank, skill_position in enumerate(top_indices[candidate_position], 1):
            skill_id = str(skill_ids[skill_position])
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "split": splits[candidate_position],
                    "strategy": strategy,
                    "rank": rank,
                    "skill_id": skill_id,
                    "score": float(top_scores[candidate_position, rank - 1]),
                    "is_relevant": (candidate_id, skill_id) in truth_pairs,
                }
            )
    return pd.DataFrame(rows)


def _aggregate_metrics(
    candidate_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    catalog_size: int,
) -> pd.DataFrame:
    metric_columns = ["precision", "recall", "ndcg", "hit_rate", "mrr"]
    rows = []
    keys = ["strategy", "split", "k"]
    for key, group in candidate_metrics.groupby(keys, sort=True):
        strategy, split, k = key
        prediction_slice = predictions[
            predictions["strategy"].eq(strategy)
            & predictions["split"].eq(split)
            & predictions["rank"].le(k)
        ]
        row: dict[str, Any] = {
            "strategy": strategy,
            "split": split,
            "k": int(k),
            "candidates": group["candidate_id"].nunique(),
            "catalog_size": catalog_size,
            "catalog_coverage": prediction_slice["skill_id"].nunique()
            / catalog_size,
        }
        row.update({column: float(group[column].mean()) for column in metric_columns})
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage_bucket(count: int) -> str:
    if count <= 2:
        return "0-2"
    if count <= 4:
        return "3-4"
    return "5"


def evaluate_retrieval(
    candidate_embeddings: np.ndarray,
    candidate_index: pd.DataFrame,
    tower_embeddings: np.ndarray,
    tower_index: pd.DataFrame,
    skill_embeddings: np.ndarray,
    skill_index: pd.DataFrame,
    truth: pd.DataFrame,
    config: RetrievalConfig,
) -> dict[str, pd.DataFrame]:
    """Run configured retrieval strategies and return predictions and metrics."""
    config.validate()
    if len(candidate_embeddings) != len(candidate_index):
        raise ValueError("candidate array and index lengths do not match")
    if len(skill_embeddings) != len(skill_index):
        raise ValueError("skill array and index lengths do not match")
    if candidate_index["candidate_id"].duplicated().any():
        raise ValueError("candidate index ids must be unique")
    if skill_index["skill_id"].duplicated().any():
        raise ValueError("skill index ids must be unique")
    expected_candidate_rows = list(range(len(candidate_index)))
    expected_skill_rows = list(range(len(skill_index)))
    if sorted(candidate_index["embedding_row"].tolist()) != expected_candidate_rows:
        raise ValueError("candidate embedding rows must be a complete permutation")
    if sorted(skill_index["embedding_row"].tolist()) != expected_skill_rows:
        raise ValueError("skill embedding rows must be a complete permutation")

    skill_order = skill_index.sort_values("embedding_row").reset_index(drop=True)
    skill_rows = skill_order["embedding_row"].to_numpy(dtype=int)
    ordered_skill_embeddings = skill_embeddings[skill_rows]
    skill_ids = skill_order["skill_id"].astype(str).to_numpy()
    known_skills = set(skill_ids)
    unknown_truth = set(truth["skill_id"].astype(str)) - known_skills
    if unknown_truth:
        raise ValueError(f"truth contains {len(unknown_truth)} unknown skills")

    candidate_order = candidate_index.sort_values("embedding_row").reset_index(drop=True)
    candidate_rows = candidate_order["embedding_row"].to_numpy(dtype=int)
    ordered_candidate_embeddings = candidate_embeddings[candidate_rows]
    all_candidate_ids = candidate_order["candidate_id"].astype(str).tolist()
    uniform_embeddings = build_uniform_candidate_embeddings(
        tower_embeddings, tower_index, all_candidate_ids
    )
    strategy_vectors = {
        "reliability_weighted": _l2_normalize(ordered_candidate_embeddings),
        "uniform_mean": uniform_embeddings,
    }

    available_counts = (
        tower_index[tower_index["is_available"]]
        .groupby("candidate_id")
        .size()
        .to_dict()
    )
    truth = truth.copy()
    truth["candidate_id"] = truth["candidate_id"].astype(str)
    truth["skill_id"] = truth["skill_id"].astype(str)
    truth_pairs = set(zip(truth["candidate_id"], truth["skill_id"], strict=True))

    predictions = []
    candidate_metrics = []
    max_k = max(config.cutoffs)
    for split in config.splits:
        mask = candidate_order["split"].eq(split).to_numpy()
        positions = np.flatnonzero(mask)
        split_ids = [all_candidate_ids[position] for position in positions]
        split_labels = [split] * len(split_ids)
        split_truth = truth[truth["candidate_id"].isin(split_ids)]
        for strategy in config.strategies:
            top_indices, top_scores = retrieve_top_k(
                strategy_vectors[strategy][positions],
                ordered_skill_embeddings,
                max_k,
                config.query_batch_size,
            )
            prediction_table = _predictions_table(
                split_ids,
                split_labels,
                skill_ids,
                top_indices,
                top_scores,
                truth_pairs,
                strategy,
            )
            predictions.append(prediction_table)
            retrieved_ids = skill_ids[top_indices]
            metric_table = compute_candidate_metrics(
                retrieved_ids, split_truth, split_ids, config.cutoffs
            )
            metric_table["strategy"] = strategy
            metric_table["split"] = split
            metric_table["available_towers"] = metric_table["candidate_id"].map(
                available_counts
            ).fillna(0).astype(int)
            metric_table["coverage_bucket"] = metric_table["available_towers"].map(
                _coverage_bucket
            )
            candidate_metrics.append(metric_table)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    candidate_metric_frame = pd.concat(candidate_metrics, ignore_index=True)
    overall = _aggregate_metrics(
        candidate_metric_frame, prediction_frame, len(skill_ids)
    )
    by_coverage = (
        candidate_metric_frame.groupby(
            ["strategy", "split", "k", "coverage_bucket"], sort=True
        )
        .agg(
            candidates=("candidate_id", "nunique"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            ndcg=("ndcg", "mean"),
            hit_rate=("hit_rate", "mean"),
            mrr=("mrr", "mean"),
        )
        .reset_index()
    )
    return {
        "retrieval_predictions": prediction_frame,
        "retrieval_candidate_metrics": candidate_metric_frame,
        "retrieval_metrics": overall,
        "retrieval_metrics_by_coverage": by_coverage,
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
    """Load aligned embedding arrays, indices, and truth labels."""
    array_names = (
        "candidate_embeddings",
        "candidate_tower_embeddings",
        "skill_embeddings",
    )
    table_names = (
        "candidate_embedding_index",
        "candidate_tower_embedding_index",
        "skill_embedding_index",
    )
    paths = [embedding_dir / f"{name}.npy" for name in array_names]
    paths += [embedding_dir / f"{name}.parquet" for name in table_names]
    paths.append(truth_path)
    missing = [path for path in paths if not path.exists()]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError("Missing retrieval inputs:\n" + lines)
    arrays = {
        name: np.load(embedding_dir / f"{name}.npy", allow_pickle=False)
        for name in array_names
    }
    tables = {
        name: pd.read_parquet(embedding_dir / f"{name}.parquet")
        for name in table_names
    }
    tables["truth"] = pd.read_parquet(truth_path)
    return arrays, tables


def write_results(
    results: dict[str, pd.DataFrame],
    embedding_dir: Path,
    truth_path: Path,
    output_dir: Path,
    config: RetrievalConfig,
) -> dict[str, Any]:
    """Write evaluation tables and their reproducibility manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "evaluation_version": config.evaluation_version,
        "config": asdict(config),
        "inputs": {},
        "outputs": {},
    }
    input_paths = [
        embedding_dir / "candidate_embeddings.npy",
        embedding_dir / "candidate_tower_embeddings.npy",
        embedding_dir / "skill_embeddings.npy",
        embedding_dir / "candidate_embedding_index.parquet",
        embedding_dir / "candidate_tower_embedding_index.parquet",
        embedding_dir / "skill_embedding_index.parquet",
        truth_path,
    ]
    for path in input_paths:
        manifest["inputs"][path.name] = {"sha256": _sha256(path)}
    for name, table in results.items():
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        manifest["outputs"][name] = {
            "path": path.name,
            "rows": len(table),
            "sha256": _sha256(path),
        }
    manifest_path = output_dir / "retrieval_manifest.json"
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
    arrays, tables = load_inputs(args.embedding_dir, args.truth_path)
    results = evaluate_retrieval(
        arrays["candidate_embeddings"],
        tables["candidate_embedding_index"],
        arrays["candidate_tower_embeddings"],
        tables["candidate_tower_embedding_index"],
        arrays["skill_embeddings"],
        tables["skill_embedding_index"],
        tables["truth"],
        config,
    )
    write_results(
        results,
        args.embedding_dir,
        args.truth_path,
        args.output_dir,
        config,
    )
    print(f"Retrieval evaluation: {args.output_dir.resolve()}")
    metrics = results["retrieval_metrics"].copy()
    for column in ("precision", "recall", "ndcg", "hit_rate", "mrr"):
        metrics[column] = metrics[column].map(lambda value: f"{value:.4f}")
    metrics["catalog_coverage"] = metrics["catalog_coverage"].map(
        lambda value: f"{value:.2%}"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
