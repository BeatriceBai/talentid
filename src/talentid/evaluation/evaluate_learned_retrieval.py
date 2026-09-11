"""Evaluate learned embeddings with exact and approximate FAISS retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from talentid.evaluation.evaluate_retrieval import (
    compute_candidate_metrics,
    retrieve_top_k,
)
from talentid.retrieval.faiss_index import (
    build_index,
    neighbor_recall,
    normalize_embeddings,
    save_index,
    search_index,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "fusion" / "v1"
DEFAULT_PAIR_PATH = (
    PROJECT_ROOT / "data" / "training" / "pairs" / "v1" / "positive_pairs.parquet"
)
DEFAULT_BASELINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v1"
    / "retrieval_metrics.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "learned_retrieval" / "v1"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "skills" / "v1"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "faiss_retrieval.toml"
METRIC_COLUMNS = ("precision", "recall", "ndcg", "hit_rate", "mrr")


@dataclass(frozen=True)
class LearnedRetrievalConfig:
    """Exact/ANN retrieval settings and sealed evaluation split."""

    evaluation_version: str = "talentskill-learned-retrieval-v1"
    split: str = "validation"
    cutoffs: tuple[int, ...] = (10, 30, 100)
    query_batch_size: int = 256
    threads: int = 1
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 256
    minimum_recall_at_max_k: float = 0.99

    def validate(self) -> None:
        if not self.evaluation_version.strip():
            raise ValueError("evaluation_version must be nonempty")
        if self.split != "validation":
            raise ValueError("only validation is allowed while test labels are sealed")
        if tuple(sorted(set(self.cutoffs))) != self.cutoffs or any(
            cutoff < 1 for cutoff in self.cutoffs
        ):
            raise ValueError("cutoffs must be positive, unique, and increasing")
        if (
            min(
                self.query_batch_size,
                self.threads,
                self.hnsw_m,
                self.ef_construction,
                self.ef_search,
            )
            < 1
        ):
            raise ValueError("retrieval counts must be positive")
        if not 0 < self.minimum_recall_at_max_k <= 1:
            raise ValueError("minimum ANN recall must be in (0, 1]")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> LearnedRetrievalConfig:
    """Load retrieval and HNSW settings from TOML."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    retrieval = raw["retrieval"]
    hnsw = raw["hnsw"]
    config = LearnedRetrievalConfig(
        evaluation_version=str(retrieval["evaluation_version"]),
        split=str(retrieval["split"]),
        cutoffs=tuple(int(value) for value in retrieval["cutoffs"]),
        query_batch_size=int(retrieval["query_batch_size"]),
        threads=int(retrieval["threads"]),
        hnsw_m=int(hnsw["m"]),
        ef_construction=int(hnsw["ef_construction"]),
        ef_search=int(hnsw["ef_search"]),
        minimum_recall_at_max_k=float(hnsw["minimum_recall_at_max_k"]),
    )
    config.validate()
    return config


def _require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _ordered_embeddings(
    embeddings: np.ndarray,
    index: pd.DataFrame,
    *,
    id_column: str,
    name: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    _require_columns(index, {"embedding_row", id_column}, name)
    if len(embeddings) != len(index):
        raise ValueError(f"{name} array and index lengths do not match")
    if index[id_column].astype(str).duplicated().any():
        raise ValueError(f"{name} ids must be unique")
    if sorted(index["embedding_row"].tolist()) != list(range(len(index))):
        raise ValueError(f"{name} rows must be a complete permutation")
    ordered = index.sort_values("embedding_row").reset_index(drop=True)
    rows = ordered["embedding_row"].to_numpy(dtype=int)
    return normalize_embeddings(embeddings[rows]), ordered


def _metrics_table(
    engine: str,
    indices: np.ndarray,
    scores: np.ndarray,
    candidate_ids: list[str],
    skill_ids: np.ndarray,
    truth: pd.DataFrame,
    config: LearnedRetrievalConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    retrieved = skill_ids[indices]
    candidate_metrics = compute_candidate_metrics(
        retrieved, truth, candidate_ids, config.cutoffs
    )
    candidate_metrics.insert(1, "split", config.split)
    candidate_metrics.insert(2, "engine", engine)
    rows = []
    prediction_rows = []
    truth_pairs = set(zip(truth["candidate_id"], truth["skill_id"], strict=True))
    for candidate_position, candidate_id in enumerate(candidate_ids):
        for rank, skill_position in enumerate(indices[candidate_position], start=1):
            skill_id = str(skill_ids[skill_position])
            prediction_rows.append(
                {
                    "candidate_id": candidate_id,
                    "split": config.split,
                    "engine": engine,
                    "rank": rank,
                    "skill_id": skill_id,
                    "score": float(scores[candidate_position, rank - 1]),
                    "is_relevant": (candidate_id, skill_id) in truth_pairs,
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    for cutoff, group in candidate_metrics.groupby("k", sort=True):
        used = predictions[predictions["rank"].le(cutoff)]
        row: dict[str, Any] = {
            "engine": engine,
            "split": config.split,
            "k": int(cutoff),
            "candidates": len(candidate_ids),
            "catalog_size": len(skill_ids),
            "catalog_coverage": used["skill_id"].nunique() / len(skill_ids),
        }
        row.update({metric: float(group[metric].mean()) for metric in METRIC_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows), predictions


def _baseline_comparison(
    learned_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        baseline_metrics,
        {"strategy", "split", "k", "catalog_coverage", *METRIC_COLUMNS},
        "baseline metrics",
    )
    baseline = baseline_metrics[
        baseline_metrics["strategy"].eq("reliability_weighted")
        & baseline_metrics["split"].eq("validation")
    ].set_index("k")
    learned = learned_metrics[learned_metrics["engine"].eq("faiss_flat")].set_index("k")
    rows = []
    for cutoff in sorted(set(baseline.index).intersection(learned.index)):
        for metric in (*METRIC_COLUMNS, "catalog_coverage"):
            old = float(baseline.loc[cutoff, metric])
            new = float(learned.loc[cutoff, metric])
            rows.append(
                {
                    "k": int(cutoff),
                    "metric": metric,
                    "frozen_baseline": old,
                    "learned_model": new,
                    "absolute_lift": new - old,
                    "relative_lift": (new / old - 1) if old else np.nan,
                }
            )
    if not rows:
        raise ValueError("baseline and learned metrics have no common cutoffs")
    return pd.DataFrame(rows)


def evaluate_learned_retrieval(
    candidate_embeddings: np.ndarray,
    candidate_index: pd.DataFrame,
    skill_embeddings: np.ndarray,
    skill_index: pd.DataFrame,
    positive_pairs: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    config: LearnedRetrievalConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Benchmark exact/HNSW search and evaluate validation ranking quality."""
    config.validate()
    _require_columns(
        candidate_index, {"embedding_row", "candidate_id", "split"}, "candidate index"
    )
    _require_columns(
        positive_pairs,
        {"candidate_id", "skill_id", "split", "relevance_score"},
        "positive pairs",
    )
    if "test" in set(positive_pairs["split"].astype(str)):
        raise ValueError("positive pairs expose sealed test labels")
    candidates, candidate_order = _ordered_embeddings(
        candidate_embeddings,
        candidate_index,
        id_column="candidate_id",
        name="candidate",
    )
    skills, skill_order = _ordered_embeddings(
        skill_embeddings, skill_index, id_column="skill_id", name="skill"
    )
    max_k = max(config.cutoffs)
    if max_k > len(skills):
        raise ValueError("maximum cutoff exceeds the skill catalog")
    validation_mask = candidate_order["split"].eq(config.split).to_numpy()
    validation_positions = np.flatnonzero(validation_mask)
    if not len(validation_positions):
        raise ValueError("candidate index contains no validation rows")
    candidate_ids = (
        candidate_order.loc[validation_mask, "candidate_id"].astype(str).tolist()
    )
    queries = candidates[validation_positions]
    truth = positive_pairs[positive_pairs["split"].eq(config.split)].copy()
    truth["candidate_id"] = truth["candidate_id"].astype(str)
    truth["skill_id"] = truth["skill_id"].astype(str)
    if set(truth["candidate_id"]) != set(candidate_ids):
        raise ValueError("validation truth and candidate index ids do not align")
    skill_ids = skill_order["skill_id"].astype(str).to_numpy()
    if not set(truth["skill_id"]) <= set(skill_ids):
        raise ValueError("validation truth contains unknown skill ids")

    started = time.perf_counter()
    numpy_indices, _numpy_scores = retrieve_top_k(
        queries, skills, max_k, config.query_batch_size
    )
    numpy_seconds = time.perf_counter() - started

    benchmark_rows = []
    metric_frames = []
    prediction_frames = []
    searched: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    indexes: dict[str, Any] = {}
    for engine, index_type in (("faiss_flat", "flat"), ("faiss_hnsw", "hnsw")):
        started = time.perf_counter()
        index = build_index(
            skills,
            index_type=index_type,
            hnsw_m=config.hnsw_m,
            ef_construction=config.ef_construction,
            ef_search=config.ef_search,
            threads=config.threads,
        )
        build_seconds = time.perf_counter() - started
        started = time.perf_counter()
        indices, scores = search_index(index, queries, max_k)
        search_seconds = time.perf_counter() - started
        indexes[engine] = index
        searched[engine] = (indices, scores)
        metrics, predictions = _metrics_table(
            engine, indices, scores, candidate_ids, skill_ids, truth, config
        )
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
        benchmark_rows.append(
            {
                "engine": engine,
                "index_build_seconds": build_seconds,
                "search_seconds": search_seconds,
                "queries": len(queries),
                "queries_per_second": len(queries) / search_seconds,
            }
        )

    fidelity_rows = []
    flat_indices = searched["faiss_flat"][0]
    hnsw_indices = searched["faiss_hnsw"][0]
    for cutoff in config.cutoffs:
        fidelity_rows.extend(
            [
                {
                    "engine": "faiss_flat",
                    "reference": "numpy_exact",
                    "k": cutoff,
                    "neighbor_recall": neighbor_recall(
                        numpy_indices[:, :cutoff], flat_indices[:, :cutoff]
                    ),
                },
                {
                    "engine": "faiss_hnsw",
                    "reference": "faiss_flat",
                    "k": cutoff,
                    "neighbor_recall": neighbor_recall(
                        flat_indices[:, :cutoff], hnsw_indices[:, :cutoff]
                    ),
                },
            ]
        )
    fidelity = pd.DataFrame(fidelity_rows)
    hnsw_at_max = fidelity[
        fidelity["engine"].eq("faiss_hnsw") & fidelity["k"].eq(max_k)
    ]["neighbor_recall"].iloc[0]
    if hnsw_at_max < config.minimum_recall_at_max_k:
        raise RuntimeError(
            f"HNSW recall@{max_k}={hnsw_at_max:.4f} is below "
            f"{config.minimum_recall_at_max_k:.4f}"
        )
    learned_metrics = pd.concat(metric_frames, ignore_index=True)
    results = {
        "learned_retrieval_metrics": learned_metrics,
        "learned_retrieval_predictions": pd.concat(
            prediction_frames, ignore_index=True
        ),
        "ann_fidelity": fidelity,
        "search_benchmark": pd.DataFrame(
            [
                {
                    "engine": "numpy_exact",
                    "index_build_seconds": 0.0,
                    "search_seconds": numpy_seconds,
                    "queries": len(queries),
                    "queries_per_second": len(queries) / numpy_seconds,
                },
                *benchmark_rows,
            ]
        ),
        "baseline_comparison": _baseline_comparison(learned_metrics, baseline_metrics),
    }
    return results, indexes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(
    model_dir: Path,
    pair_path: Path,
    baseline_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame], list[Path]]:
    """Load learned embeddings and validation-only labels."""
    paths = [
        model_dir / "learned_candidate_embeddings.npy",
        model_dir / "candidate_embedding_index.parquet",
        model_dir / "learned_skill_embeddings.npy",
        model_dir / "skill_embedding_index.parquet",
        pair_path,
        baseline_path,
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing learned-retrieval inputs:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    arrays = {
        "candidate_embeddings": np.load(paths[0], allow_pickle=False),
        "skill_embeddings": np.load(paths[2], allow_pickle=False),
    }
    tables = {
        "candidate_index": pd.read_parquet(paths[1]),
        "skill_index": pd.read_parquet(paths[3]),
        "positive_pairs": pd.read_parquet(pair_path),
        "baseline_metrics": pd.read_parquet(baseline_path),
    }
    return arrays, tables, paths


def write_results(
    results: dict[str, pd.DataFrame],
    indexes: dict[str, Any],
    input_paths: list[Path],
    output_dir: Path,
    index_dir: Path,
    config: LearnedRetrievalConfig,
) -> None:
    """Persist evaluation tables, serving index, and reproducibility metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    index_paths = {
        "flat_index": index_dir / "flat.faiss",
        "hnsw_index": index_dir / "hnsw.faiss",
    }
    save_index(indexes["faiss_flat"], index_paths["flat_index"])
    save_index(indexes["faiss_hnsw"], index_paths["hnsw_index"])
    manifest: dict[str, Any] = {
        "evaluation_version": config.evaluation_version,
        "config": asdict(config),
        "test_labels_used": 0,
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
    for name, path in index_paths.items():
        engine = f"faiss_{name.removesuffix('_index')}"
        manifest["outputs"][name] = {
            "path": str(path),
            "vectors": int(indexes[engine].ntotal),
            "sha256": _sha256(path),
        }
    (output_dir / "learned_retrieval_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--positive-pairs", type=Path, default=DEFAULT_PAIR_PATH)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    arrays, tables, input_paths = load_inputs(
        args.model_dir, args.positive_pairs, args.baseline
    )
    results, indexes = evaluate_learned_retrieval(
        arrays["candidate_embeddings"],
        tables["candidate_index"],
        arrays["skill_embeddings"],
        tables["skill_index"],
        tables["positive_pairs"],
        tables["baseline_metrics"],
        config,
    )
    write_results(
        results,
        indexes,
        input_paths,
        args.output_dir,
        args.index_dir,
        config,
    )
    print(f"Learned retrieval snapshot: {args.output_dir.resolve()}")
    print(results["learned_retrieval_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
