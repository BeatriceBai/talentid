"""Tune a validation-only reranker for evidence, popularity, and diversity."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from talentid.evaluation.evaluate_retrieval import compute_candidate_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "reranker.toml"
DEFAULT_RETRIEVAL_DIR = (
    PROJECT_ROOT / "data" / "evaluation" / "learned_retrieval" / "v1"
)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "fusion" / "v1"
DEFAULT_EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings" / "v1"
DEFAULT_PAIR_PATH = (
    PROJECT_ROOT / "data" / "training" / "pairs" / "v1" / "positive_pairs.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "reranking" / "v1"
SOURCES = (
    "resume",
    "projects",
    "job_history",
    "courses",
    "self_reported_skills",
)
METRICS = ("precision", "recall", "ndcg", "hit_rate", "mrr")


@dataclass(frozen=True)
class RerankerConfig:
    """Reranking constraints and hyperparameter grid."""

    reranker_version: str = "talentskill-reranker-v1"
    split: str = "validation"
    candidate_pool_size: int = 100
    output_k: int = 30
    maximum_ndcg_drop: float = 0.02
    evidence_weights: tuple[float, ...] = (0.15, 0.30)
    popularity_penalties: tuple[float, ...] = (0.05, 0.10)
    diversity_penalties: tuple[float, ...] = (0.00, 0.03, 0.06)

    def validate(self) -> None:
        if not self.reranker_version.strip():
            raise ValueError("reranker_version must be nonempty")
        if self.split != "validation":
            raise ValueError("only validation is allowed while test labels are sealed")
        if not 1 <= self.output_k <= self.candidate_pool_size:
            raise ValueError("output_k must be within the candidate pool")
        if self.maximum_ndcg_drop < 0:
            raise ValueError("maximum_ndcg_drop cannot be negative")
        grids = (
            self.evidence_weights,
            self.popularity_penalties,
            self.diversity_penalties,
        )
        if any(not values for values in grids):
            raise ValueError("reranking grids must be nonempty")
        if any(not 0 <= value <= 1 for values in grids for value in values):
            raise ValueError("reranking weights must be in [0, 1]")


@dataclass(frozen=True)
class CandidatePools:
    """Aligned validation candidate pools and reranking features."""

    candidate_ids: list[str]
    skill_ids: np.ndarray
    skill_positions: np.ndarray
    retrieval_scores: np.ndarray
    evidence_scores: np.ndarray
    popularity_scores: np.ndarray
    pool_similarities: np.ndarray


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> RerankerConfig:
    """Load reranking constraints and search grid."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    reranking = raw["reranking"]
    search = raw["search"]
    config = RerankerConfig(
        reranker_version=str(reranking["reranker_version"]),
        split=str(reranking["split"]),
        candidate_pool_size=int(reranking["candidate_pool_size"]),
        output_k=int(reranking["output_k"]),
        maximum_ndcg_drop=float(reranking["maximum_ndcg_drop"]),
        evidence_weights=tuple(float(value) for value in search["evidence_weights"]),
        popularity_penalties=tuple(
            float(value) for value in search["popularity_penalties"]
        ),
        diversity_penalties=tuple(
            float(value) for value in search["diversity_penalties"]
        ),
    )
    config.validate()
    return config


def _require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _normalize_allow_zeros(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("embeddings must be a finite matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _minmax_rows(values: np.ndarray) -> np.ndarray:
    minimum = values.min(axis=1, keepdims=True)
    scale = values.max(axis=1, keepdims=True) - minimum
    return np.divide(
        values - minimum,
        scale,
        out=np.zeros_like(values, dtype=np.float32),
        where=scale > 0,
    )


def validate_visible_pairs(pairs: pd.DataFrame) -> None:
    """Reject any label artifact that exposes the sealed test split."""
    _require_columns(
        pairs,
        {"candidate_id", "skill_id", "split", "relevance_score"},
        "positive pairs",
    )
    if "test" in set(pairs["split"].astype(str)):
        raise ValueError("positive pairs expose sealed test labels")
    if not {"train", "validation"}.issuperset(set(pairs["split"].astype(str))):
        raise ValueError("positive pairs contain an unknown split")


def _prediction_arrays(
    predictions: pd.DataFrame,
    candidate_ids: list[str],
    skill_lookup: dict[str, int],
    config: RerankerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    _require_columns(
        predictions,
        {"candidate_id", "split", "engine", "rank", "skill_id", "score"},
        "retrieval predictions",
    )
    selected = predictions[
        predictions["engine"].eq("faiss_flat")
        & predictions["split"].eq(config.split)
        & predictions["rank"].le(config.candidate_pool_size)
    ].copy()
    grouped = {str(key): group for key, group in selected.groupby("candidate_id")}
    positions = np.empty(
        (len(candidate_ids), config.candidate_pool_size), dtype=np.int64
    )
    scores = np.empty_like(positions, dtype=np.float32)
    expected_ranks = list(range(1, config.candidate_pool_size + 1))
    for row, candidate_id in enumerate(candidate_ids):
        if candidate_id not in grouped:
            raise ValueError(f"missing retrieval pool for {candidate_id}")
        group = grouped[candidate_id].sort_values("rank")
        if group["rank"].tolist() != expected_ranks:
            raise ValueError(f"retrieval ranks are incomplete for {candidate_id}")
        try:
            positions[row] = [skill_lookup[str(value)] for value in group["skill_id"]]
        except KeyError as error:
            raise ValueError("retrieval predictions contain unknown skills") from error
        scores[row] = group["score"].to_numpy(dtype=np.float32)
    return positions, scores


def _candidate_evidence(
    candidate_ids: list[str],
    skill_positions: np.ndarray,
    aligned_skill_embeddings: np.ndarray,
    tower_embeddings: np.ndarray,
    tower_index: pd.DataFrame,
    gates: pd.DataFrame,
) -> np.ndarray:
    _require_columns(
        tower_index,
        {"embedding_row", "candidate_id", "source", "is_available"},
        "tower index",
    )
    _require_columns(
        gates,
        {"candidate_id", "source", "gate_weight"},
        "gate weights",
    )
    if len(tower_embeddings) != len(tower_index):
        raise ValueError("tower array and index lengths do not match")
    if tower_index.duplicated(["candidate_id", "source"]).any():
        raise ValueError("tower index keys must be unique")
    if gates.duplicated(["candidate_id", "source"]).any():
        raise ValueError("gate weight keys must be unique")
    if tower_embeddings.shape[1] != aligned_skill_embeddings.shape[1]:
        raise ValueError("tower and original skill dimensions do not match")
    tower_vectors = _normalize_allow_zeros(tower_embeddings)
    tower_lookup = tower_index.set_index(["candidate_id", "source"])
    gate_lookup = gates.set_index(["candidate_id", "source"])
    dense_towers = np.zeros(
        (len(candidate_ids), len(SOURCES), tower_vectors.shape[1]), dtype=np.float32
    )
    dense_gates = np.zeros((len(candidate_ids), len(SOURCES)), dtype=np.float32)
    for candidate_row, candidate_id in enumerate(candidate_ids):
        for source_row, source in enumerate(SOURCES):
            key = (candidate_id, source)
            if key not in tower_lookup.index or key not in gate_lookup.index:
                raise ValueError(f"missing tower or gate row: {key}")
            tower = tower_lookup.loc[key]
            gate = gate_lookup.loc[key]
            if bool(tower["is_available"]):
                dense_towers[candidate_row, source_row] = tower_vectors[
                    int(tower["embedding_row"])
                ]
                dense_gates[candidate_row, source_row] = float(gate["gate_weight"])
        if not np.isclose(dense_gates[candidate_row].sum(), 1.0, atol=1e-5):
            raise ValueError(f"gate weights do not sum to one for {candidate_id}")
    evidence = np.empty(skill_positions.shape, dtype=np.float32)
    batch_size = 256
    for start in range(0, len(candidate_ids), batch_size):
        stop = min(start + batch_size, len(candidate_ids))
        pool_skills = aligned_skill_embeddings[skill_positions[start:stop]]
        tower_scores = np.einsum(
            "btd,bkd->btk",
            dense_towers[start:stop],
            pool_skills,
            optimize=True,
        )
        evidence[start:stop] = np.einsum(
            "bt,btk->bk", dense_gates[start:stop], tower_scores, optimize=True
        )
    return evidence


def assemble_candidate_pools(
    predictions: pd.DataFrame,
    candidate_index: pd.DataFrame,
    learned_skill_embeddings: np.ndarray,
    learned_skill_index: pd.DataFrame,
    original_skill_embeddings: np.ndarray,
    original_skill_index: pd.DataFrame,
    tower_embeddings: np.ndarray,
    tower_index: pd.DataFrame,
    gates: pd.DataFrame,
    positive_pairs: pd.DataFrame,
    config: RerankerConfig,
) -> tuple[CandidatePools, pd.DataFrame]:
    """Align retrieval pools, evidence, train popularity, and skill similarity."""
    config.validate()
    validate_visible_pairs(positive_pairs)
    _require_columns(
        candidate_index,
        {"embedding_row", "candidate_id", "split"},
        "candidate index",
    )
    _require_columns(
        learned_skill_index,
        {"embedding_row", "skill_id"},
        "learned skill index",
    )
    _require_columns(
        original_skill_index,
        {"embedding_row", "skill_id"},
        "original skill index",
    )
    candidates = candidate_index.sort_values("embedding_row")
    if candidate_index["candidate_id"].astype(str).duplicated().any():
        raise ValueError("candidate ids must be unique")
    candidate_ids = (
        candidates.loc[candidates["split"].eq(config.split), "candidate_id"]
        .astype(str)
        .tolist()
    )
    if not candidate_ids:
        raise ValueError("candidate index contains no validation candidates")
    learned_index = learned_skill_index.sort_values("embedding_row").reset_index(
        drop=True
    )
    if len(learned_skill_embeddings) != len(learned_index):
        raise ValueError("learned skill array and index lengths do not match")
    if sorted(learned_index["embedding_row"].tolist()) != list(
        range(len(learned_index))
    ):
        raise ValueError("learned skill rows must be a complete permutation")
    if learned_index["skill_id"].astype(str).duplicated().any():
        raise ValueError("learned skill ids must be unique")
    skill_ids = learned_index["skill_id"].astype(str).to_numpy()
    skill_lookup = {skill_id: row for row, skill_id in enumerate(skill_ids)}
    learned_skills = _normalize_allow_zeros(
        learned_skill_embeddings[
            learned_index["embedding_row"].to_numpy(dtype=np.int64)
        ]
    )
    original_lookup = original_skill_index.set_index(
        original_skill_index["skill_id"].astype(str)
    )["embedding_row"]
    if len(original_skill_embeddings) != len(original_skill_index):
        raise ValueError("original skill array and index lengths do not match")
    if original_skill_index["skill_id"].astype(str).duplicated().any():
        raise ValueError("original skill ids must be unique")
    if not set(skill_ids) <= set(original_lookup.index):
        raise ValueError("learned and original skill catalogs do not align")
    original_rows = np.asarray(
        [int(original_lookup.loc[skill_id]) for skill_id in skill_ids], dtype=np.int64
    )
    original_skills = _normalize_allow_zeros(original_skill_embeddings[original_rows])
    skill_positions, retrieval_scores = _prediction_arrays(
        predictions, candidate_ids, skill_lookup, config
    )
    evidence_scores = _candidate_evidence(
        candidate_ids,
        skill_positions,
        original_skills,
        tower_embeddings,
        tower_index,
        gates,
    )
    train_counts = (
        positive_pairs[positive_pairs["split"].eq("train")].groupby("skill_id").size()
    )
    counts = np.asarray(
        [float(train_counts.get(skill_id, 0)) for skill_id in skill_ids],
        dtype=np.float32,
    )
    popularity = np.log1p(counts)
    if popularity.max() > 0:
        popularity /= popularity.max()
    pool_vectors = learned_skills[skill_positions]
    pool_similarities = np.einsum(
        "bkd,bjd->bkj", pool_vectors, pool_vectors, optimize=True
    ).astype(np.float32)
    truth = positive_pairs[positive_pairs["split"].eq(config.split)].copy()
    truth["candidate_id"] = truth["candidate_id"].astype(str)
    truth["skill_id"] = truth["skill_id"].astype(str)
    if set(truth["candidate_id"]) != set(candidate_ids):
        raise ValueError("validation truth does not align with candidate index")
    return (
        CandidatePools(
            candidate_ids=candidate_ids,
            skill_ids=skill_ids,
            skill_positions=skill_positions,
            retrieval_scores=retrieval_scores,
            evidence_scores=evidence_scores,
            popularity_scores=popularity[skill_positions],
            pool_similarities=pool_similarities,
        ),
        truth,
    )


def rerank_pools(
    pools: CandidatePools,
    output_k: int,
    evidence_weight: float,
    popularity_penalty: float,
    diversity_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily rerank each candidate pool with deterministic tie-breaking."""
    if not 1 <= output_k <= pools.skill_positions.shape[1]:
        raise ValueError("output_k must be within the candidate pool")
    base = _minmax_rows(pools.retrieval_scores)
    evidence = _minmax_rows(pools.evidence_scores)
    static_scores = (
        (1.0 - evidence_weight) * base
        + evidence_weight * evidence
        - popularity_penalty * pools.popularity_scores
    )
    selected = np.empty((len(pools.candidate_ids), output_k), dtype=np.int64)
    selection_scores = np.empty_like(selected, dtype=np.float32)
    for candidate_row in range(len(pools.candidate_ids)):
        available = np.ones(pools.skill_positions.shape[1], dtype=bool)
        maximum_similarity = np.zeros(pools.skill_positions.shape[1], dtype=np.float32)
        for rank in range(output_k):
            scores = static_scores[candidate_row] - diversity_penalty * np.maximum(
                maximum_similarity, 0.0
            )
            scores[~available] = -np.inf
            best = int(np.flatnonzero(scores == scores.max())[0])
            selected[candidate_row, rank] = best
            selection_scores[candidate_row, rank] = scores[best]
            available[best] = False
            maximum_similarity = np.maximum(
                maximum_similarity, pools.pool_similarities[candidate_row, best]
            )
    return selected, selection_scores


def _summarize_ranking(
    pools: CandidatePools,
    selected: np.ndarray,
    truth: pd.DataFrame,
    config: RerankerConfig,
) -> dict[str, float]:
    skill_positions = np.take_along_axis(pools.skill_positions, selected, axis=1)
    retrieved_ids = pools.skill_ids[skill_positions]
    candidate_metrics = compute_candidate_metrics(
        retrieved_ids, truth, pools.candidate_ids, (config.output_k,)
    )
    summary = {metric: float(candidate_metrics[metric].mean()) for metric in METRICS}
    summary["catalog_coverage"] = len(np.unique(skill_positions)) / len(pools.skill_ids)
    selected_popularity = np.take_along_axis(pools.popularity_scores, selected, axis=1)
    summary["mean_popularity"] = float(selected_popularity.mean())
    redundancies = []
    triangle = np.triu_indices(config.output_k, k=1)
    for candidate_row, positions in enumerate(selected):
        matrix = pools.pool_similarities[candidate_row][np.ix_(positions, positions)]
        redundancies.append(float(matrix[triangle].mean()))
    summary["mean_intra_list_similarity"] = float(np.mean(redundancies))
    return summary


def tune_reranker(
    pools: CandidatePools,
    truth: pd.DataFrame,
    config: RerankerConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Select maximum coverage within the configured NDCG degradation budget."""
    configs = [(0.0, 0.0, 0.0)]
    configs.extend(
        itertools.product(
            config.evidence_weights,
            config.popularity_penalties,
            config.diversity_penalties,
        )
    )
    rows = []
    selections: list[tuple[np.ndarray, np.ndarray]] = []
    for trial, (evidence, popularity, diversity) in enumerate(configs):
        selected, scores = rerank_pools(
            pools, config.output_k, evidence, popularity, diversity
        )
        summary = _summarize_ranking(pools, selected, truth, config)
        rows.append(
            {
                "trial": trial,
                "evidence_weight": evidence,
                "popularity_penalty": popularity,
                "diversity_penalty": diversity,
                **summary,
            }
        )
        selections.append((selected, scores))
    trials = pd.DataFrame(rows)
    baseline_ndcg = float(trials.loc[trials["trial"].eq(0), "ndcg"].iloc[0])
    trials["ndcg_floor"] = baseline_ndcg - config.maximum_ndcg_drop
    trials["meets_ndcg_constraint"] = trials["ndcg"].ge(trials["ndcg_floor"])
    eligible = trials[trials["meets_ndcg_constraint"]].sort_values(
        [
            "catalog_coverage",
            "ndcg",
            "mean_intra_list_similarity",
            "mean_popularity",
            "trial",
        ],
        ascending=[False, False, True, True, True],
    )
    best_trial = int(eligible.iloc[0]["trial"])
    trials["is_selected"] = trials["trial"].eq(best_trial)
    selected, scores = selections[best_trial]
    return trials, selected, scores


def build_prediction_table(
    pools: CandidatePools,
    selected: np.ndarray,
    selection_scores: np.ndarray,
    truth: pd.DataFrame,
    config: RerankerConfig,
) -> pd.DataFrame:
    """Create auditable selected top-k rows with every scoring component."""
    truth_pairs = set(zip(truth["candidate_id"], truth["skill_id"], strict=True))
    rows = []
    for candidate_row, candidate_id in enumerate(pools.candidate_ids):
        for rank, pool_position in enumerate(selected[candidate_row], start=1):
            skill_position = pools.skill_positions[candidate_row, pool_position]
            skill_id = str(pools.skill_ids[skill_position])
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "split": config.split,
                    "rank": rank,
                    "skill_id": skill_id,
                    "retrieval_score": float(
                        pools.retrieval_scores[candidate_row, pool_position]
                    ),
                    "evidence_score": float(
                        pools.evidence_scores[candidate_row, pool_position]
                    ),
                    "popularity_score": float(
                        pools.popularity_scores[candidate_row, pool_position]
                    ),
                    "rerank_score": float(selection_scores[candidate_row, rank - 1]),
                    "is_relevant": (candidate_id, skill_id) in truth_pairs,
                }
            )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(
    retrieval_dir: Path,
    model_dir: Path,
    embedding_dir: Path,
    pair_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame], list[Path]]:
    """Load retrieval pools, embeddings, indices, gates, and visible labels."""
    paths = {
        "predictions": retrieval_dir / "learned_retrieval_predictions.parquet",
        "candidate_index": model_dir / "candidate_embedding_index.parquet",
        "learned_skills": model_dir / "learned_skill_embeddings.npy",
        "learned_skill_index": model_dir / "skill_embedding_index.parquet",
        "gates": model_dir / "gate_weights.parquet",
        "tower_embeddings": embedding_dir / "candidate_tower_embeddings.npy",
        "tower_index": embedding_dir / "candidate_tower_embedding_index.parquet",
        "original_skills": embedding_dir / "skill_embeddings.npy",
        "original_skill_index": embedding_dir / "skill_embedding_index.parquet",
        "positive_pairs": pair_path,
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing reranking inputs:\n" + "\n".join(f"  - {path}" for path in missing)
        )
    arrays = {
        "learned_skills": np.load(paths["learned_skills"], allow_pickle=False),
        "tower_embeddings": np.load(paths["tower_embeddings"], allow_pickle=False),
        "original_skills": np.load(paths["original_skills"], allow_pickle=False),
    }
    tables = {
        name: pd.read_parquet(path)
        for name, path in paths.items()
        if name not in arrays
    }
    return arrays, tables, list(paths.values())


def write_results(
    trials: pd.DataFrame,
    predictions: pd.DataFrame,
    input_paths: list[Path],
    output_dir: Path,
    config: RerankerConfig,
) -> None:
    """Persist reranking trials, selected predictions, and lineage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = trials[trials["is_selected"]].iloc[0]
    baseline = trials[trials["trial"].eq(0)].copy()
    baseline["strategy"] = "retrieval_top30"
    chosen = trials[trials["trial"].eq(int(selected["trial"]))].copy()
    chosen["strategy"] = "selected_reranker"
    comparison = pd.concat([baseline, chosen], ignore_index=True)
    outputs = {
        "reranking_trials": trials,
        "reranked_predictions": predictions,
        "reranking_comparison": comparison,
    }
    manifest: dict[str, Any] = {
        "reranker_version": config.reranker_version,
        "config": asdict(config),
        "selected_trial": int(selected["trial"]),
        "test_labels_used": 0,
        "inputs": {str(path): {"sha256": _sha256(path)} for path in input_paths},
        "outputs": {},
    }
    for name, table in outputs.items():
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        manifest["outputs"][name] = {
            "path": path.name,
            "rows": len(table),
            "sha256": _sha256(path),
        }
    (output_dir / "reranking_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--positive-pairs", type=Path, default=DEFAULT_PAIR_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    arrays, tables, input_paths = load_inputs(
        args.retrieval_dir, args.model_dir, args.embedding_dir, args.positive_pairs
    )
    pools, truth = assemble_candidate_pools(
        tables["predictions"],
        tables["candidate_index"],
        arrays["learned_skills"],
        tables["learned_skill_index"],
        arrays["original_skills"],
        tables["original_skill_index"],
        arrays["tower_embeddings"],
        tables["tower_index"],
        tables["gates"],
        tables["positive_pairs"],
        config,
    )
    trials, selected, scores = tune_reranker(pools, truth, config)
    predictions = build_prediction_table(pools, selected, scores, truth, config)
    write_results(trials, predictions, input_paths, args.output_dir, config)
    print(f"Reranking snapshot: {args.output_dir.resolve()}")
    print(trials[trials["is_selected"]].to_string(index=False))


if __name__ == "__main__":
    main()
