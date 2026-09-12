"""Tune the evidence-aware reranker over hybrid top-200 candidate pools."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from talentid.reranking.rerank_skills import (
    assemble_candidate_pools,
    build_prediction_table,
    load_config,
    tune_reranker,
    write_results,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "hybrid_reranker.toml"
DEFAULT_HYBRID_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "hybrid_retrieval"
    / "v1"
    / "hybrid_retrieval_predictions.parquet"
)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "fusion" / "v1"
DEFAULT_EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings" / "v1"
DEFAULT_PAIR_PATH = (
    PROJECT_ROOT / "data" / "training" / "pairs" / "v1" / "positive_pairs.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "reranking" / "v2"


def _require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def validate_hybrid_predictions(
    predictions: pd.DataFrame,
    *,
    split: str,
    pool_size: int,
) -> pd.DataFrame:
    """Validate hybrid lineage and adapt its engine tag for the v1 pool assembler."""
    _require_columns(
        predictions,
        {
            "candidate_id",
            "split",
            "engine",
            "rank",
            "skill_id",
            "score",
            "is_learned_protected",
        },
        "hybrid predictions",
    )
    selected = predictions[predictions["split"].astype(str).eq(split)].copy()
    if selected.empty:
        raise ValueError("hybrid predictions contain no validation rows")
    if set(selected["engine"].astype(str)) != {"hybrid_rrf"}:
        raise ValueError("hybrid predictions must use engine=hybrid_rrf")
    if selected.duplicated(["candidate_id", "skill_id"]).any():
        raise ValueError("hybrid predictions contain duplicate candidate-skill rows")
    expected_ranks = list(range(1, pool_size + 1))
    for candidate_id, group in selected.groupby("candidate_id"):
        if group.sort_values("rank")["rank"].tolist() != expected_ranks:
            raise ValueError(f"hybrid ranks are incomplete for {candidate_id}")
        protected = group[group["is_learned_protected"].astype(bool)]
        if len(protected) != 100 or set(protected["rank"]) != set(range(1, 101)):
            raise ValueError(f"learned top-100 contract failed for {candidate_id}")
    # The existing assembler only uses this tag to select one engine. Relabeling
    # occurs in memory; the persisted hybrid artifact and its lineage are unchanged.
    selected["engine"] = "faiss_flat"
    return selected


def load_hybrid_inputs(
    hybrid_path: Path,
    model_dir: Path,
    embedding_dir: Path,
    pair_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame], list[Path]]:
    """Load the hybrid pool plus the existing reranking feature artifacts."""
    paths = {
        "predictions": hybrid_path,
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
            "Missing hybrid-reranking inputs:\n"
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
    return arrays, tables, list(paths.values())


def run_hybrid_reranking(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble, tune, and persist the hybrid-aware reranker."""
    config = load_config(args.config)
    if config.candidate_pool_size != 200:
        raise ValueError("hybrid reranking requires candidate_pool_size=200")
    arrays, tables, input_paths = load_hybrid_inputs(
        args.hybrid_predictions,
        args.model_dir,
        args.embedding_dir,
        args.positive_pairs,
    )
    tables["predictions"] = validate_hybrid_predictions(
        tables["predictions"],
        split=config.split,
        pool_size=config.candidate_pool_size,
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
    return {
        "trials": trials,
        "predictions": predictions,
        "output_dir": args.output_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-predictions", type=Path, default=DEFAULT_HYBRID_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--positive-pairs", type=Path, default=DEFAULT_PAIR_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    results = run_hybrid_reranking(parse_args())
    trials = results["trials"]
    selected = trials[trials["is_selected"]]
    print(f"Hybrid reranking snapshot: {Path(results['output_dir']).resolve()}")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
