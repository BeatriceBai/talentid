"""Audit skill coverage, concentration, and candidate-pool ceilings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "coverage_audit.toml"
DEFAULT_CATALOG_PATH = (
    PROJECT_ROOT / "data" / "processed" / "onet" / "31.0" / "skills.parquet"
)
DEFAULT_PAIR_PATH = (
    PROJECT_ROOT / "data" / "training" / "pairs" / "v1" / "positive_pairs.parquet"
)
DEFAULT_FROZEN_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval"
    / "v1"
    / "retrieval_predictions.parquet"
)
DEFAULT_LEARNED_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "learned_retrieval"
    / "v1"
    / "learned_retrieval_predictions.parquet"
)
DEFAULT_RERANKED_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "reranking"
    / "v1"
    / "reranked_predictions.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "coverage_audit" / "v1"


@dataclass(frozen=True)
class CoverageAuditConfig:
    """Coverage definitions and next-milestone decision thresholds."""

    audit_version: str = "talentskill-coverage-audit-v1"
    split: str = "validation"
    retrieval_k: int = 30
    pool_k: int = 100
    minimum_pool_pair_recall: float = 0.75
    minimum_pool_truth_skill_coverage: float = 0.80
    minimum_reranked_truth_skill_coverage: float = 0.50

    def validate(self) -> None:
        if not self.audit_version.strip():
            raise ValueError("audit_version must be nonempty")
        if self.split != "validation":
            raise ValueError("only validation is allowed while test labels are sealed")
        if not 1 <= self.retrieval_k <= self.pool_k:
            raise ValueError("retrieval_k must be within pool_k")
        thresholds = (
            self.minimum_pool_pair_recall,
            self.minimum_pool_truth_skill_coverage,
            self.minimum_reranked_truth_skill_coverage,
        )
        if any(not 0 <= value <= 1 for value in thresholds):
            raise ValueError("decision thresholds must be in [0, 1]")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> CoverageAuditConfig:
    """Load coverage definitions and decision thresholds."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    audit = raw["audit"]
    decision = raw["decision"]
    config = CoverageAuditConfig(
        audit_version=str(audit["audit_version"]),
        split=str(audit["split"]),
        retrieval_k=int(audit["retrieval_k"]),
        pool_k=int(audit["pool_k"]),
        minimum_pool_pair_recall=float(decision["minimum_pool_pair_recall"]),
        minimum_pool_truth_skill_coverage=float(
            decision["minimum_pool_truth_skill_coverage"]
        ),
        minimum_reranked_truth_skill_coverage=float(
            decision["minimum_reranked_truth_skill_coverage"]
        ),
    )
    config.validate()
    return config


def _require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def validate_visible_truth(pairs: pd.DataFrame) -> pd.DataFrame:
    """Return validation truth after rejecting any test-label exposure."""
    _require_columns(
        pairs,
        {"candidate_id", "skill_id", "split"},
        "positive pairs",
    )
    normalized = pairs.copy()
    normalized["candidate_id"] = normalized["candidate_id"].astype(str)
    normalized["skill_id"] = normalized["skill_id"].astype(str)
    normalized["split"] = normalized["split"].astype(str)
    if "test" in set(normalized["split"]):
        raise ValueError("positive pairs expose sealed test labels")
    if normalized.duplicated(["candidate_id", "skill_id"]).any():
        raise ValueError("positive pairs contain duplicate candidate-skill keys")
    validation = normalized[normalized["split"].eq("validation")].copy()
    if validation.empty:
        raise ValueError("positive pairs contain no validation truth")
    return validation


def _normalize_predictions(
    table: pd.DataFrame,
    *,
    name: str,
    split: str,
    k: int,
    strategy_column: str | None = None,
    strategy_value: str | None = None,
) -> pd.DataFrame:
    required = {"candidate_id", "split", "rank", "skill_id"}
    if strategy_column:
        required.add(strategy_column)
    _require_columns(table, required, name)
    selected = table[table["split"].astype(str).eq(split)].copy()
    if strategy_column:
        selected = selected[
            selected[strategy_column].astype(str).eq(str(strategy_value))
        ].copy()
    selected = selected[selected["rank"].le(k)].copy()
    selected["candidate_id"] = selected["candidate_id"].astype(str)
    selected["skill_id"] = selected["skill_id"].astype(str)
    if selected.empty:
        raise ValueError(f"{name} contains no selected rows")
    if selected.duplicated(["candidate_id", "skill_id"]).any():
        raise ValueError(f"{name} contains duplicate candidate-skill rows")
    counts = selected.groupby("candidate_id").size()
    if not counts.eq(k).all():
        raise ValueError(f"{name} does not contain exactly {k} rows per candidate")
    return selected


def _normalized_entropy(counts: np.ndarray, catalog_size: int) -> float:
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(catalog_size) if catalog_size > 1 else 0.0


def summarize_stage(
    stage: str,
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    catalog_size: int,
) -> dict[str, Any]:
    """Calculate candidate-level relevance and catalog-level diversity."""
    truth_pairs = set(zip(truth["candidate_id"], truth["skill_id"], strict=True))
    row_pairs = list(
        zip(predictions["candidate_id"], predictions["skill_id"], strict=True)
    )
    predicted_pairs = set(row_pairs)
    truth_skills = set(truth["skill_id"])
    predicted_skills = set(predictions["skill_id"])
    candidate_truth_counts = truth.groupby("candidate_id").size()
    hits = predictions.assign(is_relevant=[pair in truth_pairs for pair in row_pairs])
    candidate_hits = hits.groupby("candidate_id")["is_relevant"].sum()
    candidate_recall = (
        candidate_hits.reindex(candidate_truth_counts.index, fill_value=0)
        / candidate_truth_counts
    )
    frequencies = predictions["skill_id"].value_counts().to_numpy(dtype=np.float64)
    top_count = max(1, math.ceil(catalog_size * 0.01))
    top_share = float(frequencies[:top_count].sum() / frequencies.sum())
    concentration = float(np.sum((frequencies / frequencies.sum()) ** 2))
    return {
        "stage": stage,
        "rows": len(predictions),
        "candidates": predictions["candidate_id"].nunique(),
        "unique_skills": len(predicted_skills),
        "catalog_size": catalog_size,
        "catalog_coverage": len(predicted_skills) / catalog_size,
        "truth_unique_skills": len(truth_skills),
        "truth_catalog_coverage": len(truth_skills) / catalog_size,
        "truth_skill_coverage": len(predicted_skills & truth_skills)
        / len(truth_skills),
        "truth_pair_recall": len(predicted_pairs & truth_pairs) / len(truth_pairs),
        "mean_candidate_recall": float(candidate_recall.mean()),
        "top_1pct_prediction_share": top_share,
        "herfindahl_index": concentration,
        "normalized_entropy": _normalized_entropy(frequencies, catalog_size),
    }


def _skill_type_summary(
    stage: str,
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    catalog_types = catalog.set_index("skill_id")["skill_type"]
    truth_skills = set(truth["skill_id"])
    predicted_skills = set(predictions["skill_id"])
    rows = []
    for skill_type, group in catalog.groupby("skill_type", sort=True):
        type_skills = set(group["skill_id"])
        type_truth = type_skills & truth_skills
        type_predicted = type_skills & predicted_skills
        rows.append(
            {
                "stage": stage,
                "skill_type": skill_type,
                "catalog_skills": len(type_skills),
                "truth_skills": len(type_truth),
                "predicted_skills": len(type_predicted),
                "catalog_coverage": len(type_predicted) / len(type_skills),
                "truth_skill_coverage": (
                    len(type_predicted & type_truth) / len(type_truth)
                    if type_truth
                    else np.nan
                ),
            }
        )
    unknown = (truth_skills | predicted_skills) - set(catalog_types.index)
    if unknown:
        raise ValueError(f"stage {stage} contains {len(unknown)} unknown skill ids")
    return pd.DataFrame(rows)


def _frequency_table(
    stage: str,
    predictions: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    counts = predictions["skill_id"].value_counts().rename("prediction_count")
    output = counts.rename_axis("skill_id").reset_index()
    output["prediction_share"] = output["prediction_count"] / len(predictions)
    output["frequency_rank"] = np.arange(1, len(output) + 1)
    output.insert(0, "stage", stage)
    return output.merge(
        catalog[["skill_id", "skill_name", "skill_type"]],
        on="skill_id",
        how="left",
        validate="many_to_one",
    )


def audit_coverage(
    catalog: pd.DataFrame,
    positive_pairs: pd.DataFrame,
    frozen_predictions: pd.DataFrame,
    learned_predictions: pd.DataFrame,
    reranked_predictions: pd.DataFrame,
    config: CoverageAuditConfig,
) -> dict[str, pd.DataFrame | str]:
    """Compare truth, retrieval pools, and final-list catalog exposure."""
    config.validate()
    _require_columns(catalog, {"skill_id", "skill_name", "skill_type"}, "catalog")
    catalog = catalog.copy()
    catalog["skill_id"] = catalog["skill_id"].astype(str)
    if catalog["skill_id"].duplicated().any():
        raise ValueError("catalog skill ids must be unique")
    truth = validate_visible_truth(positive_pairs)
    candidate_ids = set(truth["candidate_id"])
    stages = {
        "frozen_top30": _normalize_predictions(
            frozen_predictions,
            name="frozen predictions",
            split=config.split,
            k=config.retrieval_k,
            strategy_column="strategy",
            strategy_value="reliability_weighted",
        ),
        "learned_top30": _normalize_predictions(
            learned_predictions,
            name="learned predictions",
            split=config.split,
            k=config.retrieval_k,
            strategy_column="engine",
            strategy_value="faiss_flat",
        ),
        "learned_top100_pool": _normalize_predictions(
            learned_predictions,
            name="learned predictions",
            split=config.split,
            k=config.pool_k,
            strategy_column="engine",
            strategy_value="faiss_flat",
        ),
        "reranked_top30": _normalize_predictions(
            reranked_predictions,
            name="reranked predictions",
            split=config.split,
            k=config.retrieval_k,
        ),
    }
    for name, predictions in stages.items():
        if set(predictions["candidate_id"]) != candidate_ids:
            raise ValueError(f"{name} candidates do not align with validation truth")
    summary = pd.DataFrame(
        [
            summarize_stage(name, predictions, truth, len(catalog))
            for name, predictions in stages.items()
        ]
    )
    by_type = pd.concat(
        [
            _skill_type_summary(name, predictions, truth, catalog)
            for name, predictions in stages.items()
        ],
        ignore_index=True,
    )
    frequencies = pd.concat(
        [
            _frequency_table(name, predictions, catalog)
            for name, predictions in stages.items()
        ],
        ignore_index=True,
    )
    pool = summary[summary["stage"].eq("learned_top100_pool")].iloc[0]
    reranked = summary[summary["stage"].eq("reranked_top30")].iloc[0]
    needs_hybrid = (
        pool["truth_pair_recall"] < config.minimum_pool_pair_recall
        or pool["truth_skill_coverage"] < config.minimum_pool_truth_skill_coverage
        or reranked["truth_skill_coverage"]
        < config.minimum_reranked_truth_skill_coverage
    )
    recommendation = "hybrid_candidate_generation" if needs_hybrid else "llm_judge"
    return {
        "coverage_summary": summary,
        "coverage_by_skill_type": by_type,
        "skill_prediction_frequency": frequencies,
        "recommendation": recommendation,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(paths: dict[str, Path]) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    """Load the catalog, visible truth, and stage predictions."""
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing coverage-audit inputs:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    return (
        {name: pd.read_parquet(path) for name, path in paths.items()},
        list(paths.values()),
    )


def write_results(
    results: dict[str, pd.DataFrame | str],
    input_paths: list[Path],
    output_dir: Path,
    config: CoverageAuditConfig,
) -> None:
    """Persist coverage tables and the next-milestone decision."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "audit_version": config.audit_version,
        "config": asdict(config),
        "recommendation": results["recommendation"],
        "test_labels_used": 0,
        "inputs": {str(path): {"sha256": _sha256(path)} for path in input_paths},
        "outputs": {},
    }
    for name in (
        "coverage_summary",
        "coverage_by_skill_type",
        "skill_prediction_frequency",
    ):
        table = results[name]
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"{name} must be a DataFrame")
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        manifest["outputs"][name] = {
            "path": path.name,
            "rows": len(table),
            "sha256": _sha256(path),
        }
    (output_dir / "coverage_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--positive-pairs", type=Path, default=DEFAULT_PAIR_PATH)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN_PATH)
    parser.add_argument("--learned", type=Path, default=DEFAULT_LEARNED_PATH)
    parser.add_argument("--reranked", type=Path, default=DEFAULT_RERANKED_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = {
        "catalog": args.catalog,
        "positive_pairs": args.positive_pairs,
        "frozen_predictions": args.frozen,
        "learned_predictions": args.learned,
        "reranked_predictions": args.reranked,
    }
    tables, input_paths = load_inputs(paths)
    results = audit_coverage(**tables, config=config)
    write_results(results, input_paths, args.output_dir, config)
    print(f"Coverage audit: {args.output_dir.resolve()}")
    print(results["coverage_summary"].to_string(index=False))
    print(f"Recommendation: {results['recommendation']}")


if __name__ == "__main__":
    main()
