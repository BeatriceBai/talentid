"""Build leakage-safe candidate-tower and skill-text features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANDIDATE_DIR = (
    PROJECT_ROOT / "data" / "synthetic" / "candidates" / "v1"
)
DEFAULT_ONET_DIR = PROJECT_ROOT / "data" / "processed" / "onet" / "31.0"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "features" / "v1"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "features.toml"

SOURCES = (
    "resume",
    "projects",
    "job_history",
    "courses",
    "self_reported_skills",
)

CANDIDATE_TOWER_COLUMNS = [
    "candidate_id",
    "source",
    "tower_text",
    "is_available",
    "updated_at",
    "age_days",
    "item_count",
    "reliability_score",
    "character_count",
    "token_count",
    "log1p_item_count",
    "is_stale",
    "is_high_volume",
    "split",
    "feature_version",
]

SKILL_FEATURE_COLUMNS = [
    "skill_id",
    "skill_name",
    "skill_type",
    "skill_text",
    "taxonomy_version",
    "feature_version",
]

CANDIDATE_INDEX_COLUMNS = [
    "candidate_id",
    "primary_onet_soc_code",
    "years_experience",
    "split",
    "taxonomy_version",
    "feature_version",
]


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for deterministic text and metadata construction."""

    feature_version: str = "talentskill-features-v1"
    taxonomy_version: str = "onet-31.0+talentskill-v1"
    stale_after_days: int = 730
    high_volume_threshold: int = 40
    max_characters_resume: int = 6_000
    max_characters_projects: int = 4_000
    max_characters_job_history: int = 6_000
    max_characters_courses: int = 3_000
    max_characters_self_reported_skills: int = 4_000

    def validate(self) -> None:
        """Reject invalid feature settings before reading large tables."""
        if not self.feature_version.strip() or not self.taxonomy_version.strip():
            raise ValueError("feature and taxonomy versions must be nonempty")
        if self.stale_after_days < 1 or self.high_volume_threshold < 1:
            raise ValueError("thresholds must be positive")
        if any(value < 1 for value in self.max_characters.values()):
            raise ValueError("maximum character counts must be positive")

    @property
    def max_characters(self) -> dict[str, int]:
        return {
            "resume": self.max_characters_resume,
            "projects": self.max_characters_projects,
            "job_history": self.max_characters_job_history,
            "courses": self.max_characters_courses,
            "self_reported_skills": self.max_characters_self_reported_skills,
        }


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> FeatureConfig:
    """Load the feature configuration from TOML."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    features = raw["features"]
    maximums = raw["max_characters"]
    config = FeatureConfig(
        feature_version=str(features["feature_version"]),
        taxonomy_version=str(features["taxonomy_version"]),
        stale_after_days=int(features["stale_after_days"]),
        high_volume_threshold=int(features["high_volume_threshold"]),
        max_characters_resume=int(maximums["resume"]),
        max_characters_projects=int(maximums["projects"]),
        max_characters_job_history=int(maximums["job_history"]),
        max_characters_courses=int(maximums["courses"]),
        max_characters_self_reported_skills=int(
            maximums["self_reported_skills"]
        ),
    )
    config.validate()
    return config


def load_parquet_tables(
    directory: Path,
    required_names: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Load required Parquet tables with a useful missing-file error."""
    paths = {name: directory / f"{name}.parquet" for name in required_names}
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError("Missing required feature inputs:\n" + formatted)
    return {name: pd.read_parquet(path) for name, path in paths.items()}


def normalize_text(value: Any) -> str:
    """Normalize whitespace and missing values without changing case."""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def truncate_text(text: str, limit: int) -> str:
    """Apply a deterministic character budget at a word boundary."""
    if len(text) <= limit:
        return text
    shortened = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return shortened + " …"


def _date_label(value: Any, missing_label: str = "unknown") -> str:
    if value is None or pd.isna(value):
        return missing_label
    return pd.Timestamp(value).date().isoformat()


def _group_text(
    table: pd.DataFrame,
    candidate_ids: set[str],
    sort_columns: list[str],
    formatter: Callable[[pd.Series], str],
) -> dict[str, str]:
    """Create deterministic candidate-level documents from evidence rows."""
    if table.empty:
        return {}
    safe = table[table["candidate_id"].isin(candidate_ids)].copy()
    safe = safe.sort_values(
        ["candidate_id", *sort_columns],
        ascending=[True, *([False] * len(sort_columns))],
        na_position="last",
        kind="stable",
    )
    output: dict[str, str] = {}
    for candidate_id, group in safe.groupby("candidate_id", sort=False):
        parts = [normalize_text(formatter(row)) for _, row in group.iterrows()]
        output[str(candidate_id)] = "\n".join(part for part in parts if part)
    return output


def _build_resume_text(
    resumes: pd.DataFrame,
    candidate_ids: set[str],
) -> dict[str, str]:
    if resumes.empty:
        return {}
    safe = resumes[resumes["candidate_id"].isin(candidate_ids)].copy()
    safe = safe.sort_values(
        ["candidate_id", "updated_at"],
        ascending=[True, False],
        na_position="last",
        kind="stable",
    ).drop_duplicates("candidate_id", keep="first")
    return {
        str(row["candidate_id"]): normalize_text(row["resume_text"])
        for _, row in safe.iterrows()
    }


def _build_job_history_text(
    jobs: pd.DataFrame,
    candidate_ids: set[str],
) -> dict[str, str]:
    def format_job(row: pd.Series) -> str:
        end = _date_label(row["end_date"], missing_label="present")
        return (
            f"Role: {normalize_text(row['job_title'])}. "
            f"Dates: {_date_label(row['start_date'])} to {end}. "
            f"Evidence: {normalize_text(row['description'])}"
        )

    if jobs.empty:
        return {}
    safe = jobs[jobs["candidate_id"].isin(candidate_ids)].copy()
    safe["_effective_end"] = safe["end_date"].fillna(pd.Timestamp.max)
    safe = safe.sort_values(
        ["candidate_id", "_effective_end", "start_date", "job_history_id"],
        ascending=[True, False, False, False],
        kind="stable",
    )
    output: dict[str, str] = {}
    for candidate_id, group in safe.groupby("candidate_id", sort=False):
        parts = [normalize_text(format_job(row)) for _, row in group.iterrows()]
        output[str(candidate_id)] = "\n".join(part for part in parts if part)
    return output


def _build_project_text(
    projects: pd.DataFrame,
    candidate_ids: set[str],
) -> dict[str, str]:
    def format_project(row: pd.Series) -> str:
        return (
            f"Project: {normalize_text(row['project_title'])}. "
            f"Completed: {_date_label(row['completed_at'])}. "
            f"Evidence: {normalize_text(row['description'])}"
        )

    return _group_text(
        projects,
        candidate_ids,
        ["completed_at", "project_id"],
        format_project,
    )


def _build_course_text(
    courses: pd.DataFrame,
    candidate_ids: set[str],
) -> dict[str, str]:
    def format_course(row: pd.Series) -> str:
        return (
            f"Course: {normalize_text(row['course_name'])}. "
            f"Completed: {_date_label(row['completed_at'])}."
        )

    return _group_text(
        courses,
        candidate_ids,
        ["completed_at", "course_id"],
        format_course,
    )


def _build_self_report_text(
    claims: pd.DataFrame,
    candidate_ids: set[str],
) -> dict[str, str]:
    def format_claim(row: pd.Series) -> str:
        proficiency = row.get("reported_proficiency")
        level = "unknown" if pd.isna(proficiency) else str(int(proficiency))
        return f"Claimed skill: {normalize_text(row['raw_skill_name'])}; level: {level}."

    return _group_text(
        claims,
        candidate_ids,
        ["claimed_at", "claim_id"],
        format_claim,
    )


def _build_tower_texts(
    candidate_tables: dict[str, pd.DataFrame],
    candidate_ids: set[str],
) -> dict[str, dict[str, str]]:
    """Build every tower using only fields observable in a real profile."""
    return {
        "resume": _build_resume_text(candidate_tables["resumes"], candidate_ids),
        "projects": _build_project_text(
            candidate_tables["projects"], candidate_ids
        ),
        "job_history": _build_job_history_text(
            candidate_tables["job_history"], candidate_ids
        ),
        "courses": _build_course_text(candidate_tables["courses"], candidate_ids),
        "self_reported_skills": _build_self_report_text(
            candidate_tables["self_reported_skills"], candidate_ids
        ),
    }


def build_candidate_tower_features(
    candidate_tables: dict[str, pd.DataFrame],
    config: FeatureConfig,
) -> pd.DataFrame:
    """Create exactly one feature row for every candidate and source."""
    candidates = candidate_tables["candidates"].copy()
    status = candidate_tables["source_status"].copy()
    candidate_ids = set(candidates["candidate_id"].astype(str))
    tower_texts = _build_tower_texts(candidate_tables, candidate_ids)

    split_lookup = candidates.set_index("candidate_id")["split"].to_dict()
    rows: list[dict[str, Any]] = []
    for _, status_row in status.iterrows():
        candidate_id = str(status_row["candidate_id"])
        source = str(status_row["source"])
        if source not in SOURCES:
            raise ValueError(f"Unknown source in source_status: {source}")
        text = truncate_text(
            normalize_text(tower_texts[source].get(candidate_id, "")),
            config.max_characters[source],
        )
        item_count = int(status_row["item_count"])
        available = bool(status_row["is_available"])
        age_days = (
            None
            if pd.isna(status_row["age_days"])
            else int(status_row["age_days"])
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "source": source,
                "tower_text": text,
                "is_available": available,
                "updated_at": status_row["updated_at"],
                "age_days": age_days,
                "item_count": item_count,
                "reliability_score": float(status_row["reliability_score"]),
                "character_count": len(text),
                "token_count": len(text.split()),
                "log1p_item_count": round(math.log1p(item_count), 6),
                "is_stale": bool(
                    available
                    and age_days is not None
                    and age_days > config.stale_after_days
                ),
                "is_high_volume": bool(
                    available and item_count > config.high_volume_threshold
                ),
                "split": split_lookup[candidate_id],
                "feature_version": config.feature_version,
            }
        )
    return pd.DataFrame(rows, columns=CANDIDATE_TOWER_COLUMNS).sort_values(
        ["candidate_id", "source"]
    ).reset_index(drop=True)


def build_skill_features(
    skills: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Create one textual representation for each canonical skill."""
    rows = []
    for _, row in skills.sort_values("skill_id").iterrows():
        name = normalize_text(row["skill_name"])
        skill_type = normalize_text(row["skill_type"])
        description = normalize_text(row.get("description"))
        category = normalize_text(row.get("category_name"))
        parts = [f"Skill: {name}.", f"Type: {skill_type}."]
        if description:
            parts.append(f"Description: {description}")
        if category:
            parts.append(f"Category: {category}.")
        rows.append(
            {
                "skill_id": str(row["skill_id"]),
                "skill_name": name,
                "skill_type": skill_type,
                "skill_text": normalize_text(" ".join(parts)),
                "taxonomy_version": config.taxonomy_version,
                "feature_version": config.feature_version,
            }
        )
    return pd.DataFrame(rows, columns=SKILL_FEATURE_COLUMNS)


def build_candidate_index(
    candidates: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Keep non-model candidate metadata for joins and evaluation slices."""
    result = candidates[
        [
            "candidate_id",
            "primary_onet_soc_code",
            "years_experience",
            "split",
            "taxonomy_version",
        ]
    ].copy()
    result["feature_version"] = config.feature_version
    return result[CANDIDATE_INDEX_COLUMNS].sort_values("candidate_id").reset_index(
        drop=True
    )


def build_feature_tables(
    candidate_tables: dict[str, pd.DataFrame],
    onet_tables: dict[str, pd.DataFrame],
    config: FeatureConfig,
) -> dict[str, pd.DataFrame]:
    """Build and validate every feature-layer table."""
    config.validate()
    tables = {
        "candidate_tower_features": build_candidate_tower_features(
            candidate_tables, config
        ),
        "skill_features": build_skill_features(onet_tables["skills"], config),
        "candidate_index": build_candidate_index(
            candidate_tables["candidates"], config
        ),
    }
    validate_feature_tables(tables, candidate_tables, onet_tables, config)
    return tables


def validate_feature_tables(
    feature_tables: dict[str, pd.DataFrame],
    candidate_tables: dict[str, pd.DataFrame],
    onet_tables: dict[str, pd.DataFrame],
    config: FeatureConfig,
) -> None:
    """Enforce feature coverage, keys, budgets, and leakage boundaries."""
    towers = feature_tables["candidate_tower_features"]
    skills = feature_tables["skill_features"]
    candidate_count = len(candidate_tables["candidates"])

    if len(towers) != candidate_count * len(SOURCES):
        raise ValueError("Expected one row per candidate and source")
    if towers.duplicated(["candidate_id", "source"]).any():
        raise ValueError("Duplicate candidate-source feature rows found")
    if skills["skill_id"].duplicated().any():
        raise ValueError("Duplicate skill feature rows found")
    if set(skills["skill_id"]) != set(onet_tables["skills"]["skill_id"]):
        raise ValueError("Skill feature coverage does not match the taxonomy")

    unavailable = towers[~towers["is_available"]]
    if unavailable["tower_text"].ne("").any():
        raise ValueError("Unavailable towers must have empty model text")
    if unavailable["reliability_score"].ne(0).any():
        raise ValueError("Unavailable towers must have zero reliability")
    available = towers[towers["is_available"]]
    if available["tower_text"].eq("").any():
        raise ValueError("Available towers must have nonempty model text")

    for source, maximum in config.max_characters.items():
        source_lengths = towers.loc[
            towers["source"].eq(source), "character_count"
        ]
        if source_lengths.gt(maximum + 2).any():
            raise ValueError(f"{source} exceeds its character budget")

    forbidden_columns = {
        "canonical_skill_id",
        "oracle_is_true_skill",
        "skill_id",
        "onet_soc_code",
    }
    leaked_columns = forbidden_columns & set(towers.columns)
    if leaked_columns:
        raise ValueError(f"Forbidden model columns found: {sorted(leaked_columns)}")
    if any("oracle_" in column or "truth" in column for column in towers.columns):
        raise ValueError("Oracle or truth columns found in model features")


def file_sha256(path: Path) -> str:
    """Hash a source artifact for feature lineage."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_feature_artifacts(
    feature_tables: dict[str, pd.DataFrame],
    output_dir: Path,
    config: FeatureConfig,
    input_paths: list[Path],
) -> None:
    """Write Parquet features and a deterministic lineage manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = {}
    for name, table in feature_tables.items():
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        output_rows[name] = len(table)
        print(f"{name}: {len(table):,} rows -> {path}")

    manifest = {
        "feature_version": config.feature_version,
        "taxonomy_version": config.taxonomy_version,
        "config": asdict(config),
        "inputs": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(input_paths)
        ],
        "output_rows": output_rows,
    }
    manifest_path = output_dir / "feature_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"manifest -> {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR
    )
    parser.add_argument("--onet-dir", type=Path, default=DEFAULT_ONET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Load, transform, validate, and persist feature artifacts."""
    args = parse_args()
    config = load_config(args.config)
    candidate_names = (
        "candidates",
        "resumes",
        "job_history",
        "projects",
        "courses",
        "self_reported_skills",
        "source_status",
    )
    candidate_tables = load_parquet_tables(args.candidate_dir, candidate_names)
    onet_tables = load_parquet_tables(args.onet_dir, ("skills",))
    feature_tables = build_feature_tables(candidate_tables, onet_tables, config)
    input_paths = [
        args.candidate_dir / f"{name}.parquet" for name in candidate_names
    ]
    input_paths.append(args.onet_dir / "skills.parquet")
    input_paths.append(args.config)
    write_feature_artifacts(
        feature_tables,
        args.output_dir,
        config,
        input_paths,
    )


if __name__ == "__main__":
    main()
