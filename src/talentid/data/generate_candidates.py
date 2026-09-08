"""Generate deterministic synthetic candidate profiles grounded in O*NET."""

from __future__ import annotations

import argparse
import math
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ONET_DIR = PROJECT_ROOT / "data" / "processed" / "onet" / "31.0"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic" / "candidates" / "v1"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "synthetic_candidates.toml"

SOURCES = (
    "resume",
    "projects",
    "job_history",
    "courses",
    "self_reported_skills",
)

TABLE_COLUMNS = {
    "candidates": [
        "candidate_id",
        "primary_onet_soc_code",
        "profile_segment",
        "skill_claim_style",
        "years_experience",
        "profile_created_at",
        "split",
        "taxonomy_version",
        "generation_seed",
    ],
    "resumes": [
        "resume_id",
        "candidate_id",
        "resume_text",
        "updated_at",
    ],
    "job_history": [
        "job_history_id",
        "candidate_id",
        "onet_soc_code",
        "job_title",
        "start_date",
        "end_date",
        "description",
    ],
    "projects": [
        "project_id",
        "candidate_id",
        "project_title",
        "description",
        "completed_at",
    ],
    "courses": [
        "course_id",
        "candidate_id",
        "course_name",
        "skill_id",
        "completed_at",
    ],
    "self_reported_skills": [
        "claim_id",
        "candidate_id",
        "raw_skill_name",
        "canonical_skill_id",
        "reported_proficiency",
        "claimed_at",
        "alias_type",
        "oracle_is_true_skill",
    ],
    "candidate_skill_truth": [
        "candidate_id",
        "skill_id",
        "proficiency",
        "relevance_score",
        "taxonomy_version",
    ],
    "source_status": [
        "candidate_id",
        "source",
        "is_available",
        "updated_at",
        "age_days",
        "item_count",
        "reliability_score",
    ],
}


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for one reproducible synthetic population snapshot."""

    n_candidates: int = 10_000
    seed: int = 20_260_908
    snapshot_date: str = "2026-09-01"
    taxonomy_version: str = "onet-31.0+talentskill-v1"
    train_fraction: float = 0.80
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    resume_missing_rate: float = 0.08
    projects_missing_rate: float = 0.22
    job_history_missing_rate: float = 0.04
    courses_missing_rate: float = 0.35
    self_reported_skills_missing_rate: float = 0.12
    stale_resume_rate: float = 0.22
    noisy_claim_rate: float = 0.12
    max_true_skills: int = 45
    max_self_reported_skills: int = 100

    def validate(self) -> None:
        """Reject invalid configuration values before generation starts."""
        if self.n_candidates < 1:
            raise ValueError("n_candidates must be positive")
        if self.max_true_skills < 1 or self.max_self_reported_skills < 1:
            raise ValueError("skill count limits must be positive")

        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
            raise ValueError("train, validation, and test fractions must sum to 1")

        probabilities = (
            self.resume_missing_rate,
            self.projects_missing_rate,
            self.job_history_missing_rate,
            self.courses_missing_rate,
            self.self_reported_skills_missing_rate,
            self.stale_resume_rate,
            self.noisy_claim_rate,
        )
        if any(value < 0 or value > 1 for value in probabilities):
            raise ValueError("rates must be between 0 and 1")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> GenerationConfig:
    """Load generator settings from TOML using only the standard library."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    generation = raw["generation"]
    splits = raw["splits"]
    missingness = raw["missingness"]
    quality = raw["quality"]

    config = GenerationConfig(
        n_candidates=int(generation["n_candidates"]),
        seed=int(generation["seed"]),
        snapshot_date=str(generation["snapshot_date"]),
        taxonomy_version=str(generation["taxonomy_version"]),
        train_fraction=float(splits["train"]),
        validation_fraction=float(splits["validation"]),
        test_fraction=float(splits["test"]),
        resume_missing_rate=float(missingness["resume"]),
        projects_missing_rate=float(missingness["projects"]),
        job_history_missing_rate=float(missingness["job_history"]),
        courses_missing_rate=float(missingness["courses"]),
        self_reported_skills_missing_rate=float(
            missingness["self_reported_skills"]
        ),
        stale_resume_rate=float(quality["stale_resume_rate"]),
        noisy_claim_rate=float(quality["noisy_claim_rate"]),
        max_true_skills=int(quality["max_true_skills"]),
        max_self_reported_skills=int(quality["max_self_reported_skills"]),
    )
    config.validate()
    return config


def load_onet_tables(onet_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the normalized O*NET tables required by the generator."""
    required = (
        "occupations",
        "job_titles",
        "tasks",
        "skills",
        "occupation_skills",
    )
    paths = {name: onet_dir / f"{name}.parquet" for name in required}
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing processed O*NET tables. Run ingest_onet first:\n" + formatted
        )
    return {name: pd.read_parquet(path) for name, path in paths.items()}


def _frame(rows: list[dict[str, Any]], table_name: str) -> pd.DataFrame:
    """Create a table with stable columns, including when it has no rows."""
    return pd.DataFrame(rows, columns=TABLE_COLUMNS[table_name])


def _random_date_before(
    snapshot: pd.Timestamp,
    rng: np.random.Generator,
    minimum_days: int,
    maximum_days: int,
) -> pd.Timestamp:
    age_days = int(rng.integers(minimum_days, maximum_days + 1))
    return snapshot - pd.Timedelta(days=age_days)


def _split_for_candidate(index: int, config: GenerationConfig) -> str:
    """Assign stable candidate-level splits without cross-split leakage."""
    position = (index * 0.6180339887498949) % 1.0
    if position < config.train_fraction:
        return "train"
    if position < config.train_fraction + config.validation_fraction:
        return "validation"
    return "test"


def _skill_relevance(relationships: pd.DataFrame) -> pd.Series:
    importance = pd.to_numeric(
        relationships["importance"], errors="coerce"
    ).fillna(0.0) / 5.0
    level = pd.to_numeric(
        relationships["level"], errors="coerce"
    ).fillna(0.0) / 7.0
    hot = relationships["hot_technology"].fillna(False).astype(bool)
    demand = relationships["in_demand"].fillna(False).astype(bool)
    not_relevant = relationships["not_relevant"].fillna(False).astype(bool)

    rated_score = 0.65 * importance + 0.35 * level
    software_score = 0.42 + 0.20 * hot + 0.14 * demand
    score = rated_score.where(
        relationships["skill_type"].ne("software"), software_score
    )
    return score.clip(0.03, 1.0).where(~not_relevant, 0.03)


def _build_lookups(
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    occupations = tables["occupations"].copy()
    relationships = tables["occupation_skills"].copy()
    relationships["relevance_score"] = _skill_relevance(relationships)

    eligible_codes = sorted(
        set(occupations["onet_soc_code"])
        & set(relationships["onet_soc_code"])
    )
    if not eligible_codes:
        raise ValueError("No occupations have skill relationships")

    occupation_rows = occupations.set_index("onet_soc_code").to_dict("index")
    skill_rows = tables["skills"].set_index("skill_id").to_dict("index")

    titles = {
        code: group["job_title"].dropna().astype(str).tolist()
        for code, group in tables["job_titles"].groupby("onet_soc_code")
    }
    tasks = {
        code: group["task"].dropna().astype(str).tolist()
        for code, group in tables["tasks"].groupby("onet_soc_code")
    }
    occupation_skills = {
        code: group[
            ["skill_id", "skill_name", "relevance_score"]
        ].drop_duplicates("skill_id")
        for code, group in relationships.groupby("onet_soc_code")
    }
    major_groups: dict[str, list[str]] = {}
    for code in eligible_codes:
        major_groups.setdefault(code[:2], []).append(code)

    return {
        "eligible_codes": eligible_codes,
        "occupations": occupation_rows,
        "skills": skill_rows,
        "all_skill_ids": np.array(sorted(skill_rows), dtype=object),
        "titles": titles,
        "tasks": tasks,
        "occupation_skills": occupation_skills,
        "major_groups": major_groups,
    }


def _sample_true_skills(
    occupation_code: str,
    years_experience: int,
    lookups: dict[str, Any],
    config: GenerationConfig,
    rng: np.random.Generator,
) -> list[tuple[str, float, float]]:
    candidates = lookups["occupation_skills"][occupation_code]
    desired = int(
        np.clip(
            rng.poisson(22 + 0.8 * years_experience),
            12,
            config.max_true_skills,
        )
    )
    count = min(desired, len(candidates))
    weights = candidates["relevance_score"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    positions = rng.choice(len(candidates), size=count, replace=False, p=weights)

    output = []
    for position in positions:
        row = candidates.iloc[int(position)]
        relevance = float(row["relevance_score"])
        experience_effect = min(years_experience / 20.0, 1.0)
        proficiency = float(
            np.clip(
                1.0 + 3.0 * relevance + experience_effect + rng.normal(0, 0.35),
                1.0,
                5.0,
            )
        )
        output.append((str(row["skill_id"]), relevance, proficiency))
    return output


def _surface_skill_name(
    canonical_name: str,
    rng: np.random.Generator,
) -> tuple[str, str]:
    """Create realistic aliases while retaining the canonical mapping."""
    normalized = " ".join(canonical_name.casefold().split())
    manual_aliases = {
        "amazon web services aws": "AWS",
        "google cloud platform gcp": "GCP",
        "microsoft azure": "Azure",
        "microsoft excel": "Excel",
        "python": "Python programming",
        "structured query language sql": "SQL",
    }
    if normalized in manual_aliases and rng.random() < 0.70:
        return manual_aliases[normalized], "manual_synonym"
    if normalized.startswith("microsoft ") and rng.random() < 0.35:
        return canonical_name.split(" ", 1)[1], "vendor_shortened"
    if rng.random() < 0.12:
        return canonical_name.lower(), "case_variant"
    if " and " in normalized and rng.random() < 0.12:
        return canonical_name.replace(" and ", " & "), "symbol_variant"
    return canonical_name, "canonical"


def _sample_noise_skills(
    all_skill_ids: np.ndarray,
    excluded: set[str],
    count: int,
    rng: np.random.Generator,
) -> list[str]:
    """Sample unique nontruth skills without scanning the full taxonomy."""
    available_count = len(all_skill_ids) - len(excluded)
    target = min(count, available_count)
    selected: set[str] = set()
    while len(selected) < target:
        remaining = target - len(selected)
        draw_count = min(len(all_skill_ids), max(remaining * 2, 8))
        draws = rng.choice(all_skill_ids, size=draw_count, replace=False)
        selected.update(
            str(skill_id)
            for skill_id in draws
            if str(skill_id) not in excluded
        )
    return sorted(selected)[:target]


def _source_availability(
    segment: str,
    config: GenerationConfig,
    rng: np.random.Generator,
) -> dict[str, bool]:
    multipliers = {"complete": 0.45, "partial": 1.0, "sparse": 2.25}
    missing_rates = {
        "resume": config.resume_missing_rate,
        "projects": config.projects_missing_rate,
        "job_history": config.job_history_missing_rate,
        "courses": config.courses_missing_rate,
        "self_reported_skills": config.self_reported_skills_missing_rate,
    }
    multiplier = multipliers[segment]
    available = {
        source: rng.random() >= min(rate * multiplier, 0.92)
        for source, rate in missing_rates.items()
    }
    if not any(available.values()):
        available["job_history"] = True
    return available


def _reliability(
    source: str,
    age_days: int | None,
    item_count: int,
) -> float:
    bases = {
        "resume": 0.90,
        "projects": 0.82,
        "job_history": 0.88,
        "courses": 0.72,
        "self_reported_skills": 0.65,
    }
    half_lives = {
        "resume": 1_825,
        "projects": 1_460,
        "job_history": 2_920,
        "courses": 2_190,
        "self_reported_skills": 1_095,
    }
    if item_count == 0:
        return 0.0
    freshness = math.exp(-max(age_days or 0, 0) / half_lives[source])
    volume_penalty = 1.0
    if source == "self_reported_skills" and item_count > 30:
        volume_penalty = 1.0 / (1.0 + 0.35 * math.log1p(item_count - 30))
    return round(float(np.clip(bases[source] * freshness * volume_penalty, 0, 1)), 4)


def generate_candidate_tables(
    onet_tables: dict[str, pd.DataFrame],
    config: GenerationConfig,
) -> dict[str, pd.DataFrame]:
    """Generate normalized candidate tables and oracle skill truth."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    snapshot = pd.Timestamp(config.snapshot_date)
    lookups = _build_lookups(onet_tables)

    codes = np.array(lookups["eligible_codes"], dtype=object)
    shuffled_positions = rng.permutation(len(codes))
    ranks = np.empty(len(codes), dtype=int)
    ranks[shuffled_positions] = np.arange(1, len(codes) + 1)
    occupation_weights = 1.0 / np.power(ranks, 0.75)
    occupation_weights = occupation_weights / occupation_weights.sum()

    rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in TABLE_COLUMNS
    }

    for index in range(config.n_candidates):
        candidate_id = f"cand_{index + 1:07d}"
        occupation_code = str(rng.choice(codes, p=occupation_weights))
        occupation = lookups["occupations"][occupation_code]
        segment = str(rng.choice(
            ["complete", "partial", "sparse"], p=[0.55, 0.30, 0.15]
        ))
        claim_style = str(rng.choice(
            ["terse", "typical", "exhaustive"], p=[0.24, 0.66, 0.10]
        ))
        years_experience = int(np.clip(rng.gamma(2.2, 4.2), 0, 35))
        profile_created_at = _random_date_before(snapshot, rng, 30, 1_825)
        availability = _source_availability(segment, config, rng)

        rows["candidates"].append(
            {
                "candidate_id": candidate_id,
                "primary_onet_soc_code": occupation_code,
                "profile_segment": segment,
                "skill_claim_style": claim_style,
                "years_experience": years_experience,
                "profile_created_at": profile_created_at,
                "split": _split_for_candidate(index, config),
                "taxonomy_version": config.taxonomy_version,
                "generation_seed": config.seed,
            }
        )

        true_skills = _sample_true_skills(
            occupation_code,
            years_experience,
            lookups,
            config,
            rng,
        )
        true_ids = {skill_id for skill_id, _, _ in true_skills}
        true_names = [
            str(lookups["skills"][skill_id]["skill_name"])
            for skill_id, _, _ in true_skills
        ]
        for skill_id, relevance, proficiency in true_skills:
            rows["candidate_skill_truth"].append(
                {
                    "candidate_id": candidate_id,
                    "skill_id": skill_id,
                    "proficiency": round(proficiency, 3),
                    "relevance_score": round(relevance, 4),
                    "taxonomy_version": config.taxonomy_version,
                }
            )

        source_dates: dict[str, pd.Timestamp | None] = {
            source: None for source in SOURCES
        }
        source_counts = {source: 0 for source in SOURCES}

        occupation_tasks = lookups["tasks"].get(occupation_code, [])
        task_sample = []
        if occupation_tasks:
            task_count = min(len(occupation_tasks), 3)
            task_sample = rng.choice(
                occupation_tasks, size=task_count, replace=False
            ).tolist()

        if availability["job_history"]:
            job_count = int(np.clip(1 + rng.poisson(1.2), 1, 4))
            career_end = snapshot
            for job_index in range(job_count):
                if job_index == 0:
                    job_code = occupation_code
                else:
                    related_codes = lookups["major_groups"][occupation_code[:2]]
                    job_code = str(rng.choice(related_codes))
                titles = lookups["titles"].get(job_code, [])
                fallback_title = lookups["occupations"][job_code]["title"]
                job_title = str(rng.choice(titles)) if titles else str(fallback_title)
                duration_days = int(rng.integers(365, 1_461))
                gap_days = int(rng.integers(0, 121))
                end_date = pd.NaT if job_index == 0 else career_end
                start_date = career_end - pd.Timedelta(days=duration_days)
                description_tasks = lookups["tasks"].get(job_code, [])
                description = (
                    str(rng.choice(description_tasks))
                    if description_tasks
                    else f"Performed responsibilities related to {job_title}."
                )
                rows["job_history"].append(
                    {
                        "job_history_id": f"{candidate_id}_job_{job_index + 1:02d}",
                        "candidate_id": candidate_id,
                        "onet_soc_code": job_code,
                        "job_title": job_title,
                        "start_date": start_date,
                        "end_date": end_date,
                        "description": description,
                    }
                )
                career_end = start_date - pd.Timedelta(days=gap_days)
            source_dates["job_history"] = snapshot
            source_counts["job_history"] = job_count

        if availability["projects"]:
            project_count = int(np.clip(1 + rng.poisson(1.3), 1, 5))
            latest_project_date = None
            for project_index in range(project_count):
                completed_at = _random_date_before(snapshot, rng, 15, 1_825)
                latest_project_date = max(
                    completed_at,
                    latest_project_date or completed_at,
                )
                skill_name = str(rng.choice(true_names))
                task = (
                    str(rng.choice(occupation_tasks))
                    if occupation_tasks
                    else f"Delivered work for {occupation['title']}."
                )
                rows["projects"].append(
                    {
                        "project_id": f"{candidate_id}_project_{project_index + 1:02d}",
                        "candidate_id": candidate_id,
                        "project_title": f"{skill_name} applied project",
                        "description": f"Applied {skill_name} to {task[:220]}",
                        "completed_at": completed_at,
                    }
                )
            source_dates["projects"] = latest_project_date
            source_counts["projects"] = project_count

        if availability["courses"]:
            course_count = min(
                int(np.clip(1 + rng.poisson(2.0), 1, 8)), len(true_skills)
            )
            course_positions = rng.choice(
                len(true_skills), size=course_count, replace=False
            )
            latest_course_date = None
            for course_index, position in enumerate(course_positions, start=1):
                skill_id = true_skills[int(position)][0]
                skill_name = str(lookups["skills"][skill_id]["skill_name"])
                completed_at = _random_date_before(snapshot, rng, 20, 2_190)
                latest_course_date = max(
                    completed_at,
                    latest_course_date or completed_at,
                )
                rows["courses"].append(
                    {
                        "course_id": f"{candidate_id}_course_{course_index:02d}",
                        "candidate_id": candidate_id,
                        "course_name": f"Applied {skill_name}: Foundations and Practice",
                        "skill_id": skill_id,
                        "completed_at": completed_at,
                    }
                )
            source_dates["courses"] = latest_course_date
            source_counts["courses"] = course_count

        if availability["self_reported_skills"]:
            claim_ranges = {
                "terse": (3, 7),
                "typical": (8, 30),
                "exhaustive": (40, config.max_self_reported_skills),
            }
            lower, upper = claim_ranges[claim_style]
            upper = min(upper, config.max_self_reported_skills)
            lower = min(lower, upper)
            desired_claims = int(rng.integers(lower, upper + 1))
            desired_claims = min(desired_claims, len(lookups["all_skill_ids"]))
            target_noise_rate = config.noisy_claim_rate
            if claim_style == "exhaustive":
                target_noise_rate = max(target_noise_rate, 0.30)
            target_true_count = round(desired_claims * (1 - target_noise_rate))
            true_claim_count = min(target_true_count, len(true_ids))
            claimed_true = (
                rng.choice(
                    np.array(sorted(true_ids), dtype=object),
                    size=true_claim_count,
                    replace=False,
                ).tolist()
                if true_claim_count
                else []
            )
            noise_count = min(
                desired_claims - true_claim_count,
                len(lookups["all_skill_ids"]) - len(true_ids),
            )
            claimed_noise = _sample_noise_skills(
                lookups["all_skill_ids"],
                true_ids,
                noise_count,
                rng,
            )
            claims = [(str(skill_id), True) for skill_id in claimed_true]
            claims.extend((str(skill_id), False) for skill_id in claimed_noise)
            rng.shuffle(claims)
            claimed_at = _random_date_before(snapshot, rng, 0, 1_460)
            for claim_index, (skill_id, is_true) in enumerate(claims, start=1):
                skill_name = str(lookups["skills"][skill_id]["skill_name"])
                raw_name, alias_type = _surface_skill_name(skill_name, rng)
                rows["self_reported_skills"].append(
                    {
                        "claim_id": f"{candidate_id}_claim_{claim_index:03d}",
                        "candidate_id": candidate_id,
                        "raw_skill_name": raw_name,
                        "canonical_skill_id": skill_id,
                        "reported_proficiency": int(rng.integers(2, 6)),
                        "claimed_at": claimed_at,
                        "alias_type": alias_type,
                        "oracle_is_true_skill": is_true,
                    }
                )
            source_dates["self_reported_skills"] = claimed_at
            source_counts["self_reported_skills"] = len(claims)

        if availability["resume"]:
            is_stale = rng.random() < config.stale_resume_rate
            age_bounds = (731, 3_650) if is_stale else (0, 730)
            updated_at = _random_date_before(snapshot, rng, *age_bounds)
            displayed_skills = true_names[: min(len(true_names), 18)]
            summary = (
                f"{occupation['title']} with {years_experience} years of experience."
            )
            resume_parts = [summary]
            if task_sample:
                resume_parts.append("Experience: " + " ".join(task_sample))
            resume_parts.append("Skills: " + ", ".join(displayed_skills) + ".")
            rows["resumes"].append(
                {
                    "resume_id": f"{candidate_id}_resume_01",
                    "candidate_id": candidate_id,
                    "resume_text": "\n".join(resume_parts),
                    "updated_at": updated_at,
                }
            )
            source_dates["resume"] = updated_at
            source_counts["resume"] = 1

        for source in SOURCES:
            updated_at = source_dates[source]
            age_days = (
                int((snapshot - updated_at).days)
                if updated_at is not None
                else None
            )
            item_count = source_counts[source]
            rows["source_status"].append(
                {
                    "candidate_id": candidate_id,
                    "source": source,
                    "is_available": item_count > 0,
                    "updated_at": updated_at,
                    "age_days": age_days,
                    "item_count": item_count,
                    "reliability_score": _reliability(
                        source, age_days, item_count
                    ),
                }
            )

    tables = {name: _frame(table_rows, name) for name, table_rows in rows.items()}
    validate_candidate_tables(tables, onet_tables, config)
    return tables


def validate_candidate_tables(
    tables: dict[str, pd.DataFrame],
    onet_tables: dict[str, pd.DataFrame],
    config: GenerationConfig,
) -> None:
    """Validate row-level keys, foreign keys, ranges, and source contracts."""
    candidates = tables["candidates"]
    if len(candidates) != config.n_candidates:
        raise ValueError("Candidate row count does not match configuration")
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("Duplicate candidate IDs found")

    key_contracts = {
        "resumes": ["resume_id"],
        "job_history": ["job_history_id"],
        "projects": ["project_id"],
        "courses": ["course_id"],
        "self_reported_skills": ["claim_id"],
        "candidate_skill_truth": ["candidate_id", "skill_id"],
        "source_status": ["candidate_id", "source"],
    }
    candidate_ids = set(candidates["candidate_id"])
    for table_name, keys in key_contracts.items():
        table = tables[table_name]
        if table.duplicated(keys).any():
            raise ValueError(f"Duplicate keys found in {table_name}")
        unknown = set(table["candidate_id"]) - candidate_ids
        if unknown:
            raise ValueError(f"Unknown candidate IDs found in {table_name}")

    expected_status_rows = config.n_candidates * len(SOURCES)
    if len(tables["source_status"]) != expected_status_rows:
        raise ValueError("Every candidate must have one status row per source")

    occupation_ids = set(onet_tables["occupations"]["onet_soc_code"])
    skill_ids = set(onet_tables["skills"]["skill_id"])
    if not set(candidates["primary_onet_soc_code"]) <= occupation_ids:
        raise ValueError("Unknown primary occupation IDs found")
    if not set(tables["job_history"]["onet_soc_code"]) <= occupation_ids:
        raise ValueError("Unknown job-history occupation IDs found")
    if not set(tables["candidate_skill_truth"]["skill_id"]) <= skill_ids:
        raise ValueError("Unknown truth skill IDs found")
    if not set(tables["courses"]["skill_id"]) <= skill_ids:
        raise ValueError("Unknown course skill IDs found")
    if not set(tables["self_reported_skills"]["canonical_skill_id"]) <= skill_ids:
        raise ValueError("Unknown self-reported skill IDs found")

    reliability = tables["source_status"]["reliability_score"]
    if not reliability.between(0, 1).all():
        raise ValueError("Reliability scores must be between zero and one")
    proficiency = tables["candidate_skill_truth"]["proficiency"]
    if not proficiency.between(1, 5).all():
        raise ValueError("Truth proficiency must be between one and five")


def write_candidate_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Write all generated tables as Parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        print(f"{name}: {len(table):,} rows -> {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--onet-dir", type=Path, default=DEFAULT_ONET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-candidates", type=int)
    return parser.parse_args()


def main() -> None:
    """Generate and persist a complete synthetic candidate snapshot."""
    args = parse_args()
    config = load_config(args.config)
    if args.n_candidates is not None:
        config = replace(config, n_candidates=args.n_candidates)
    onet_tables = load_onet_tables(args.onet_dir)
    tables = generate_candidate_tables(onet_tables, config)
    write_candidate_tables(tables, args.output_dir)


if __name__ == "__main__":
    main()
