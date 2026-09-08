"""Tests for leakage-safe candidate and skill feature construction."""

from __future__ import annotations

import pandas as pd

from talentid.features.build_features import (
    SOURCES,
    FeatureConfig,
    build_feature_tables,
    normalize_text,
)


def mini_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    candidates = pd.DataFrame(
        {
            "candidate_id": ["cand_1", "cand_2"],
            "primary_onet_soc_code": ["15-0000.00", "11-0000.00"],
            "years_experience": [8, 2],
            "split": ["train", "test"],
            "taxonomy_version": ["taxonomy-v1", "taxonomy-v1"],
        }
    )
    source_status_rows = []
    availability = {
        "cand_1": {source: True for source in SOURCES},
        "cand_2": {
            "resume": True,
            "projects": False,
            "job_history": True,
            "courses": False,
            "self_reported_skills": True,
        },
    }
    item_counts = {
        "resume": 1,
        "projects": 1,
        "job_history": 1,
        "courses": 1,
        "self_reported_skills": 1,
    }
    for candidate_id in candidates["candidate_id"]:
        for source in SOURCES:
            is_available = availability[candidate_id][source]
            source_status_rows.append(
                {
                    "candidate_id": candidate_id,
                    "source": source,
                    "is_available": is_available,
                    "updated_at": pd.Timestamp("2026-08-01")
                    if is_available
                    else pd.NaT,
                    "age_days": 31 if is_available else pd.NA,
                    "item_count": item_counts[source] if is_available else 0,
                    "reliability_score": 0.8 if is_available else 0.0,
                }
            )

    candidate_tables = {
        "candidates": candidates,
        "resumes": pd.DataFrame(
            {
                "resume_id": ["r1", "r2"],
                "candidate_id": ["cand_1", "cand_2"],
                "resume_text": ["Python\n engineer", "Operations manager"],
                "updated_at": pd.to_datetime(["2026-08-01", "2026-08-01"]),
            }
        ),
        "job_history": pd.DataFrame(
            {
                "job_history_id": ["j1", "j2"],
                "candidate_id": ["cand_1", "cand_2"],
                "onet_soc_code": ["SECRET_OCCUPATION", "SECRET_OCCUPATION"],
                "job_title": ["Engineer", "Manager"],
                "start_date": pd.to_datetime(["2020-01-01", "2024-01-01"]),
                "end_date": [pd.NaT, pd.NaT],
                "description": ["Built models", "Led operations"],
            }
        ),
        "projects": pd.DataFrame(
            {
                "project_id": ["p1"],
                "candidate_id": ["cand_1"],
                "project_title": ["Forecasting"],
                "description": ["Built a demand model"],
                "completed_at": pd.to_datetime(["2026-07-01"]),
            }
        ),
        "courses": pd.DataFrame(
            {
                "course_id": ["c1"],
                "candidate_id": ["cand_1"],
                "course_name": ["Applied machine learning"],
                "skill_id": ["SECRET_COURSE_SKILL_ID"],
                "completed_at": pd.to_datetime(["2025-01-01"]),
            }
        ),
        "self_reported_skills": pd.DataFrame(
            {
                "claim_id": ["s1", "s2"],
                "candidate_id": ["cand_1", "cand_2"],
                "raw_skill_name": ["Python", "planning"],
                "canonical_skill_id": ["SECRET_CANONICAL_ID", "SECRET_ID_2"],
                "reported_proficiency": [5, 3],
                "claimed_at": pd.to_datetime(["2026-08-01", "2026-08-01"]),
                "oracle_is_true_skill": [True, False],
            }
        ),
        "source_status": pd.DataFrame(source_status_rows),
        "candidate_skill_truth": pd.DataFrame(
            {
                "candidate_id": ["cand_1"],
                "skill_id": ["SECRET_TRUTH_ID"],
            }
        ),
    }
    onet_tables = {
        "skills": pd.DataFrame(
            {
                "skill_id": ["skill:1", "skill:2"],
                "skill_name": ["Python", "Planning"],
                "skill_type": ["software", "essential"],
                "description": ["Programming language", "Developing plans"],
                "category_name": ["Development software", pd.NA],
            }
        )
    }
    return candidate_tables, onet_tables


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  Python\n\tengineering  ") == "Python engineering"
    assert normalize_text(pd.NA) == ""


def test_builds_exact_candidate_source_grid_and_missing_rows() -> None:
    candidate_tables, onet_tables = mini_inputs()
    config = FeatureConfig(taxonomy_version="taxonomy-v1")
    features = build_feature_tables(candidate_tables, onet_tables, config)
    towers = features["candidate_tower_features"]

    assert len(towers) == 2 * len(SOURCES)
    assert not towers.duplicated(["candidate_id", "source"]).any()
    missing = towers[~towers["is_available"]]
    assert missing["tower_text"].eq("").all()
    assert missing["reliability_score"].eq(0).all()
    assert set(features["candidate_index"]["split"]) == {"train", "test"}


def test_oracle_and_canonical_identifiers_do_not_leak_into_model_text() -> None:
    candidate_tables, onet_tables = mini_inputs()
    config = FeatureConfig(taxonomy_version="taxonomy-v1")
    towers = build_feature_tables(candidate_tables, onet_tables, config)[
        "candidate_tower_features"
    ]
    combined_text = " ".join(towers["tower_text"])

    assert "SECRET_CANONICAL_ID" not in combined_text
    assert "SECRET_COURSE_SKILL_ID" not in combined_text
    assert "SECRET_OCCUPATION" not in combined_text
    assert "SECRET_TRUTH_ID" not in combined_text
    assert "canonical_skill_id" not in towers.columns
    assert "oracle_is_true_skill" not in towers.columns


def test_character_budgets_and_quality_flags() -> None:
    candidate_tables, onet_tables = mini_inputs()
    candidate_tables["resumes"].loc[0, "resume_text"] = "word " * 100
    candidate_tables["source_status"].loc[
        lambda frame: frame["candidate_id"].eq("cand_1")
        & frame["source"].eq("resume"),
        "age_days",
    ] = 900
    config = FeatureConfig(
        taxonomy_version="taxonomy-v1",
        max_characters_resume=80,
    )
    towers = build_feature_tables(candidate_tables, onet_tables, config)[
        "candidate_tower_features"
    ]
    resume = towers.query("candidate_id == 'cand_1' and source == 'resume'").iloc[0]

    assert resume["character_count"] <= 82
    assert bool(resume["is_stale"])


def test_current_job_is_ordered_before_older_jobs() -> None:
    candidate_tables, onet_tables = mini_inputs()
    older_job = candidate_tables["job_history"].iloc[[0]].copy()
    older_job["job_history_id"] = "j0"
    older_job["job_title"] = "Older role"
    older_job["start_date"] = pd.Timestamp("2017-01-01")
    older_job["end_date"] = pd.Timestamp("2019-12-31")
    candidate_tables["job_history"].loc[
        len(candidate_tables["job_history"])
    ] = older_job.iloc[0]
    status_mask = candidate_tables["source_status"]["candidate_id"].eq(
        "cand_1"
    ) & candidate_tables["source_status"]["source"].eq("job_history")
    candidate_tables["source_status"].loc[status_mask, "item_count"] = 2

    towers = build_feature_tables(
        candidate_tables,
        onet_tables,
        FeatureConfig(taxonomy_version="taxonomy-v1"),
    )["candidate_tower_features"]
    text = towers.query(
        "candidate_id == 'cand_1' and source == 'job_history'"
    )["tower_text"].iloc[0]

    assert text.index("Role: Engineer") < text.index("Role: Older role")
