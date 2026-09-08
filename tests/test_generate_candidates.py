"""Tests for deterministic, O*NET-grounded candidate generation."""

from __future__ import annotations

import pandas as pd

from talentid.data.generate_candidates import (
    SOURCES,
    GenerationConfig,
    generate_candidate_tables,
)


def mini_onet_tables() -> dict[str, pd.DataFrame]:
    """Build a small in-memory taxonomy so CI needs no downloaded data."""
    codes = ["11-1000.00", "11-2000.00", "15-1000.00", "15-2000.00"]
    occupations = pd.DataFrame(
        {
            "onet_soc_code": codes,
            "title": ["Executive", "Manager", "Analyst", "Engineer"],
            "description": ["Synthetic occupation"] * len(codes),
        }
    )
    job_titles = pd.DataFrame(
        {
            "onet_soc_code": [code for code in codes for _ in range(2)],
            "job_title": [
                title
                for occupation in occupations["title"]
                for title in (occupation, f"Senior {occupation}")
            ],
        }
    )
    tasks = pd.DataFrame(
        {
            "task_id": range(1, 9),
            "onet_soc_code": [code for code in codes for _ in range(2)],
            "task": [f"Complete representative task {index}." for index in range(8)],
        }
    )
    skills = pd.DataFrame(
        {
            "skill_id": [f"skill:{index:03d}" for index in range(30)],
            "skill_name": ["Python", "Microsoft Excel"]
            + [f"Skill {index}" for index in range(2, 30)],
            "skill_type": ["software"] * 20 + ["knowledge"] * 10,
        }
    )
    relationships = []
    for occupation_index, code in enumerate(codes):
        for skill_index in range(18):
            index = (skill_index + 4 * occupation_index) % len(skills)
            relationships.append(
                {
                    "onet_soc_code": code,
                    "skill_id": skills.loc[index, "skill_id"],
                    "skill_name": skills.loc[index, "skill_name"],
                    "skill_type": skills.loc[index, "skill_type"],
                    "importance": 2.0 + (skill_index % 4) * 0.7,
                    "level": 2.0 + (skill_index % 5),
                    "hot_technology": skill_index % 7 == 0,
                    "in_demand": skill_index % 5 == 0,
                    "not_relevant": False,
                }
            )
    return {
        "occupations": occupations,
        "job_titles": job_titles,
        "tasks": tasks,
        "skills": skills,
        "occupation_skills": pd.DataFrame(relationships),
    }


def small_config() -> GenerationConfig:
    return GenerationConfig(
        n_candidates=80,
        seed=12345,
        stale_resume_rate=0.35,
        max_true_skills=15,
        max_self_reported_skills=30,
    )


def test_generation_is_deterministic() -> None:
    onet = mini_onet_tables()
    first = generate_candidate_tables(onet, small_config())
    second = generate_candidate_tables(onet, small_config())

    assert first.keys() == second.keys()
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_candidate_keys_foreign_keys_and_ranges() -> None:
    onet = mini_onet_tables()
    config = small_config()
    tables = generate_candidate_tables(onet, config)
    candidates = tables["candidates"]
    candidate_ids = set(candidates["candidate_id"])
    skill_ids = set(onet["skills"]["skill_id"])

    assert len(candidates) == config.n_candidates
    assert candidates["candidate_id"].is_unique
    assert len(tables["source_status"]) == config.n_candidates * len(SOURCES)
    assert set(tables["candidate_skill_truth"]["candidate_id"]) <= candidate_ids
    assert set(tables["candidate_skill_truth"]["skill_id"]) <= skill_ids
    assert set(tables["self_reported_skills"]["canonical_skill_id"]) <= skill_ids
    assert tables["source_status"]["reliability_score"].between(0, 1).all()
    assert tables["candidate_skill_truth"]["proficiency"].between(1, 5).all()


def test_simulated_quality_problems_are_present() -> None:
    tables = generate_candidate_tables(mini_onet_tables(), small_config())
    status = tables["source_status"]
    claims = tables["self_reported_skills"]

    assert status["is_available"].any()
    assert (~status["is_available"]).any()
    assert status.loc[~status["is_available"], "reliability_score"].eq(0).all()
    assert status.query("source == 'resume' and is_available")["age_days"].gt(730).any()
    assert (~claims["oracle_is_true_skill"]).any()
    assert claims["alias_type"].ne("canonical").any()
    assert set(tables["candidates"]["split"]) == {"train", "validation", "test"}

