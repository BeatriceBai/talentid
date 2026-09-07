"""Integration contracts for the generated O*NET 31.0 Parquet tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "onet" / "31.0"

EXPECTED_ROWS = {
    "occupations": 1_016,
    "job_titles": 66_764,
    "tasks": 18_838,
    "skills": 8_821,
    "occupation_skills": 93_636,
}


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    """Load generated tables, or skip when local O*NET data is unavailable."""
    paths = {
        name: DATA_DIR / f"{name}.parquet"
        for name in EXPECTED_ROWS
    }
    missing = [path for path in paths.values() if not path.exists()]

    if missing:
        pytest.skip(
            "Generated O*NET data is unavailable; run the ingestion pipeline."
        )

    return {
        name: pd.read_parquet(path)
        for name, path in paths.items()
    }


def test_expected_row_counts(tables: dict[str, pd.DataFrame]) -> None:
    actual = {name: len(table) for name, table in tables.items()}
    assert actual == EXPECTED_ROWS


def test_primary_and_composite_keys(tables: dict[str, pd.DataFrame]) -> None:
    key_contracts = {
        "occupations": ["onet_soc_code"],
        "job_titles": ["onet_soc_code", "job_title", "title_type"],
        "tasks": ["task_id"],
        "skills": ["skill_id"],
        "occupation_skills": ["onet_soc_code", "skill_id"],
    }

    for table_name, keys in key_contracts.items():
        table = tables[table_name]
        assert not table[keys].isna().any().any(), table_name
        assert not table.duplicated(keys).any(), table_name


def test_foreign_keys(tables: dict[str, pd.DataFrame]) -> None:
    occupation_ids = set(tables["occupations"]["onet_soc_code"])
    skill_ids = set(tables["skills"]["skill_id"])

    for table_name in ("job_titles", "tasks", "occupation_skills"):
        referenced = set(tables[table_name]["onet_soc_code"])
        assert referenced <= occupation_ids, table_name

    referenced_skills = set(tables["occupation_skills"]["skill_id"])
    assert referenced_skills <= skill_ids


def test_rating_ranges(tables: dict[str, pd.DataFrame]) -> None:
    relationships = tables["occupation_skills"]
    importance = relationships["importance"].dropna()
    level = relationships["level"].dropna()

    assert importance.between(1.0, 5.0).all()
    assert level.between(0.0, 7.0).all()


def test_skill_type_counts(tables: dict[str, pd.DataFrame]) -> None:
    counts = tables["skills"]["skill_type"].value_counts().to_dict()

    assert counts == {
        "software": 8_753,
        "knowledge": 33,
        "transferable": 25,
        "essential": 10,
    }

