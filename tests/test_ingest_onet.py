"""Fast unit tests for the O*NET ingestion helpers."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from talentid.data import ingest_onet


def test_stable_software_id_is_normalized_and_deterministic() -> None:
    expected = ingest_onet.stable_software_id("Python")

    assert ingest_onet.stable_software_id(" python ") == expected
    assert ingest_onet.stable_software_id("PYTHON") == expected
    assert ingest_onet.stable_software_id("C++") != (
        ingest_onet.stable_software_id("C#")
    )


def test_concat_frames_preserves_requested_schema_without_warning() -> None:
    left = pd.DataFrame(
        {
            "id": [1],
            "left_only": pd.Series([pd.NA], dtype="string"),
        }
    )
    right = pd.DataFrame(
        {
            "id": [2],
            "left_only": pd.Series(["value"], dtype="string"),
            "right_only": pd.Series(["present"], dtype="string"),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = ingest_onet.concat_frames(
            [left, right],
            columns=["id", "left_only", "right_only"],
        )

    assert result.columns.tolist() == ["id", "left_only", "right_only"]
    assert result.shape == (2, 3)
    assert result.loc[1, "right_only"] == "present"


def test_missing_suppression_flag_does_not_drop_valid_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = pd.DataFrame(
        {
            "O*NET-SOC Code": ["15-0000.00"] * 4,
            "Element ID": ["2.A.1", "2.A.1", "2.A.2", "2.A.2"],
            "Element Name": ["Valid Skill"] * 2 + ["Suppressed Skill"] * 2,
            "Scale ID": ["IM", "LV", "IM", "LV"],
            "Data Value": [4.0, 5.0, 3.0, 4.0],
            "Recommend Suppress": [pd.NA, "N", "Y", "Y"],
            "Not Relevant": [pd.NA, "N", pd.NA, "N"],
            "Date": ["08/2026"] * 4,
            "Domain Source": ["Analyst"] * 4,
        }
    )
    monkeypatch.setattr(
        ingest_onet,
        "read_source",
        lambda _filename: raw.copy(),
    )

    result = ingest_onet.build_rated_skills(
        "ignored.xlsx",
        "essential",
    )

    assert result["skill_name"].tolist() == ["Valid Skill"]
    assert result.loc[0, "importance"] == pytest.approx(4.0)
    assert result.loc[0, "level"] == pytest.approx(5.0)
    assert result.loc[0, "skill_id"] == "onet:2.A.1"

