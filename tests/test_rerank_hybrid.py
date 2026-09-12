"""Tests for hybrid-pool reranking contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from talentid.reranking.rerank_hybrid import validate_hybrid_predictions


def _predictions() -> pd.DataFrame:
    rows = []
    for candidate_id in ("c1", "c2"):
        for rank in range(1, 201):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "split": "validation",
                    "engine": "hybrid_rrf",
                    "rank": rank,
                    "skill_id": f"s{rank}",
                    "score": 1.0 / rank,
                    "is_learned_protected": rank <= 100,
                }
            )
    return pd.DataFrame(rows)


def test_hybrid_adapter_preserves_rows_and_changes_only_engine() -> None:
    original = _predictions()
    adapted = validate_hybrid_predictions(original, split="validation", pool_size=200)
    assert len(adapted) == len(original)
    assert set(adapted["engine"]) == {"faiss_flat"}
    assert set(original["engine"]) == {"hybrid_rrf"}
    pd.testing.assert_series_equal(adapted["rank"], original["rank"])
    pd.testing.assert_series_equal(adapted["skill_id"], original["skill_id"])


def test_incomplete_hybrid_pool_is_rejected() -> None:
    incomplete = _predictions().query("not (candidate_id == 'c1' and rank == 200)")
    with pytest.raises(ValueError, match="ranks are incomplete"):
        validate_hybrid_predictions(incomplete, split="validation", pool_size=200)


def test_learned_top100_contract_is_required() -> None:
    broken = _predictions()
    broken.loc[
        (broken["candidate_id"] == "c1") & (broken["rank"] == 100),
        "is_learned_protected",
    ] = False
    with pytest.raises(ValueError, match="learned top-100 contract"):
        validate_hybrid_predictions(broken, split="validation", pool_size=200)


def test_wrong_engine_is_rejected() -> None:
    wrong = _predictions()
    wrong["engine"] = "faiss_flat"
    with pytest.raises(ValueError, match="engine=hybrid_rrf"):
        validate_hybrid_predictions(wrong, split="validation", pool_size=200)
