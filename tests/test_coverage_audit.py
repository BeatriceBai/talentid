"""Tests for coverage and candidate-pool ceiling diagnostics."""

from __future__ import annotations

import pandas as pd
import pytest

from talentid.evaluation.audit_coverage import (
    CoverageAuditConfig,
    audit_coverage,
    validate_visible_truth,
)


def _prediction_table(
    candidate_skills: dict[str, list[str]],
    *,
    strategy_column: str | None = None,
    strategy_value: str | None = None,
) -> pd.DataFrame:
    rows = []
    for candidate_id, skills in candidate_skills.items():
        for rank, skill_id in enumerate(skills, start=1):
            row = {
                "candidate_id": candidate_id,
                "split": "validation",
                "rank": rank,
                "skill_id": skill_id,
            }
            if strategy_column:
                row[strategy_column] = strategy_value
            rows.append(row)
    return pd.DataFrame(rows)


def _inputs() -> tuple[pd.DataFrame, ...]:
    catalog = pd.DataFrame(
        {
            "skill_id": ["s1", "s2", "s3", "s4", "s5"],
            "skill_name": ["one", "two", "three", "four", "five"],
            "skill_type": ["core", "core", "software", "software", "software"],
        }
    )
    truth = pd.DataFrame(
        {
            "candidate_id": ["c1", "c1", "c2"],
            "skill_id": ["s1", "s2", "s3"],
            "split": ["validation"] * 3,
        }
    )
    frozen = _prediction_table(
        {"c1": ["s1", "s4"], "c2": ["s4", "s5"]},
        strategy_column="strategy",
        strategy_value="reliability_weighted",
    )
    learned = _prediction_table(
        {"c1": ["s1", "s4", "s2"], "c2": ["s3", "s4", "s5"]},
        strategy_column="engine",
        strategy_value="faiss_flat",
    )
    reranked = _prediction_table({"c1": ["s1", "s2"], "c2": ["s3", "s5"]})
    return catalog, truth, frozen, learned, reranked


def test_audit_separates_top30_from_pool_ceiling() -> None:
    config = CoverageAuditConfig(
        retrieval_k=2,
        pool_k=3,
        minimum_pool_pair_recall=0.9,
        minimum_pool_truth_skill_coverage=0.9,
        minimum_reranked_truth_skill_coverage=0.9,
    )
    results = audit_coverage(*_inputs(), config)
    summary = results["coverage_summary"].set_index("stage")
    assert summary.loc["learned_top100_pool", "truth_pair_recall"] == 1.0
    assert summary.loc["reranked_top30", "truth_skill_coverage"] == 1.0
    assert results["recommendation"] == "llm_judge"


def test_audit_recommends_hybrid_when_pool_misses_truth() -> None:
    inputs = list(_inputs())
    inputs[3] = _prediction_table(
        {"c1": ["s4", "s5", "s1"], "c2": ["s4", "s5", "s1"]},
        strategy_column="engine",
        strategy_value="faiss_flat",
    )
    results = audit_coverage(
        *inputs,
        CoverageAuditConfig(retrieval_k=2, pool_k=3),
    )
    assert results["recommendation"] == "hybrid_candidate_generation"


def test_test_labels_and_test_configuration_are_rejected() -> None:
    with pytest.raises(ValueError, match="only validation"):
        CoverageAuditConfig(split="test").validate()
    exposed = pd.DataFrame(
        {"candidate_id": ["hidden"], "skill_id": ["s1"], "split": ["test"]}
    )
    with pytest.raises(ValueError, match="sealed test"):
        validate_visible_truth(exposed)
