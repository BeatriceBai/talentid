"""Tests for constrained evidence- and diversity-aware reranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from talentid.reranking.rerank_skills import (
    CandidatePools,
    RerankerConfig,
    assemble_candidate_pools,
    build_prediction_table,
    rerank_pools,
    tune_reranker,
    validate_visible_pairs,
)


def _pools() -> tuple[CandidatePools, pd.DataFrame]:
    skill_ids = np.asarray(["s1", "s2", "s3", "s4"])
    pools = CandidatePools(
        candidate_ids=["c1", "c2"],
        skill_ids=skill_ids,
        skill_positions=np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]]),
        retrieval_scores=np.asarray(
            [[1.0, 0.9, 0.8, 0.7], [1.0, 0.9, 0.8, 0.7]], dtype=np.float32
        ),
        evidence_scores=np.asarray(
            [[0.1, 1.0, 0.2, 0.0], [0.1, 0.2, 1.0, 0.0]], dtype=np.float32
        ),
        popularity_scores=np.asarray(
            [[1.0, 0.2, 0.1, 0.0], [1.0, 0.2, 0.1, 0.0]], dtype=np.float32
        ),
        pool_similarities=np.asarray(
            [
                [
                    [1.0, 0.9, 0.0, 0.0],
                    [0.9, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.1],
                    [0.0, 0.0, 0.1, 1.0],
                ],
                [
                    [1.0, 0.9, 0.0, 0.0],
                    [0.9, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.1],
                    [0.0, 0.0, 0.1, 1.0],
                ],
            ],
            dtype=np.float32,
        ),
    )
    truth = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2"],
            "skill_id": ["s2", "s3"],
            "split": ["validation", "validation"],
            "relevance_score": [1.0, 1.0],
        }
    )
    return pools, truth


def test_evidence_and_popularity_change_the_ranking() -> None:
    pools, _ = _pools()
    baseline, _ = rerank_pools(pools, 2, 0.0, 0.0, 0.0)
    reranked, _ = rerank_pools(pools, 2, 0.5, 0.2, 0.1)
    assert baseline[:, 0].tolist() == [0, 0]
    assert reranked[:, 0].tolist() == [1, 2]


def test_tuning_respects_ndcg_floor_and_builds_predictions() -> None:
    pools, truth = _pools()
    config = RerankerConfig(
        candidate_pool_size=4,
        output_k=2,
        maximum_ndcg_drop=0.0,
        evidence_weights=(0.5,),
        popularity_penalties=(0.2,),
        diversity_penalties=(0.1,),
    )
    trials, selected, scores = tune_reranker(pools, truth, config)
    chosen = trials[trials["is_selected"]].iloc[0]
    assert chosen["ndcg"] >= chosen["ndcg_floor"]
    predictions = build_prediction_table(pools, selected, scores, truth, config)
    assert len(predictions) == 4
    assert predictions.groupby("candidate_id").size().eq(2).all()


def test_reranking_is_deterministic() -> None:
    pools, _ = _pools()
    first = rerank_pools(pools, 3, 0.3, 0.1, 0.05)
    second = rerank_pools(pools, 3, 0.3, 0.1, 0.05)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])


def test_candidate_pool_assembly_aligns_tower_evidence() -> None:
    sources = (
        "resume",
        "projects",
        "job_history",
        "courses",
        "self_reported_skills",
    )
    predictions = pd.DataFrame(
        {
            "candidate_id": ["c1"] * 4,
            "split": ["validation"] * 4,
            "engine": ["faiss_flat"] * 4,
            "rank": [1, 2, 3, 4],
            "skill_id": ["s1", "s2", "s3", "s4"],
            "score": [1.0, 0.8, 0.2, 0.0],
        }
    )
    candidate_index = pd.DataFrame(
        {"embedding_row": [0], "candidate_id": ["c1"], "split": ["validation"]}
    )
    skill_vectors = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    skill_index = pd.DataFrame(
        {"embedding_row": range(4), "skill_id": ["s1", "s2", "s3", "s4"]}
    )
    tower_index = pd.DataFrame(
        {
            "embedding_row": range(5),
            "candidate_id": ["c1"] * 5,
            "source": sources,
            "is_available": [True] * 5,
        }
    )
    gates = pd.DataFrame(
        {
            "candidate_id": ["c1"] * 5,
            "source": sources,
            "gate_weight": [0.2] * 5,
        }
    )
    positives = pd.DataFrame(
        {
            "candidate_id": ["c1"],
            "skill_id": ["s1"],
            "split": ["validation"],
            "relevance_score": [1.0],
        }
    )
    config = RerankerConfig(candidate_pool_size=4, output_k=2)
    pools, truth = assemble_candidate_pools(
        predictions,
        candidate_index,
        skill_vectors,
        skill_index,
        skill_vectors,
        skill_index,
        np.tile([[1.0, 0.0]], (5, 1)),
        tower_index,
        gates,
        positives,
        config,
    )
    assert pools.evidence_scores.shape == (1, 4)
    np.testing.assert_allclose(pools.evidence_scores[0], [1.0, 0.0, -1.0, 0.0])
    assert truth["candidate_id"].tolist() == ["c1"]


def test_test_labels_and_test_configuration_are_rejected() -> None:
    with pytest.raises(ValueError, match="only validation"):
        RerankerConfig(split="test").validate()
    exposed = pd.DataFrame(
        {
            "candidate_id": ["hidden"],
            "skill_id": ["s1"],
            "split": ["test"],
            "relevance_score": [1.0],
        }
    )
    with pytest.raises(ValueError, match="sealed test"):
        validate_visible_pairs(exposed)
