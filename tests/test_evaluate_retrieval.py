"""Tests for deterministic retrieval and ranking metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from talentid.evaluation.evaluate_retrieval import (
    RetrievalConfig,
    build_uniform_candidate_embeddings,
    compute_candidate_metrics,
    evaluate_retrieval,
    retrieve_top_k,
)


def test_retrieve_top_k_returns_cosine_order_with_stable_ties() -> None:
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    skills = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    indices, scores = retrieve_top_k(queries, skills, max_k=3, batch_size=1)
    assert indices.tolist() == [[0, 1, 2]]
    np.testing.assert_allclose(scores, [[1.0, 1.0, 0.0]])


def test_candidate_metrics_match_hand_calculation() -> None:
    retrieved = np.asarray([["s1", "s3", "s2"]])
    truth = pd.DataFrame(
        {
            "candidate_id": ["c1", "c1"],
            "skill_id": ["s1", "s2"],
            "relevance_score": [1.0, 0.5],
        }
    )
    metrics = compute_candidate_metrics(retrieved, truth, ["c1"], (1, 3))
    at_1 = metrics[metrics["k"].eq(1)].iloc[0]
    at_3 = metrics[metrics["k"].eq(3)].iloc[0]
    assert at_1["precision"] == 1.0
    assert at_1["recall"] == 0.5
    assert at_1["mrr"] == 1.0
    assert at_3["precision"] == 2 / 3
    assert at_3["recall"] == 1.0
    assert 0 < at_3["ndcg"] < 1


def _mini_inputs() -> tuple[
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
]:
    sources = (
        "resume",
        "projects",
        "job_history",
        "courses",
        "self_reported_skills",
    )
    candidate_embeddings = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    candidate_index = pd.DataFrame(
        {
            "embedding_row": [0, 1],
            "candidate_id": ["c1", "c2"],
            "split": ["validation", "validation"],
        }
    )
    tower_rows = []
    tower_vectors = []
    for candidate_id, vector in (("c1", [1, 0]), ("c2", [0, 1])):
        for source in sources:
            tower_rows.append(
                {
                    "embedding_row": len(tower_rows),
                    "candidate_id": candidate_id,
                    "source": source,
                    "is_available": True,
                }
            )
            tower_vectors.append(vector)
    tower_index = pd.DataFrame(tower_rows)
    skill_embeddings = np.asarray([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    skill_index = pd.DataFrame(
        {"embedding_row": [0, 1, 2], "skill_id": ["s1", "s2", "s3"]}
    )
    truth = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2"],
            "skill_id": ["s1", "s2"],
            "relevance_score": [1.0, 1.0],
        }
    )
    return (
        candidate_embeddings,
        candidate_index,
        np.asarray(tower_vectors, dtype=np.float32),
        tower_index,
        skill_embeddings,
        skill_index,
        truth,
    )


def test_uniform_fusion_and_end_to_end_evaluation() -> None:
    inputs = _mini_inputs()
    uniform = build_uniform_candidate_embeddings(
        inputs[2], inputs[3], ["c1", "c2"]
    )
    np.testing.assert_allclose(uniform, inputs[0])

    results = evaluate_retrieval(
        *inputs,
        RetrievalConfig(cutoffs=(1, 2), query_batch_size=1),
    )
    metrics = results["retrieval_metrics"]
    at_1 = metrics[metrics["k"].eq(1)]
    assert len(at_1) == 2
    assert at_1["recall"].eq(1.0).all()
    assert at_1["ndcg"].eq(1.0).all()
    assert len(results["retrieval_predictions"]) == 2 * 2 * 2


def test_default_configuration_does_not_open_the_test_split() -> None:
    config = RetrievalConfig()
    assert config.splits == ("validation",)
    assert "test" not in config.splits

