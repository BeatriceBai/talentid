"""Tests for recall-preserving hybrid candidate generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from talentid.retrieval.build_hybrid_pool import (
    HybridRetrievalConfig,
    build_hybrid_retrieval,
    validate_visible_truth,
)


def _learned() -> pd.DataFrame:
    rows = []
    for candidate_id, skills in {"c1": ["s1", "s4"], "c2": ["s4", "s1"]}.items():
        for rank, skill_id in enumerate(skills, start=1):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "split": "validation",
                    "engine": "faiss_flat",
                    "rank": rank,
                    "skill_id": skill_id,
                    "score": 1.0 / rank,
                }
            )
    return pd.DataFrame(rows)


def _inputs() -> dict[str, object]:
    skill_embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    skill_index = pd.DataFrame(
        {"embedding_row": range(4), "skill_id": ["s1", "s2", "s3", "s4"]}
    )
    tower_embeddings = np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    tower_index = pd.DataFrame(
        {
            "embedding_row": [0, 1],
            "candidate_id": ["c1", "c2"],
            "source": ["resume", "projects"],
            "is_available": [True, True],
            "reliability_score": [1.0, 1.0],
        }
    )
    candidate_index = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2"],
            "primary_onet_soc_code": ["o1", "o2"],
            "split": ["validation", "validation"],
        }
    )
    occupation_skills = pd.DataFrame(
        {
            "onet_soc_code": ["o1", "o2"],
            "skill_id": ["s2", "s3"],
            "importance": [5.0, 5.0],
            "level": [7.0, 7.0],
            "hot_technology": [False, False],
            "in_demand": [False, False],
            "not_relevant": [False, False],
        }
    )
    positive_pairs = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2"],
            "skill_id": ["s2", "s3"],
            "split": ["validation", "validation"],
        }
    )
    return {
        "learned_predictions": _learned(),
        "tower_embeddings": tower_embeddings,
        "tower_index": tower_index,
        "skill_embeddings": skill_embeddings,
        "skill_index": skill_index,
        "candidate_index": candidate_index,
        "occupation_skills": occupation_skills,
        "positive_pairs": positive_pairs,
    }


def test_hybrid_pool_preserves_dense_candidates_and_adds_relevant_skills() -> None:
    config = HybridRetrievalConfig(
        learned_k=2,
        pool_size=3,
        tower_k=2,
        occupation_k=1,
        evaluation_cutoffs=(2, 3),
    )
    results = build_hybrid_retrieval(**_inputs(), config=config)
    predictions = results["hybrid_retrieval_predictions"]
    metrics = results["hybrid_retrieval_metrics"].set_index("stage")
    assert predictions.groupby("candidate_id").size().eq(3).all()
    protected = predictions[predictions["is_learned_protected"]]
    assert protected.groupby("candidate_id").size().eq(2).all()
    assert metrics.loc["learned_top2", "truth_pair_recall"] == 0.0
    assert metrics.loc["hybrid_top3", "truth_pair_recall"] == 1.0


def test_hybrid_pool_is_deterministic() -> None:
    config = HybridRetrievalConfig(
        learned_k=2,
        pool_size=3,
        tower_k=2,
        occupation_k=1,
        evaluation_cutoffs=(2, 3),
    )
    first = build_hybrid_retrieval(**_inputs(), config=config)
    second = build_hybrid_retrieval(**_inputs(), config=config)
    pd.testing.assert_frame_equal(
        first["hybrid_retrieval_predictions"],
        second["hybrid_retrieval_predictions"],
    )


def test_test_labels_and_test_configuration_are_rejected() -> None:
    with pytest.raises(ValueError, match="only validation"):
        HybridRetrievalConfig(split="test").validate()
    exposed = pd.DataFrame(
        {"candidate_id": ["hidden"], "skill_id": ["s1"], "split": ["test"]}
    )
    with pytest.raises(ValueError, match="sealed test"):
        validate_visible_truth(exposed, "validation")
