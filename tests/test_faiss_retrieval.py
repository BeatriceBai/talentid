"""Tests for exact and approximate learned-embedding retrieval."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")

from talentid.evaluation.evaluate_learned_retrieval import (
    LearnedRetrievalConfig,
    evaluate_learned_retrieval,
    write_results,
)
from talentid.evaluation.evaluate_retrieval import retrieve_top_k
from talentid.retrieval.faiss_index import (
    build_index,
    load_index,
    neighbor_recall,
    save_index,
    search_index,
)


def test_faiss_flat_matches_numpy_exact() -> None:
    skills = np.asarray([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    queries = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    expected, _ = retrieve_top_k(queries, skills, max_k=3, batch_size=1)
    index = build_index(skills, index_type="flat")
    observed, scores = search_index(index, queries, 3)
    assert neighbor_recall(expected, observed) == 1.0
    np.testing.assert_allclose(scores[:, 0], 1.0)


def test_hnsw_recall_and_index_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(7)
    skills = rng.normal(size=(250, 16)).astype(np.float32)
    queries = skills[:25] + rng.normal(scale=0.01, size=(25, 16)).astype(np.float32)
    flat = build_index(skills, index_type="flat")
    hnsw = build_index(
        skills,
        index_type="hnsw",
        hnsw_m=16,
        ef_construction=100,
        ef_search=100,
    )
    expected, _ = search_index(flat, queries, 10)
    observed, _ = search_index(hnsw, queries, 10)
    assert neighbor_recall(expected, observed) >= 0.99
    path = tmp_path / "skills.faiss"
    save_index(hnsw, path)
    restored = load_index(path, ef_search=100)
    restored_neighbors, _ = search_index(restored, queries, 10)
    np.testing.assert_array_equal(restored_neighbors, observed)


def _mini_inputs() -> tuple[
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    candidate_embeddings = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    candidate_index = pd.DataFrame(
        {
            "embedding_row": [0, 1],
            "candidate_id": ["c1", "c2"],
            "split": ["validation", "validation"],
        }
    )
    skill_embeddings = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    skill_index = pd.DataFrame(
        {
            "embedding_row": [0, 1, 2, 3],
            "skill_id": ["s1", "s2", "s3", "s4"],
        }
    )
    positives = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2"],
            "skill_id": ["s1", "s2"],
            "split": ["validation", "validation"],
            "relevance_score": [1.0, 1.0],
        }
    )
    baseline_rows = []
    for cutoff in (1, 2):
        baseline_rows.append(
            {
                "strategy": "reliability_weighted",
                "split": "validation",
                "k": cutoff,
                "catalog_coverage": 0.25,
                "precision": 0.5,
                "recall": 0.5,
                "ndcg": 0.5,
                "hit_rate": 0.5,
                "mrr": 0.5,
            }
        )
    return (
        candidate_embeddings,
        candidate_index,
        skill_embeddings,
        skill_index,
        positives,
        pd.DataFrame(baseline_rows),
    )


def test_end_to_end_learned_retrieval(tmp_path) -> None:
    config = LearnedRetrievalConfig(
        cutoffs=(1, 2),
        hnsw_m=2,
        ef_construction=20,
        ef_search=20,
        minimum_recall_at_max_k=0.5,
    )
    results, indexes = evaluate_learned_retrieval(*_mini_inputs(), config)
    at_1 = results["learned_retrieval_metrics"]
    at_1 = at_1[at_1["k"].eq(1)]
    assert at_1["recall"].eq(1.0).all()
    assert results["ann_fidelity"]["neighbor_recall"].min() >= 0.5
    assert len(results["baseline_comparison"]) == 12
    assert indexes["faiss_flat"].ntotal == 4
    assert indexes["faiss_hnsw"].ntotal == 4
    output_dir = tmp_path / "evaluation"
    index_dir = tmp_path / "indexes"
    write_results(results, indexes, [], output_dir, index_dir, config)
    assert (index_dir / "flat.faiss").exists()
    assert (index_dir / "hnsw.faiss").exists()
    assert (output_dir / "learned_retrieval_manifest.json").exists()


def test_test_split_and_test_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="only validation"):
        LearnedRetrievalConfig(split="test").validate()
    inputs = list(_mini_inputs())
    inputs[4] = pd.concat(
        [
            inputs[4],
            pd.DataFrame(
                {
                    "candidate_id": ["hidden"],
                    "skill_id": ["s1"],
                    "split": ["test"],
                    "relevance_score": [1.0],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="sealed test"):
        evaluate_learned_retrieval(
            *inputs,
            LearnedRetrievalConfig(cutoffs=(1, 2), minimum_recall_at_max_k=0.5),
        )
