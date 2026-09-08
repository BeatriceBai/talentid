"""Tests for frozen text encoding and reliability-aware fusion."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from talentid.embeddings.build_embeddings import (
    EmbeddingConfig,
    build_embedding_artifacts,
    reliability_weighted_fusion,
)


class FakeEncoder:
    """Offline deterministic encoder for CI."""

    dimension = 4

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        del batch_size
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            rows.append([float(value + 1) for value in digest[: self.dimension]])
        return np.asarray(rows, dtype=np.float32)


def mini_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sources = (
        "resume",
        "projects",
        "job_history",
        "courses",
        "self_reported_skills",
    )
    rows = []
    for candidate_id, split in (("cand_2", "test"), ("cand_1", "train")):
        for index, source in enumerate(sources):
            available = candidate_id == "cand_1" or source in {"resume", "job_history"}
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "source": source,
                    "tower_text": f"{candidate_id} {source}" if available else "",
                    "is_available": available,
                    "reliability_score": (index + 1) / 10 if available else 0.0,
                    "split": split,
                    "feature_version": "features-v1",
                }
            )
    towers = pd.DataFrame(rows).sample(frac=1, random_state=7)
    skills = pd.DataFrame(
        {
            "skill_id": ["skill:2", "skill:1"],
            "skill_text": ["Planning", "Python programming"],
            "feature_version": ["features-v1", "features-v1"],
        }
    )
    candidates = pd.DataFrame(
        {
            "candidate_id": ["cand_1", "cand_2"],
            "split": ["train", "test"],
        }
    )
    return towers, skills, candidates


def test_reliability_weighted_fusion_math() -> None:
    vectors = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    weights = np.asarray([1.0, 3.0, 2.0], dtype=np.float32)
    fused, ids = reliability_weighted_fusion(
        vectors, weights, ["cand_a", "cand_a", "cand_b"]
    )
    assert ids == ["cand_a", "cand_b"]
    np.testing.assert_allclose(fused[0], [0.31622776, 0.9486833], rtol=1e-6)
    np.testing.assert_allclose(
        fused[1], [0.70710677, 0.70710677], rtol=1e-6
    )


def test_missing_towers_are_zero_and_available_vectors_are_normalized() -> None:
    towers, skills, candidates = mini_features()
    artifacts = build_embedding_artifacts(
        towers,
        skills,
        candidates,
        FakeEncoder(),
        EmbeddingConfig(),
    )
    index = artifacts["candidate_tower_embedding_index"]
    vectors = artifacts["candidate_tower_embeddings"]
    missing_rows = index.loc[~index["is_available"], "embedding_row"]
    available_rows = index.loc[index["is_available"], "embedding_row"]
    assert np.count_nonzero(vectors[missing_rows]) == 0
    np.testing.assert_allclose(
        np.linalg.norm(vectors[available_rows], axis=1), 1.0, rtol=1e-6
    )


def test_arrays_and_indices_are_deterministically_aligned() -> None:
    towers, skills, candidates = mini_features()
    artifacts = build_embedding_artifacts(
        towers,
        skills,
        candidates,
        FakeEncoder(),
        EmbeddingConfig(),
    )
    tower_index = artifacts["candidate_tower_embedding_index"]
    skill_index = artifacts["skill_embedding_index"]
    candidate_index = artifacts["candidate_embedding_index"]

    assert tower_index["embedding_row"].tolist() == list(range(10))
    assert tower_index.iloc[0]["candidate_id"] == "cand_1"
    assert tower_index.iloc[0]["source"] == "resume"
    assert skill_index["skill_id"].tolist() == ["skill:1", "skill:2"]
    assert candidate_index["candidate_id"].tolist() == ["cand_1", "cand_2"]
    assert artifacts["candidate_embeddings"].shape == (2, 4)
    np.testing.assert_allclose(
        np.linalg.norm(artifacts["candidate_embeddings"], axis=1),
        1.0,
        rtol=1e-6,
    )


def test_encoder_input_has_role_prefixes() -> None:
    class RecordingEncoder(FakeEncoder):
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
            self.calls.append(list(texts))
            return super().encode(texts, batch_size)

    towers, skills, candidates = mini_features()
    encoder = RecordingEncoder()
    build_embedding_artifacts(
        towers,
        skills,
        candidates,
        encoder,
        EmbeddingConfig(),
    )
    assert encoder.calls[0][0].startswith("Candidate resume evidence: ")
    assert encoder.calls[1] == ["Skill: Python programming", "Skill: Planning"]
