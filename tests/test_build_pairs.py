"""Tests for leakage-safe positive and hard-negative pair construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from talentid.training.build_pairs import PairMiningConfig, build_pair_tables


def mini_inputs() -> tuple[
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
]:
    candidate_embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    candidate_index = pd.DataFrame(
        {
            "embedding_row": [0, 1, 2],
            "candidate_id": ["c_train", "c_validation", "c_test"],
            "split": ["train", "validation", "test"],
        }
    )
    skill_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.8, 0.2],
            [0.6, 0.4],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    skill_index = pd.DataFrame(
        {
            "embedding_row": list(range(6)),
            "skill_id": [f"s{index}" for index in range(1, 7)],
        }
    )
    truth = pd.DataFrame(
        {
            "candidate_id": ["c_train", "c_validation", "c_test"],
            "skill_id": ["s1", "s2", "s3"],
            "relevance_score": [1.0, 0.8, 0.7],
            "proficiency": [5.0, 4.0, 3.0],
            "taxonomy_version": ["v1", "v1", "v1"],
        }
    )
    return (
        candidate_embeddings,
        candidate_index,
        skill_embeddings,
        skill_index,
        truth,
    )


def config() -> PairMiningConfig:
    return PairMiningConfig(
        hard_negatives_per_candidate=2,
        random_negatives_per_candidate=1,
        query_batch_size=1,
        seed=17,
    )


def test_test_truth_is_excluded_and_negatives_are_train_only() -> None:
    tables = build_pair_tables(*mini_inputs(), config())
    positives = tables["positive_pairs"]
    negatives = tables["negative_pairs"]
    assert set(positives["split"]) == {"train", "validation"}
    assert "c_test" not in set(positives["candidate_id"])
    assert set(negatives["candidate_id"]) == {"c_train"}
    assert set(negatives["split"]) == {"train"}


def test_negatives_are_unique_and_never_overlap_truth() -> None:
    tables = build_pair_tables(*mini_inputs(), config())
    positives = tables["positive_pairs"]
    negatives = tables["negative_pairs"]
    positive_keys = set(zip(positives["candidate_id"], positives["skill_id"]))
    negative_keys = set(zip(negatives["candidate_id"], negatives["skill_id"]))
    assert not positive_keys & negative_keys
    assert not negatives.duplicated(["candidate_id", "skill_id"]).any()
    assert negatives["negative_type"].value_counts().to_dict() == {
        "hard_retrieval": 2,
        "random": 1,
    }


def test_hard_negatives_follow_baseline_retrieval_order() -> None:
    tables = build_pair_tables(*mini_inputs(), config())
    hard = tables["negative_pairs"].query("negative_type == 'hard_retrieval'")
    hard = hard.sort_values("retrieval_rank")
    assert hard["skill_id"].tolist() == ["s4", "s5"]
    assert hard["retrieval_score"].is_monotonic_decreasing


def test_pair_generation_is_deterministic_and_weights_are_bounded() -> None:
    first = build_pair_tables(*mini_inputs(), config())
    second = build_pair_tables(*mini_inputs(), config())
    pd.testing.assert_frame_equal(first["positive_pairs"], second["positive_pairs"])
    pd.testing.assert_frame_equal(first["negative_pairs"], second["negative_pairs"])
    assert first["positive_pairs"]["sample_weight"].between(0.1, 1.0).all()
    train_weight = first["positive_pairs"].query("candidate_id == 'c_train'")[
        "sample_weight"
    ].iloc[0]
    assert train_weight == 1.0


def test_configuration_rejects_test_leakage() -> None:
    unsafe = PairMiningConfig(positive_splits=("train", "test"))
    try:
        unsafe.validate()
    except ValueError as error:
        assert "test split" in str(error)
    else:
        raise AssertionError("test leakage configuration was accepted")

