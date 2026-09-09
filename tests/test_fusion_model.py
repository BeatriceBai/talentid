"""Tests for reliability-aware neural fusion and training data alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from talentid.modeling.fusion import (
    ReliabilityAwareFusion,
    weighted_contrastive_loss,
)
from talentid.training.train_fusion import (
    FusionTrainingConfig,
    prepare_fusion_data,
    train_model,
)


def test_initial_gates_follow_reliability_and_mask_missing_towers() -> None:
    torch.manual_seed(7)
    model = ReliabilityAwareFusion(
        input_dimension=3,
        output_dimension=2,
        quality_dimension=6,
        source_count=5,
        dropout=0.0,
    ).eval()
    towers = torch.randn(1, 5, 3)
    quality = torch.zeros(1, 5, 6)
    quality[0, 0, 0] = 0.8
    quality[0, 1, 0] = 0.2
    availability = torch.tensor([[True, True, False, False, False]])
    candidates, gates = model.encode_candidates(towers, quality, availability)

    np.testing.assert_allclose(gates.detach().numpy()[0, :2], [0.8, 0.2], rtol=1e-5)
    assert torch.count_nonzero(gates[0, 2:]) == 0
    np.testing.assert_allclose(
        torch.linalg.vector_norm(candidates, dim=1).detach().numpy(),
        [1.0],
        rtol=1e-5,
    )


def test_weighted_contrastive_loss_rewards_better_positive_alignment() -> None:
    candidate = torch.tensor([[1.0, 0.0]])
    positive_good = torch.tensor([[1.0, 0.0]])
    positive_bad = torch.tensor([[0.0, 1.0]])
    negatives = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]])
    weights = torch.ones(1)
    good_loss = weighted_contrastive_loss(
        candidate, positive_good, negatives, weights, 0.1
    )
    bad_loss = weighted_contrastive_loss(
        candidate, positive_bad, negatives, weights, 0.1
    )
    assert good_loss < bad_loss


def mini_tables() -> tuple[
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
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
    tower_vectors = []
    tower_index_rows = []
    feature_rows = []
    for candidate_id, base in (("c_train", [1, 0, 0]), ("c_val", [0, 1, 0])):
        for source_number, source in enumerate(sources):
            available = not (candidate_id == "c_val" and source == "courses")
            tower_vectors.append(base if available else [0, 0, 0])
            tower_index_rows.append(
                {
                    "embedding_row": len(tower_index_rows),
                    "candidate_id": candidate_id,
                    "source": source,
                    "is_available": available,
                }
            )
            feature_rows.append(
                {
                    "candidate_id": candidate_id,
                    "source": source,
                    "is_available": available,
                    "reliability_score": 0.8 if available else 0.0,
                    "age_days": 10 if available else pd.NA,
                    "item_count": source_number + 1 if available else 0,
                    "is_stale": False,
                    "is_high_volume": False,
                }
            )
    candidate_index = pd.DataFrame(
        {
            "embedding_row": [0, 1],
            "candidate_id": ["c_train", "c_val"],
            "split": ["train", "validation"],
        }
    )
    skill_vectors = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float32
    )
    skill_index = pd.DataFrame(
        {"embedding_row": range(4), "skill_id": ["s1", "s2", "s3", "s4"]}
    )
    positives = pd.DataFrame(
        {
            "candidate_id": ["c_train", "c_val"],
            "skill_id": ["s1", "s2"],
            "split": ["train", "validation"],
            "sample_weight": [1.0, 0.8],
            "relevance_score": [1.0, 0.8],
        }
    )
    negatives = pd.DataFrame(
        {
            "candidate_id": ["c_train", "c_train"],
            "skill_id": ["s3", "s4"],
            "split": ["train", "train"],
            "negative_type": ["hard_retrieval", "random"],
        }
    )
    return (
        np.asarray(tower_vectors, dtype=np.float32),
        pd.DataFrame(tower_index_rows),
        candidate_index,
        pd.DataFrame(feature_rows),
        skill_vectors,
        skill_index,
        positives,
        negatives,
    )


def test_prepare_fusion_data_aligns_sources_and_quality_features() -> None:
    config = FusionTrainingConfig(
        input_dimension=3,
        output_dimension=2,
        epochs=1,
        batch_size=2,
        hard_negatives_per_example=1,
        random_negatives_per_example=1,
        validation_cutoff=2,
    )
    data = prepare_fusion_data(*mini_tables(), config)
    assert data.tower_embeddings.shape == (2, 5, 3)
    assert data.quality_features.shape == (2, 5, 6)
    assert data.availability.shape == (2, 5)
    assert not data.availability[1, 3]
    assert np.count_nonzero(data.quality_features[1, 3]) == 0
    assert data.candidate_splits == ["train", "validation"]


def test_prepare_fusion_data_rejects_test_labels() -> None:
    tables = list(mini_tables())
    test_row = tables[6].iloc[[0]].copy()
    test_row["candidate_id"] = "c_val"
    test_row["split"] = "test"
    tables[6] = pd.concat([tables[6], test_row], ignore_index=True)
    config = FusionTrainingConfig(input_dimension=3, output_dimension=2)
    with pytest.raises(ValueError, match="sealed test split"):
        prepare_fusion_data(*tables, config)


def test_one_epoch_training_smoke_test() -> None:
    config = FusionTrainingConfig(
        input_dimension=3,
        output_dimension=2,
        gate_hidden_dimension=4,
        dropout=0.0,
        epochs=1,
        batch_size=1,
        hard_negatives_per_example=1,
        random_negatives_per_example=1,
        validation_cutoff=2,
        query_batch_size=1,
        device="cpu",
    )
    data = prepare_fusion_data(*mini_tables(), config)
    model, history, device = train_model(data, config)
    assert isinstance(model, ReliabilityAwareFusion)
    assert str(device) == "cpu"
    assert history["epoch"].tolist() == [0, 1]
    assert history["is_best"].sum() == 1
